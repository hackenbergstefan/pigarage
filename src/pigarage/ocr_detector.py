import logging
import re
import time
from queue import Queue

import cv2
import numpy as np
import pytesseract

from .config import VISUAL_DEBUG
from .config import config as pigarage_config
from .util import PausableNotifingThread


def put_text(plate: cv2.typing.MatLike, text: list[str]) -> cv2.typing.MatLike:
    extend = cv2.bitwise_not(
        np.zeros((plate.shape[0], 200, plate.shape[2])).astype(plate.dtype)
    )
    plate = np.hstack([extend, plate])
    for i, line in enumerate(text):
        plate = cv2.putText(
            plate,
            line,
            org=(0, 10 + i * 15),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(255, 0, 0),
            thickness=1,
        )
    return plate


def cv2_mask_non_plate(  # noqa: PLR0913
    plate: cv2.typing.MatLike,
    threshold: int,
    min_contours: int = 4,
    min_plate_area: float = 0.5,
    min_symbol_height: float = 0.3,
    min_symbols: int = 4,
) -> tuple[float, cv2.typing.MatLike, np.ndarray] | None:
    # Find contours in the plate image using threshold
    plate_bw = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    plate_bw = cv2.GaussianBlur(plate_bw, ksize=(5, 5), sigmaX=3.0)
    _, plate_bw = cv2.threshold(plate_bw, threshold, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(
        plate_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    # Check if there are enough contours (characters of the plate)
    if len(contours) < min_contours:
        return None

    # Calculate areas of contours and sort them descending
    areas = np.array([cv2.contourArea(c) for c in contours])
    if np.all(areas == 0):
        return None
    idxs = np.argsort(areas)[::-1]
    # Check that plate area is large enough
    if areas[idxs[0]] / (plate.shape[0] * plate.shape[1]) < min_plate_area:
        return None

    # Check if the second largest contour is on second level
    # Each entry of hierarchy represents: [Next, Previous, First_Child, Parent]
    # where the entries are indices and Parent == -1 is no parent, i.e. top level
    # Let parent_second_max Parent of second largest contour,
    # so it is on second level if hierarchy[parent_second_max].Parent == -1
    parent_second_max = hierarchy[0, idxs[1], 3]
    if hierarchy[0, parent_second_max, 3] != -1:
        return None

    # Check amount if symbol heights that are large enough
    area_height = cv2.boundingRect(contours[idxs[0]])[3]
    symbols_with_min_height = np.array(
        [
            h
            for i in range(len(contours))
            if (h := cv2.boundingRect(contours[i])[3] / area_height)
            and hierarchy[0, i, 3] == parent_second_max
            and h > min_symbol_height
        ]
    )
    if len(symbols_with_min_height) < min_symbols:
        return None

    # Create a mask to remove non-character contours
    mask = cv2.bitwise_not(np.zeros(plate.shape).astype(plate.dtype))
    mask = cv2.drawContours(
        mask,
        [contours[i] for i in idxs[1:]],
        -1,
        color=(0, 0, 0),
        thickness=cv2.FILLED,
    )
    # Return the standard deviation of character areas,
    # the mask, and the largest contour
    return np.std(symbols_with_min_height), mask, contours[idxs[0]]


def cv2_fix_perspective(
    plate: cv2.typing.MatLike,
    contour: np.ndarray,
) -> cv2.typing.MatLike:
    # Get rotated bounding rect of contour
    rect = cv2.minAreaRect(contour)
    (_rect_x, _rect_y), (rect_width, rect_height), _rect_angle = rect
    box = np.int32(cv2.boxPoints(rect))
    # Calculate transformation matrix
    aspect = rect_height / rect_width
    if aspect > 1.0:
        aspect = rect_width / rect_height
    _img_h, img_w = plate.shape[:2]
    new_w, new_h = (img_w, int(aspect * img_w))
    if rect_width > rect_height:
        dst = np.float32(
            [
                [0.0, new_h],
                [0, 0],
                [new_w, 0.0],
                [new_w, new_h],
            ]
        )
    else:
        dst = np.float32(
            [
                [0, 0],
                [new_w, 0.0],
                [new_w, new_h],
                [0.0, new_h],
            ]
        )
    mat = cv2.getPerspectiveTransform(np.float32(box), dst)
    return cv2.warpPerspective(
        plate,
        mat,
        dsize=(new_w, new_h),
    )


def cv2_improve_plate_img(
    plate: cv2.typing.MatLike,
) -> None | tuple[cv2.typing.MatLike, cv2.typing.MatLike, int, float]:
    preprocessed = [
        (*p, threshold)
        for threshold in range(0, 255, 5)
        if (p := cv2_mask_non_plate(plate, threshold)) is not None
    ]
    if len(preprocessed) == 0:
        return None

    if VISUAL_DEBUG:
        stacked = []
        for std, p, c, threshold in preprocessed:
            p = cv2.drawContours(p, [c], -1, (0, 255, 0), 2)  # noqa: PLW2901
            p = put_text(p, [f"{threshold}", f"{std:.3e}"])  # noqa: PLW2901
            stacked.append(p)
        cv2.imshow("foo", np.vstack(stacked))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    std, plate, plate_contour, threshold = sorted(
        preprocessed,
        key=lambda p: p[0],
    )[0]
    return plate, plate_contour, threshold, std


def plate2text(plate: cv2.typing.MatLike) -> str:
    return pytesseract.image_to_string(
        plate,
        config="-l eng --oem 3 --psm 13 "
        "-c tessedit_char_whitelist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ .'",
    ).strip()


class OcrDetector(PausableNotifingThread):
    def __init__(
        self,
        detected_plates: Queue,
        allowed_plates: list[str],
        ocr_regex: str = r"[A-Z]{1,2}\.? ?\.?[A-Z]{0,2} ?[0-9]{2,4}$",
        *,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self._debug = debug
        self._detected_plates = detected_plates
        self.detected_ocrs = Queue(maxsize=1)
        self._ocr_regex = ocr_regex
        self.allowed_plates = allowed_plates

    def _postprocess(self, ocr: str) -> str:
        ocr = re.search(self._ocr_regex, ocr)
        if ocr:
            return ocr.group(0).replace(" ", "").replace(".", "")
        return None

    def process(self) -> None:
        plate = self._detected_plates.get()

        if self._debug:
            cv2.imwrite(
                pigarage_config.logdir
                / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_ocr_pre.jpg",
                plate,
            )
        plate, plate_contour, _, _ = cv2_improve_plate_img(plate)
        plate = cv2_fix_perspective(plate, plate_contour)
        if self._debug:
            cv2.imwrite(
                pigarage_config.logdir
                / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_ocr_post.jpg",
                plate,
            )
        result = plate2text(plate)
        ocr = self._postprocess(result)
        logging.getLogger(__name__).info(f"OCR: '{result.strip()}' -> '{ocr}'")
        if ocr is not None and ocr in self.allowed_plates:
            self.detected_ocrs.put(ocr)
            self._notify_waiters()
            self.pause()
