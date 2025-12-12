import logging
import re
import time
from queue import Queue

import cv2
import numpy as np
import pytesseract

from .config import config as pigarage_config
from .util import PausableNotifingThread


def cv2_mask_non_plate(
    plate: cv2.typing.MatLike,
    threshold: int,
    min_contours: int = 4,
    min_char_area: float = 0.01,
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
    idxs = np.argsort(areas)[::-1]
    # Normalize the areas to the plate area
    areas = [a / areas[idxs[0]] for a in areas]

    # Check if the second largest contour is on level 0 (second level)
    char_level = hierarchy[0, idxs[1], 3]
    if char_level != 0:
        return None

    # Check that the average area of characters is large enough
    symbols = areas[hierarchy[0, :, 3] == char_level]
    if np.mean(symbols) < min_char_area:
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
    return np.std(symbols), mask, contours[idxs[0]]


def cv2_fix_perspective(
    plate: cv2.typing.MatLike,
    contour: np.ndarray,
) -> cv2.typing.MatLike:
    # Get rotated bounding rect of contour
    rect = cv2.minAreaRect(contour)
    (_rect_x, _rect_y), (rect_width, rect_height), _rect_angle = rect
    box = np.uint(cv2.boxPoints(rect))
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

    def _improve_image(self, plate: cv2.typing.MatLike) -> None | cv2.typing.MatLike:
        plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        preprocessed = [
            p
            for threshold in range(0, 255, 5)
            if (p := cv2_mask_non_plate(plate, threshold)) is not None
        ]
        if len(preprocessed) == 0:
            return None

        _, plate, plate_contour = sorted(
            preprocessed,
            key=lambda p: p[0],
            reverse=True,
        )[0]
        return cv2_fix_perspective(plate, plate_contour)

    def _ocr(self, plate: cv2.typing.MatLike) -> str:
        return pytesseract.image_to_string(
            plate,
            config="--psm 13 "
            "-c tessedit_char_whitelist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ .'",
        )

    def process(self) -> None:
        plate = self._detected_plates.get()

        if self._debug:
            cv2.imwrite(
                pigarage_config.logdir
                / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_ocr_pre.jpg",
                plate,
            )
        plate = self._improve_image(plate)
        if self._debug:
            cv2.imwrite(
                pigarage_config.logdir
                / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_ocr_post.jpg",
                plate,
            )
        result = self._ocr(plate)
        ocr = self._postprocess(result)
        logging.getLogger(__name__).info(f"OCR: '{result.strip()}' -> '{ocr}'")
        if ocr is not None and ocr in self.allowed_plates:
            self.detected_ocrs.put(ocr)
            self._notify_waiters()
            self.pause()
