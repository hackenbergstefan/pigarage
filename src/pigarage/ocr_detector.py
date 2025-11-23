import logging
import re

import cv2
import numpy as np
import pytesseract


def cv2_mask_non_plate(plate):
    # Find contour with biggest area
    plate = cv2.GaussianBlur(plate, ksize=(3, 3), sigmaX=1.0)
    for threshold in (100, 50, 150, 200, 250):
        _, plate2 = cv2.threshold(plate, threshold, 255, cv2.THRESH_BINARY)
        if np.max(plate2) - np.min(plate2) > 0:
            plate = plate2
            break
    contours, _ = cv2.findContours(plate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)
    if len(contours) == 0:
        return plate, None
    contour = contours[0]

    # Make everything else white
    mask = np.zeros(plate.shape).astype(plate.dtype)
    mask = cv2.drawContours(mask, [contour], 0, (255, 255, 255), thickness=cv2.FILLED)
    plate = cv2.bitwise_or(plate, 255 - mask)
    return plate, contour


def cv2_fix_perspective(plate, contour):
    # Get rotated bounding rect of contour
    rect = cv2.minAreaRect(contour)
    (_rect_x, _rect_y), (rect_width, rect_height), _rect_angle = rect
    box = np.uint(cv2.boxPoints(rect))
    # Calculate transformation matrix
    aspect = rect_height / rect_width
    if aspect > 1.0:
        aspect = rect_width / rect_height
    img_h, img_w = plate.shape[:2]
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
    plate = cv2.warpPerspective(
        plate,
        mat,
        dsize=(new_w, new_h),
    )
    return plate


class OcrDetector:
    def __init__(self, *, debug=False):
        self.debug = debug

    def postprocess(self, ocr: str) -> str:
        ocr = re.search("[A-Z]{1,2}\.? ?\.?[A-Z]{0,2} ?[0-9]{2,4}$", ocr)
        if ocr:
            return ocr.group(0).replace(" ", "").replace(".", "")
        return None

    def _improve_image(self, plate: cv2.typing.MatLike) -> None | cv2.typing.MatLike:
        plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        plate, plate_contour = cv2_mask_non_plate(plate)
        if plate_contour is not None:
            plate = cv2_fix_perspective(plate, plate_contour)
        return plate

    def process(self, plate: cv2.typing.MatLike) -> None | cv2.typing.MatLike:
        plate = self._improve_image(plate)
        cv2.imwrite("/tmp/ocr.jpg", plate)
        result = pytesseract.image_to_string(
            plate,
            config="--psm 13 -c tessedit_char_whitelist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ .'",
        )
        logging.getLogger(__name__).debug(result)
        return self.postprocess(result)
