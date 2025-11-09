import re

import cv2
import pytesseract


class OcrDetector:
    def __init__(self, *, debug=False):
        self.debug = debug

    def postprocess(self, ocr: str) -> str:
        ocr = re.search("A [A-Z]{0,2}[0-9]{3,4}$", ocr)
        if ocr:
            return ocr.group(0)
        return None

    def process(self, plate: cv2.typing.MatLike) -> None | cv2.typing.MatLike:
        plate = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY)
        result = pytesseract.image_to_string(
            plate,
            config="--psm 13 -c tessedit_char_whitelist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ .'",
        )
        return result
