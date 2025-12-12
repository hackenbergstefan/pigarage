import time
from queue import Queue
from threading import Thread

import cv2
import numpy as np
import pytest
from pigarage import ocr_detector

from . import download_lnpr_plate

VISUAL_DEBUG = False


@pytest.mark.parametrize(
    "id, expected_threshold, expected_std",
    [
        ("08HdV8ArxuVKXgxdUor1", 55, 0.003),
        ("DmbceUxkj0VtBqPK29wG", 65, 0.02),
        ("qTPu96zhef7AbiBSFFTD", 150, 0.01),
    ],
)
def test_improve_plate_img(id, expected_threshold, expected_std):
    img = download_lnpr_plate(id)
    plate, contour, threshold, std = ocr_detector.cv2_improve_plate_img(img)
    if VISUAL_DEBUG:
        plate = cv2.drawContours(plate, [contour], -1, (0, 255, 0), 2)
        cv2.imshow(
            "Improved Plate",
            np.vstack([plate, ocr_detector.cv2_fix_perspective(plate, contour)]),
        )
        cv2.waitKey(0)
    assert threshold == expected_threshold
    assert std < expected_std


@pytest.mark.parametrize(
    "id, expected_ocr",
    [
        ("08HdV8ArxuVKXgxdUor1", "KSC124"),
        ("uJsY6e391eOodCkFLMJA", "A777AK77"),
        ("qTPu96zhef7AbiBSFFTD", "R275ULO"),
    ],
)
def test_ocr(id, expected_ocr):
    img = download_lnpr_plate(id)
    plate, contour, _, _ = ocr_detector.cv2_improve_plate_img(img)
    plate = ocr_detector.cv2_fix_perspective(plate, contour)
    ocr = ocr_detector.plate2text(plate).replace(" ", "")
    assert expected_ocr in ocr


@pytest.mark.parametrize(
    "id, expected_ocr",
    [
        ("08HdV8ArxuVKXgxdUor1", ["KSC124", "KKSC124"]),
        ("uJsY6e391eOodCkFLMJA", ["A777AK77"]),
        ("qTPu96zhef7AbiBSFFTD", ["R275ULO"]),
    ],
)
def test_ocr_detector(id, expected_ocr):
    detected_plates = Queue(maxsize=0)
    detected_plates.put(download_lnpr_plate(id))
    detector = ocr_detector.OcrDetector(
        detected_plates=detected_plates,
        allowed_plates=expected_ocr,
        ocr_regex=".*",
    )
    Thread(target=lambda: time.sleep(0.3) or detector.start()).start()
    detector.wait()
    assert detector.detected_ocrs.get_nowait() in expected_ocr
