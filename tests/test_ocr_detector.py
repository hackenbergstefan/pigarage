from queue import Queue
from urllib.request import urlopen

import cv2
import numpy as np
from pigarage.ocr_detector import OcrDetector


def test_ocr_detector():
    img = cv2.imdecode(
        np.asarray(
            bytearray(
                urlopen(
                    "https://source.roboflow.com/zD7y6XOoQnh7WC160Ae7/08HdV8ArxuVKXgxdUor1/original.jpg"
                ).read()
            ),
            dtype=np.uint8,
        ),
        -1,
    )
    plate_center = (545, 654)
    plate_size = (170, 866)
    img = img[
        plate_center[0] - plate_size[0] // 2 : plate_center[0] + plate_size[0] // 2,
        plate_center[1] - plate_size[1] // 2 : plate_center[1] + plate_size[1] // 2,
    ]

    plates = Queue(maxsize=0)
    plates.put(img)
    detector = OcrDetector(
        detected_plates=plates,
        allowed_plates=["K3SC124"],
        ocr_regex=".*",
    )
    detector.start()
    detector.wait()
    assert detector.detected_ocrs.get_nowait() == "K3SC124"
