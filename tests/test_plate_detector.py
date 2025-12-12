import time
from threading import Thread
from unittest.mock import MagicMock
from urllib.request import urlopen

import cv2
import numpy as np
from pigarage.plate_detector import PlateDetector


def test_no_plate_detected():
    mock = MagicMock()
    mock.capture_array.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    detector = PlateDetector(cam=mock)
    detector.start()
    detector.wait(timeout=1)
    detector.pause()
    assert mock.capture_array.call_count > 2


def test_plate_detected():
    img = cv2.imdecode(
        np.asarray(
            bytearray(
                urlopen(
                    "https://source.roboflow.com/zD7y6XOoQnh7WC160Ae7/00T2dhevWWZv5lzFbvfU/original.jpg"
                ).read()
            ),
            dtype=np.uint8,
        ),
        -1,
    )
    mock = MagicMock()
    mock.capture_array.return_value = img
    detector = PlateDetector(cam=mock)
    detector._on_notifying = lambda: detector.pause()
    Thread(target=lambda: time.sleep(0.3) or detector.start()).start()
    detector.wait()
    assert mock.capture_array.call_count == 1
    assert detector.detected_plates.qsize() == 1


def test_direction_detected():
    img = cv2.imdecode(
        np.asarray(
            bytearray(
                urlopen(
                    "https://source.roboflow.com/zD7y6XOoQnh7WC160Ae7/00T2dhevWWZv5lzFbvfU/original.jpg"
                ).read()
            ),
            dtype=np.uint8,
        ),
        -1,
    )
    imgs = [
        np.concatenate([np.zeros((20 * i, img.shape[1], img.shape[2])), img], axis=0)
        for i in range(6)
    ]
    mock = MagicMock()
    mock.capture_array.side_effect = imgs
    detector = PlateDetector(cam=mock)
    detector._on_notifying = (
        lambda: detector.detected_directions.qsize() > 0 and detector.pause()
    )
    Thread(target=lambda: time.sleep(0.3) or detector.start()).start()
    for _ in range(5):
        detector.wait()
    assert detector.detected_directions.get_nowait() == "arriving"
