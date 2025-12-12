import time
from threading import Thread
from unittest.mock import MagicMock

import numpy as np
import pytest
from pigarage.plate_detector import PlateDetector

from . import download_lnpr_image


def test_no_plate_detected():
    mock = MagicMock()
    mock.capture_array.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    detector = PlateDetector(cam=mock)
    Thread(target=lambda: time.sleep(0.3) or detector.start()).start()
    detector.wait(timeout=2)
    detector.pause()
    assert mock.capture_array.call_count >= 1


@pytest.mark.parametrize(
    "id",
    ["08HdV8ArxuVKXgxdUor1", "uJsY6e391eOodCkFLMJA", "qTPu96zhef7AbiBSFFTD"],
)
def test_plate_detected(id):
    img = download_lnpr_image(id)
    mock = MagicMock()
    mock.capture_array.return_value = img
    detector = PlateDetector(cam=mock)
    detector._on_notifying = lambda: detector.pause()
    Thread(target=lambda: time.sleep(0.3) or detector.start()).start()
    detector.wait()
    assert mock.capture_array.call_count == 1
    assert detector.detected_plates.qsize() == 1


def test_direction_detected():
    img = download_lnpr_image("uJsY6e391eOodCkFLMJA")
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
    for _ in range(4):
        detector.wait()
    assert detector.detected_directions.get_nowait() == "arriving"
