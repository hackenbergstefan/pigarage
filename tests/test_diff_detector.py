import time
from threading import Thread
from unittest.mock import MagicMock

import numpy as np
from pigarage.diff_detector import DifferenceDetector


def test_no_diff_detected():
    mock = MagicMock()
    mock.capture_buffer.return_value = np.array([[1, 2], [3, 4]])
    detector = DifferenceDetector(cam=mock)
    detector.start()
    time.sleep(0.3)
    assert mock.capture_buffer.call_count > 2


def test_diff_detected():
    mock = MagicMock()
    notification = MagicMock()
    mock.capture_buffer.side_effect = [
        np.array([[1, 2], [3, 4]]),
        np.array([[1, 2], [3, 5]]),
    ]
    detector = DifferenceDetector(cam=mock, threshold=0.1, on_notifying=notification)
    Thread(target=lambda: time.sleep(0.3) or detector.start()).start()
    detector.wait()
    assert mock.capture_buffer.call_count == 2
    assert detector._paused is True
    assert notification.call_count == 1
