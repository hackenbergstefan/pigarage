from collections.abc import Callable

import cv2
import numpy as np

try:
    from picamera2 import Picamera2
except ImportError:
    from unittest.mock import MagicMock

    Picamera2 = MagicMock()

from .util import PausableNotifingThread


def mse(img1: cv2.typing.MatLike, img2: cv2.typing.MatLike) -> float:
    """Compute Mean Squared Error between two images."""
    return np.sum((img1.astype(np.float32) - img2.astype(np.float32)) ** 2) / (
        img1.shape[0] * img1.shape[1]
    )


class DifferenceDetector(PausableNotifingThread):
    def __init__(
        self,
        cam: Picamera2,
        cam_setting: str = "lores",
        threshold: float = 1000.0,
        on_resume: Callable[[], None] = lambda: None,
        on_notifying: Callable[[], None] = lambda: None,
    ) -> None:
        super().__init__(on_resume=on_resume, on_notifying=on_notifying)
        self.cam = cam
        self.cam_setting = cam_setting
        self.threshold = threshold
        self._previous = None

    def pause(self) -> None:
        self._previous = None
        return super().pause()

    def process(self) -> None:
        """
        Capture new image and compare with previous.
        If differing sufficiently, notify and pause.
        """  # noqa: D205
        img = self.cam.capture_buffer(self.cam_setting)

        if self._previous is None:
            self._previous = img
            return
        mse_value = mse(self._previous, img)
        self._previous = img
        if mse_value > self.threshold:
            self._log.debug(f"motion detected. mse: {mse_value}")
            self._notify_waiters()
            self.pause()
