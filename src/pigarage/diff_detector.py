import logging
import time

import cv2
import numpy as np
from picamera2 import Picamera2

from .util import DetectionThread


class DifferenceDetector(DetectionThread):
    def __init__(
        self,
        cam: Picamera2,
        cam_setting="lores",
        threshold: float = 10.0,
    ):
        super().__init__(cam=cam, cam_setting=cam_setting)

        self.threshold = threshold
        self._previous = None

    def process(self, img: cv2.typing.MatLike) -> bool:
        if self._previous is None:
            self._previous = img
            return None
        mse = np.square(np.subtract(self._previous, img)).mean()
        self._previous = img
        if mse > self.threshold:
            self.last_motion = time.time()
            logging.getLogger(__name__).debug(f"motion detected. mse: {mse}")
            return True
        return None
