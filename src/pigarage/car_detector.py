import logging
from pathlib import Path

import cv2
import ultralytics.utils.files
from picamera2 import Picamera2
from ultralytics import YOLO

from .util import DetectionThread

"""
DEPRECATED
"""


def increment_path_exists_ok(
    path: str | Path, exist_ok: bool = False, sep: str = "", mkdir: bool = False
) -> Path:
    return path


ultralytics.utils.files.increment_path = increment_path_exists_ok


class CarDetector(DetectionThread):
    def __init__(
        self,
        cam: Picamera2,
        cam_setting="main",
        *,
        debug=False,
    ):
        super().__init__(cam=cam, cam_setting=cam_setting)
        self.model = YOLO("yolo11n.pt")
        self.debug = debug
        # self.detected = Queue(maxsize=1)
        self.history = []
        self.history_length = 4

    def process(self, img: cv2.typing.MatLike) -> None | cv2.typing.MatLike:
        results = self.model.predict(
            source=img,
            verbose=False,
            save=self.debug,
            project="/tmp",
            name="car_detector",
            # classes=[2],  # Only detect cars
            # imgsz=256,
        )
        if results[0].boxes:
            x1, y1, x2, y2 = map(int, results[0].boxes[0].xyxy[0].tolist())
            self.history.append((y1 + y2) / 2)

        if len(self.history) == self.history_length:
            ys = sorted(range(self.history_length), key=lambda i: self.history[i])
            print(ys)
            self.history.clear()

            if abs(ys[0] - ys[-1]) < 2:
                return None

            car = img[y1:y2, x1:x2]
            if ys == list(range(self.history_length)):
                direction = "arriving"
                logging.getLogger(__name__).debug(f"direction: {direction}")
                return direction, car
            if ys == list(reversed(range(self.history_length))):
                direction = "leaving"
                logging.getLogger(__name__).debug(f"direction: {direction}")
                return direction, car

        # logging.getLogger(__name__).debug("no car found")
        return None
