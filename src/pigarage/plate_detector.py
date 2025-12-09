import logging
from pathlib import Path
from queue import Queue

import cv2
import ultralytics
from huggingface_hub import hf_hub_download
from picamera2 import Picamera2
from ultralytics import YOLO

from .util import DetectionThread


def increment_path_exists_ok(
    path: str | Path, exist_ok: bool = False, sep: str = "", mkdir: bool = False
) -> Path:
    return path


ultralytics.utils.files.increment_path = increment_path_exists_ok


class PlateDetector(DetectionThread):
    def __init__(
        self,
        cam: Picamera2,
        cam_setting="main",
        *,
        debug=False,
    ):
        super().__init__(cam=cam, cam_setting=cam_setting)
        self.model = YOLO(
            hf_hub_download(
                "morsetechlab/yolov11-license-plate-detection",
                "license-plate-finetune-v1n.pt",
            )
        )
        self.debug = debug
        self.detected = Queue(maxsize=0)
        self.history = []
        self.history_length = 4

    def process(self, img: cv2.typing.MatLike) -> None | cv2.typing.MatLike:
        results = self.model.predict(
            source=img,
            verbose=False,
            save=self.debug,
            save_crop=self.debug,
            project="/tmp",
            name="plate_detector",
            imgsz=512,
        )
        plate = None
        if results[0].boxes:
            x1, y1, x2, y2 = map(int, results[0].boxes[0].xyxy[0].tolist())
            self.history.append((y1 + y2) / 2)
            logging.getLogger(__name__).debug("Plate found")
            plate = img[y1:y2, x1:x2]

        if len(self.history) == self.history_length:
            ys = sorted(range(self.history_length), key=lambda i: self.history[i])
            history_diff = abs(self.history[0] - self.history[-1])
            self.history.clear()
            if history_diff < 50:
                return None

            if ys == list(range(self.history_length)):
                direction = "arriving"
                logging.getLogger(__name__).debug(f"direction: {direction}")
                return plate, direction
            if ys == list(reversed(range(self.history_length))):
                direction = "leaving"
                logging.getLogger(__name__).debug(f"direction: {direction}")
                return plate, direction

        return None
