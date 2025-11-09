import logging
from datetime import time
from queue import Queue

import cv2
from huggingface_hub import hf_hub_download
from picamera2 import Picamera2
from ultralytics import YOLO

from .util import DetectionThread


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

    def process(self, img: cv2.typing.MatLike) -> None | cv2.typing.MatLike:
        results = self.model.predict(source=img, verbose=False)
        if results[0].boxes:
            x1, y1, x2, y2 = map(int, results[0].boxes[0].xyxy[0].tolist())
            logging.getLogger(__name__).debug("Plate found")
            plate = img[y1:y2, x1:x2]
            if self.debug:
                cv2.imwrite(f"plate_{time.strftime('%Y-%m-%d_%H-%M-%S')}.jpg", plate)
            return plate
        return None
