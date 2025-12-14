import logging
import time
from collections.abc import Callable
from pathlib import Path
from queue import Queue

import cv2
import ultralytics
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

try:
    from picamera2 import Picamera2
except ImportError:
    from unittest.mock import MagicMock

    Picamera2 = MagicMock()

from .config import config as pigarage_config
from .util import PausableNotifingThread


def increment_path_exists_ok(
    path: str | Path,
    *,
    exist_ok: bool = False,  # noqa: ARG001
    sep: str = "",  # noqa: ARG001
    mkdir: bool = False,  # noqa: ARG001
) -> Path:
    return path


ultralytics.utils.files.increment_path = increment_path_exists_ok
ultralytics.utils.LOGGER.setLevel(logging.WARNING)


class PlateDetector(PausableNotifingThread):
    def __init__(  # noqa: PLR0913
        self,
        cam: Picamera2,
        cam_setting: str = "main",
        on_resume: Callable[[], None] = lambda: None,
        on_notifying: Callable[[], None] = lambda: None,
        direction_min_distance: int = 50,
        *,
        debug: bool = False,
    ) -> None:
        super().__init__(on_resume=on_resume, on_notifying=on_notifying)
        self.model = YOLO(
            hf_hub_download(
                "morsetechlab/yolov11-license-plate-detection",
                "license-plate-finetune-v1n.pt",
            )
        )
        self._cam = cam
        self._cam_setting = cam_setting
        self._debug = debug
        self.detected_plates = Queue(maxsize=0)
        self.detected_directions = Queue(maxsize=0)
        self._previous = None
        self._direction_min_distance = direction_min_distance

    def resume(self) -> None:
        while self.detected_plates.qsize() > 0:
            self.detected_plates.get_nowait()
        while self.detected_directions.qsize() > 0:
            self.detected_directions.get_nowait()
        self._previous = None
        return super().resume()

    def process(self) -> None:
        img = self._cam.capture_array(self._cam_setting)
        results = self.model.predict(
            source=img,
            verbose=False,
            save=self._debug,
            save_crop=self._debug,
            project="/tmp",  # noqa: S108
            name="plate_detector",
            imgsz=512,
        )
        center_y = None
        if results[0].boxes:
            x1, y1, x2, y2 = map(int, results[0].boxes[0].xyxy[0].tolist())
            if self._debug:
                cv2.imwrite(
                    pigarage_config.logdir
                    / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_plate.jpg",
                    cv2.rectangle(
                        img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=3
                    ),
                )

            center_y = (y1 + y2) / 2
            self._log.debug(f"Plate found {center_y} {img.shape}")
            if self._previous is None:
                self._previous = center_y
            self.detected_plates.put(img[y1:y2, x1:x2])
            self._notify_waiters()

        if center_y is not None:
            diff = center_y - self._previous
            if abs(diff) < self._direction_min_distance:
                return

            direction = "arriving" if diff > 0 else "leaving"
            self._log.debug(f"direction: {direction}")
            self.detected_directions.put(direction)
            self._notify_waiters()
