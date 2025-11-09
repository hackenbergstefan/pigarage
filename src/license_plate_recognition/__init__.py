import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Thread
from typing import Literal

import cv2
import numpy as np
import paho.mqtt.client as mqtt
import pytesseract
from libcamera import Transform
from picamera2 import Picamera2

from .diff_detector import DifferenceDetector
from .motion_detector import MotionDetector
from .ocr_detector import OcrDetector
from .plate_detector import PlateDetector

Picamera2.set_logging(logging.ERROR)


def sleep_until(until: float):
    while time.time() < until:
        time.sleep(0.5)


@contextmanager
def run_for(seconds: float):
    end_time = time.time() + seconds
    while time.time() < end_time:
        yield


@dataclass
class Point:
    x: float
    y: float


@dataclass
class Rect:
    x: float
    y: float
    w: float
    h: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.w, self.h)

    def area(self) -> float:
        return self.w * self.h

    def center(self) -> Point:
        return Point(self.x + self.w / 2, self.y + self.h / 2)

    def crop_img(self, img: cv2.typing.MatLike) -> cv2.typing.MatLike:
        return img[
            self.y : self.y + self.h,
            self.x : self.x + self.w,
        ]


class Runner:
    def __init__(
        self,
        diff_detector: DifferenceDetector,
        motion_detector: MotionDetector,
        plate_detector: PlateDetector,
        ocr_detector: OcrDetector,
        mqtt_client: mqtt.Client,
    ):
        self.diff_detector = diff_detector
        self.motion_detector = motion_detector
        self.plate_detector = plate_detector
        self.ocr_detector = ocr_detector
        self.mqtt_client = mqtt_client

    def run(self):
        self.diff_detector.start()
        self.motion_detector.start_paused()

        while True:
            self.diff_detector.next_detected()
            motion_direction = self.motion_detector.next_detected(timeout=15.0)
            if not motion_direction:
                continue

            if self.plate_detector is None:
                continue

            plate = self.plate_detector.next_detected(timeout=15.0)
            if not plate:
                continue

            if self.ocr_detector is None:
                continue
            # Detect plates for the next seconds...
            with run_for(15):
                ocr = self.ocr_detector.process(plate)
                if ocr:
                    logging.getLogger(__name__).info(f"OCR: {ocr}")
                    if self.mqtt_client is not None:
                        self.mqtt_client.publish(
                            "garage/plate",
                            f"{motion_direction} {ocr}",
                            qos=1,
                        )
                plate = self.plate_detector.detected.get(timeout=1)


def main():
    logging.basicConfig(level=logging.DEBUG)
    cam = Picamera2()
    config = cam.create_still_configuration(
        main={"size": (2592, 1944), "format": "RGB888"},
        lores={"size": (480, 360), "format": "RGB888"},
        transform=Transform(hflip=True, vflip=True),
    )
    cam.configure(config)
    cam.start()

    # mqtt_client = mqtt.Client()
    # mqtt_client.connect("localhost", 1883, 60)

    diff_detector = DifferenceDetector(cam, cam_setting="lowres")
    motion_detector = MotionDetector(cam, cam_setting="lowres")
    plate_detector = PlateDetector(cam, cam_setting="main", debug=True)
    ocr_detector = OcrDetector(debug=True)

    runner = Runner(
        diff_detector=diff_detector,
        motion_detector=motion_detector,
        plate_detector=None,
        ocr_detector=None,
        mqtt_client=None,
    )
    runner.run()


if __name__ == "__main__":
    main()
