import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from queue import Empty
from threading import Thread
from typing import Literal

import cv2
import numpy as np
import paho.mqtt.client as mqtt
import pytesseract
from libcamera import Transform
from picamera2 import Picamera2

from .car_detector import CarDetector
from .diff_detector import DifferenceDetector
from .ocr_detector import OcrDetector
from .plate_detector import PlateDetector

Picamera2.set_logging(logging.ERROR)


def sleep_until(until: float):
    while time.time() < until:
        time.sleep(0.5)


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
        car_detector: CarDetector,
        plate_detector: PlateDetector,
        ocr_detector: OcrDetector,
        mqtt_client: mqtt.Client,
    ):
        self.diff_detector = diff_detector
        self.car_detector = car_detector
        self.plate_detector = plate_detector
        self.ocr_detector = ocr_detector
        self.mqtt_client = mqtt_client

    def run(self):
        self.diff_detector.start()
        # if self.car_detector:
        #     self.car_detector.start_paused()
        if self.plate_detector:
            self.plate_detector.start_paused()

        while True:
            self.diff_detector.next_detected()

            # if self.car_detector is None:
            #     continue

            # result = self.car_detector.next_detected(timeout=10.0)
            # if result is None:
            #     continue
            # motion_direction, _ = result

            if self.plate_detector is None:
                continue
            result = self.plate_detector.next_detected(timeout=10.0)
            if result is None:
                continue
            plate, motion_direction = result

            if self.ocr_detector is None:
                continue

            # Detect plates for the next seconds...
            self.plate_detector.resume()
            end_time = time.time() + 5
            while time.time() < end_time:
                ocr = self.ocr_detector.process(plate)
                if ocr:
                    logging.getLogger(__name__).info(f"OCR: {ocr}")
                    if self.mqtt_client is not None:
                        message = f"{motion_direction} {ocr}"
                        logging.getLogger(__name__).info(f"MQTT publish: '{message}'")
                        self.mqtt_client.publish("garage/plate", message, qos=1)
                try:
                    plate, motion_direction = self.plate_detector.detected.get(
                        timeout=1
                    )
                except Empty:
                    break
            self.plate_detector.pause()
