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

from . import Runner
from .car_detector import CarDetector
from .diff_detector import DifferenceDetector
from .motion_detector import MotionDetector
from .ocr_detector import OcrDetector
from .plate_detector import PlateDetector


def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    cam = Picamera2()
    config = cam.create_still_configuration(
        main={"size": (2592, 1944), "format": "RGB888"},
        lores={"size": (480, 360)},
        # transform=Transform(hflip=True, vflip=True),
    )
    cam.configure(config)
    cam.start()

    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    mqtt_client.username_pw_set(username="broker", password="123")
    mqtt_client.connect("192.168.1.4")

    diff_detector = DifferenceDetector(cam, cam_setting="lores", threshold=5)
    car_detector = CarDetector(cam, cam_setting="main", debug=True)
    plate_detector = PlateDetector(cam, cam_setting="main", debug=True)
    ocr_detector = OcrDetector(debug=True)

    runner = Runner(
        diff_detector=diff_detector,
        car_detector=car_detector,
        plate_detector=plate_detector,
        ocr_detector=ocr_detector,
        mqtt_client=mqtt_client,
    )
    runner.run()


if __name__ == "__main__":
    main()
