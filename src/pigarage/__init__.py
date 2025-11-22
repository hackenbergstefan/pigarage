import logging
import time
from queue import Empty

import paho.mqtt.client as mqtt
from picamera2 import Picamera2

from .diff_detector import DifferenceDetector
from .ocr_detector import OcrDetector
from .plate_detector import PlateDetector

Picamera2.set_logging(logging.ERROR)


class Runner:
    def __init__(
        self,
        diff_detector: DifferenceDetector,
        plate_detector: PlateDetector,
        ocr_detector: OcrDetector,
        mqtt_client: mqtt.Client,
    ):
        self.diff_detector = diff_detector
        self.plate_detector = plate_detector
        self.ocr_detector = ocr_detector
        self.mqtt_client = mqtt_client

    def run(self):
        self.diff_detector.start()
        if self.plate_detector:
            self.plate_detector.start_paused()

        while True:
            # Wait for next diff ...
            self.diff_detector.next_detected()

            if self.plate_detector is None:
                continue

            # Wait for next plate ...
            result = self.plate_detector.next_detected(timeout=10.0)
            if result is None:
                continue
            plate, motion_direction = result

            if self.ocr_detector is None:
                continue

            # Detect plates and try ocr for the next seconds...
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
