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


# class MotionDetector:
#     def __init__(
#         self,
#         cam: Picamera2,
#         detection_area: Rect,
#         diff_detector: DifferenceDetector,
#         *,
#         debug: bool = False,
#     ):
#         self.cam = cam
#         self.detection_area = detection_area
#         self.diff_detector = diff_detector
#         self.motion_tracker = ForegroundTracker()

#     def process(self, img: cv2.typing.MatLike) -> None | str:
#         cropped = self.detection_area.crop_img(img)
#         diff = self.diff_detector.update(cropped)
#         if not diff:
#             return None

#         motion = self.motion_tracker.process(img)
#         return motion


# class YoloDetector:
#     def __init__(self):
#         from huggingface_hub import hf_hub_download
#         from ultralytics import YOLO

#         self.model = YOLO(
#             hf_hub_download(
#                 "morsetechlab/yolov11-license-plate-detection",
#                 "license-plate-finetune-v1n.pt",
#             )
#         )

#     def predict(self, img: cv2.typing.MatLike) -> None | cv2.typing.MatLike:
#         results = self.model.predict(source=img, verbose=False)
#         if results[0].boxes:
#             x1, y1, x2, y2 = map(int, results[0].boxes[0].xyxy[0].tolist())
#             logging.getLogger(__name__).debug("Plate found")
#             plate = img[y1:y2, x1:x2]
#             img = cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=4)
#             cv2.imwrite(f"plate_{time.strftime('%Y-%m-%d_%H-%M-%S')}.jpg", img)
#             return plate
#         return None


# class PlateRecognizer:
#     def __init__(
#         self,
#         cam: Picamera2,
#         detector: MotionDetector,
#         mqtt_client: mqtt.Client,
#     ):
#         self.cam = cam
#         self.detector = detector
#         self.yolo = YoloDetector()
#         self.mqtt_client = mqtt_client

#     def loop_forever(self):
#         while True:
#             img = self.cam.capture_array()
#             direction = self.detector.process(img)
#             if direction is None:
#                 # time.sleep(0.1)
#                 continue
#             logging.getLogger(__name__).debug(f"direction: {direction}")
#             img = self.detector.detection_area.crop_img(img)
#             plate = self.yolo.predict(img)
#             if plate is None:
#                 continue
#             logging.getLogger(__name__).debug("Plate found")
#             if direction == "arriving":
#                 later = time.time() + 4
#                 ocr1 = self.ocr(plate)
#                 logging.getLogger(__name__).info(f"OCR1: {ocr1}")
#                 sleep_until(later)
#                 plate = self.yolo.predict(
#                     self.detector.detection_area.crop_img(self.cam.capture_array())
#                 )
#                 if plate is None:
#                     continue
#                 ocr2 = self.ocr(plate)
#                 logging.getLogger(__name__).info(f"OCR2: {ocr2}")
#                 self.mqtt_client.publish("garage/plate", f"arriving {ocr2}", qos=1)
#                 # Reset recognized plate after 1s
#                 time.sleep(1)
#                 self.mqtt_client.publish("garage/plate", None, qos=1)

#             time.sleep(2)

#     def ocr(self, plate) -> str:
#         plate = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY)
#         result = pytesseract.image_to_string(
#             plate,
#             config="--psm 13 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
#         )
#         return result


# # class Recognizer:
# #     def __init__(
# #         self,
# #         mqtt_broker: str,
# #         mqtt_user: str,
# #         mqtt_password: str,
# #         resolution=(2592, 1944),
# #         slices=(slice(0, 300), slice(300, 600)),
# #         detection_threshold=50,
# #     ):
# #         self._log = logging.getLogger(__name__)
# #         self.detection_threshold = detection_threshold
# #         self.slices = slices

# #         self.cam = Picamera2()
# #         config = self.cam.create_still_configuration(
# #             main={"size": resolution, "format": "RGB888"},
# #             # lores={"size": low_resolution},
# #             transform=Transform(hflip=True, vflip=True),
# #         )
# #         self.cam.configure(config)
# #         self.cam.start()

# #         self.model = YOLO(
# #             hf_hub_download(
# #                 "morsetechlab/yolov11-license-plate-detection",
# #                 "license-plate-finetune-v1n.pt",
# #             )
# #         )

# #         # self.ocr_reader = easyocr.Reader(["en"], gpu=False)

# #         self.mqtt = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
# #         self.mqtt.username_pw_set(username=mqtt_user, password=mqtt_password)
# #         self.mqtt.connect(mqtt_broker)

# #         self.plate_a = None
# #         self.plate_b = None

# #     def loop_forever(self):
# #         prev = None
# #         while True:
# #             # Reset plates
# #             if self.plate_a and time.time() - self.plate_a[1] > 30:
# #                 self._log.debug("Reset plate_a")
# #                 self.plate_a = None
# #             if self.plate_b and time.time() - self.plate_b[1] > 30:
# #                 self._log.debug("Reset plate_b")
# #                 self.plate_b = None

# #             # New try
# #             cur = self.cam.capture_array()
# #             if prev is None:
# #                 prev = cur
# #                 continue

# #             mse = np.square(np.subtract(cur, prev)).mean()
# #             if mse < self.detection_threshold:
# #                 self._log.debug(f"Nothing changed ({mse:.2f})...")
# #                 prev = cur
# #                 time.sleep(1)
# #                 continue

# #             # Check for license plate
# #             self._log.info(f"Change ({mse:.2f})...")
# #             cv2.imwrite(
# #                 f"img_{time.strftime('%Y-%m-%d_%H-%M-%S')}.jpg",
# #                 cur,
# #             )
# #             self.detect_and_react(cur)
# #             prev = cur

# #     def detect_and_react(self, img):
# #         img_a, img_b = img[self.slices[0], :], img[self.slices[1], :]
# #         if self.plate_a is None:
# #             plate_a = self.yolo(img_a)
# #             if plate_a is not None:
# #                 self._log.info(f"Plate found in A: {plate_a.shape}")
# #                 self.plate_a = (plate_a, time.time())
# #                 cv2.imwrite(
# #                     f"plate_{time.strftime('%Y-%m-%d_%H-%M-%S')}_a.jpg",
# #                     self.plate_a[0],
# #                 )
# #         if self.plate_b is None:
# #             plate_b = self.yolo(img_b)
# #             if plate_b is not None:
# #                 self._log.info(f"Plate found in B: {plate_b.shape}")
# #                 self.plate_b = (plate_b, time.time())
# #                 cv2.imwrite(
# #                     f"plate_{time.strftime('%Y-%m-%d_%H-%M-%S')}_b.jpg",
# #                     self.plate_b[0],
# #                 )

# #         if self.plate_a is not None and self.plate_b is not None:
# #             # Wait and check again...
# #             time.sleep(3)
# #             plate_b2 = self.yolo(self.cam.capture_array()[self.slices[1], :])
# #             if plate_b2 is not None:
# #                 self._log.info("Plate still there")
# #                 number = self.ocr(plate_b2)
# #                 if number:
# #                     self.mqtt.publish("garage/plate", number, qos=1)
# #                     # Reset recognized plate after 1s
# #                     time.sleep(1)
# #                     self.mqtt.publish("garage/plate", None, qos=1)

# #     def yolo(self, img):
# #         results = self.model.predict(source=img)
# #         if results[0].boxes:
# #             x1, y1, x2, y2 = map(int, results[0].boxes[0].xyxy[0].tolist())
# #             return img[y1:y2, x1:x2]
# #         return None

# #     def ocr(self, plate) -> str:
# #         plate = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY)
# #         lowres = 320
# #         plate = cv2.resize(
# #             plate,
# #             (lowres, int(plate.shape[0] / plate.shape[1] * lowres)),
# #         )
# #         number = self.ocr_reader.readtext(
# #             plate,
# #             detail=0,
# #             allowlist="ABCDEFGHIJKLMOPQRSTUVWXYZ0123456789",
# #         )
# #         if len(number) != 2:
# #             return None
# #         self._log.info(f"OCR: {number}")
# #         cv2.imwrite(
# #             f"plate_{time.strftime('%Y-%m-%d_%H-%M-%S')}_{''.join(number[::-1])}.jpg",
# #             plate,
# #         )
# #         return f"{number[1][0]} {number[0]}"

# #     def ocr2(self, plate) -> str:
# #         plate = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY)
# #         result = pytesseract.image_to_string(
# #             plate,
# #             config="--psm 13 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
# #         )
# #         return result


# def main():
#     # r = Recognizer(
#     #     mqtt_broker="192.168.1.4",
#     #     mqtt_user="broker",
#     #     mqtt_password="123",
#     #     detection_threshold=50,
#     # )
#     # img = cv2.imread("out2.jpg")
#     # print(r.ocr2(img))
#     cam = Picamera2()
#     config = cam.create_still_configuration(
#         main={"size": (2592, 1944), "format": "RGB888"},
#         # lores={"size": low_resolution},
#         transform=Transform(hflip=True, vflip=True),
#     )
#     cam.configure(config)
#     cam.start()
#     detector = MotionDetector(
#         cam=cam,
#         detection_area=Rect(0, 0, 2592, 500),
#         diff_detector=DifferenceDetector(threshold=20, blind_time=15),
#     )
#     mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
#     mqtt_client.username_pw_set(username="broker", password="123")
#     mqtt_client.connect("192.168.1.4")
#     PlateRecognizer(cam=cam, detector=detector, mqtt_client=mqtt_client).loop_forever()


# if __name__ == "__main__":
#     logging.basicConfig(
#         level=logging.DEBUG,
#         format="%(asctime)s %(levelname)-8s %(message)s",
#         datefmt="%Y-%m-%d %H:%M:%S",
#     )
#     logging.getLogger("picamera2").setLevel(logging.CRITICAL)
#     main()


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

            plate = self.plate_detector.next_detected(timeout=15.0)
            if not plate:
                continue

            # Detect plates for the next seconds...
            with run_for(15):
                ocr = self.ocr_detector.process(plate)
                if ocr:
                    logging.getLogger(__name__).info(f"OCR: {ocr}")
                    self.mqtt_client.publish(
                        "garage/plate",
                        f"{motion_direction} {ocr}",
                        qos=1,
                    )
                plate = self.plate_detector.detected.get(timeout=1)
