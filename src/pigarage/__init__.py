import logging
import time
from queue import Empty
from typing import Callable, Literal

import paho.mqtt.client as mqtt
from paho.mqtt.client import MQTTMessage
from paho.mqtt.subscribeoptions import SubscribeOptions
from picamera2 import Picamera2
from RPi import GPIO

from .diff_detector import DifferenceDetector
from .gate import Gate
from .ir_barrier import IRBarrier
from .ir_light import IRLight
from .neopixel import NeopixelSpi
from .ocr_detector import OcrDetector
from .plate_detector import PlateDetector

Picamera2.set_logging(logging.ERROR)


class LicensePlateProcessor:
    def __init__(
        self,
        diff_detector: DifferenceDetector,
        plate_detector: PlateDetector,
        ocr_detector: OcrDetector,
        on_idle: Callable[[], None] = lambda: None,
        on_plate_detected: Callable[[], None] = lambda: None,
        on_ocr_detected: Callable[
            [str, Literal["arriving", "leaving"]], None
        ] = lambda ocr, direction: None,
    ):
        self.diff_detector = diff_detector
        self.plate_detector = plate_detector
        self.ocr_detector = ocr_detector
        self.on_idle = on_idle
        self.on_plate_detected = on_plate_detected
        self.on_ocr_detected = on_ocr_detected

    def run(self):
        self.diff_detector.start()
        if self.plate_detector:
            self.plate_detector.start_paused()

        while True:
            # Wait for next diff ...
            self.on_idle()
            self.diff_detector.next_detected()

            if self.plate_detector is None:
                continue

            # Wait for next plate ...
            result = self.plate_detector.next_detected(timeout=10.0)
            if result is None:
                continue
            plate, motion_direction = result
            self.on_plate_detected()

            if self.ocr_detector is None:
                continue

            # Detect plates and try ocr for the next seconds...
            self.plate_detector.resume()
            end_time = time.time() + 5
            while time.time() < end_time:
                ocr = self.ocr_detector.process(plate)
                if ocr:
                    logging.getLogger(__name__).info(f"OCR: {ocr}")
                    self.on_ocr_detected(ocr, motion_direction)
                try:
                    plate, motion_direction = self.plate_detector.detected.get(
                        timeout=1
                    )
                except Empty:
                    break
            self.plate_detector.pause()


class PiGarage:
    def __init__(
        self,
        gpio_ir_barrier_power: int,
        gpio_ir_barrier_sensor: int,
        gpio_ir_light_power: int,
        gpio_gate_button: int,
        gpio_gate_closed: int,
        gpio_gate_opened: int,
        mqtt_host: str,
        mqtt_username: str,
        mqtt_password: str,
        debug: bool,
        allowed_plates: list[str],
    ):
        self.allowed_plates = allowed_plates

        # Setup MQTT client
        self.mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.mqtt_client.username_pw_set(username=mqtt_username, password=mqtt_password)
        self.mqtt_client.connect(mqtt_host)
        options = SubscribeOptions(qos=2, noLocal=True)
        self.mqtt_client.subscribe("pigarage/+", options=options)
        self.mqtt_client.on_message = self.mqtt_receive
        self.mqtt_client.loop_start()

        # Setup GPIO devices
        self.gate = Gate(
            gpio_gate_button=gpio_gate_button,
            gpio_gate_closed=gpio_gate_closed,
            gpio_gate_opened=gpio_gate_opened,
            on_opened=lambda _: logging.getLogger(__name__).info("on_opened")
            or self.mqtt_client.publish("pigarage/gate", b"opened"),
            on_closed=lambda _: logging.getLogger(__name__).info("on_closed")
            or self.mqtt_client.publish("pigarage/gate", b"closed"),
        )
        self.ir_barrier = IRBarrier(
            gpio_ir_barrier_power=gpio_ir_barrier_power,
            gpio_ir_barrier_sensor=gpio_ir_barrier_sensor,
        )
        self.neopixel = NeopixelSpi(bus=0, device=0, leds=12)
        self.ir_light = IRLight(gpio_ir_light_power)

        # Setup camera
        self.cam = Picamera2()
        self.cam.configure(
            self.cam.create_still_configuration(
                main={"size": (2592, 1944), "format": "RGB888"},
                lores={"size": (480, 360)},
                # transform=Transform(hflip=True, vflip=True),
            )
        )
        self.cam.start()

        # Setup license plate processor
        self.license_plate_processor = LicensePlateProcessor(
            diff_detector=DifferenceDetector(
                self.cam,
                cam_setting="lores",
                threshold=5,
            ),
            plate_detector=PlateDetector(
                self.cam,
                cam_setting="main",
                debug=debug,
            ),
            ocr_detector=OcrDetector(debug=debug),
            on_idle=self.on_idle,
            on_plate_detected=self.plate_detected,
            on_ocr_detected=self.ocr_detected,
        ).run()

    def on_idle(self):
        self.neopixel.clear()

    def ocr_detected(
        self,
        plate: str,
        motion_direction: Literal["arriving", "leaving"],
    ) -> None:
        logging.getLogger(__name__).info(
            f"Detected plate: {plate} ({motion_direction})"
        )
        self.neopixel.roll(color=(0, 255, 0), duration=1.0)
        if plate in self.allowed_plates:
            with self.ir_barrier:
                logging.getLogger(__name__).info(
                    f"Checking gate {motion_direction} "
                    f"closed: {self.gate.is_closed()} "
                    f"opened: {self.gate.is_opened()} "
                    f"ir_barrier: {self.ir_barrier.is_blocked}"
                )
                if (
                    motion_direction == "arriving"
                    and self.gate.is_closed()
                    and not self.ir_barrier.is_blocked
                ):
                    logging.getLogger(__name__).info("Opening gate...")
                    self.neopixel.roll(color=(255, 0, 0))
                    time.sleep(2)
                    self.gate.open()
                    self.neopixel.clear()
                if (
                    motion_direction == "leaving"
                    and self.gate.is_opened()
                    and not self.ir_barrier.is_blocked
                ):
                    self.neopixel.roll(color=(255, 0, 0))
                    time.sleep(2)
                    logging.getLogger(__name__).info("Closing gate...")
                    self.gate.close()
                    self.neopixel.clear()

    def plate_detected(self) -> None:
        self.neopixel.roll(color=(0, 0, 255), duration=1.0)

    def mqtt_receive(self, client, data, message: MQTTMessage):
        logging.getLogger(__name__).debug(message.topic, message.payload)
        match message.topic:
            case "pigarage/gate":
                if message.payload == b"open":
                    self.gate.open()
                elif message.payload == b"close":
                    self.gate.close()
