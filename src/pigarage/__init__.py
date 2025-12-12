import contextlib
import logging
import time
from collections.abc import Callable
from queue import Empty
from typing import Literal

import paho.mqtt.client as mqtt
from paho.mqtt.client import MQTTMessage
from paho.mqtt.subscribeoptions import SubscribeOptions

try:
    from picamera2 import Picamera2
except ImportError:
    from unittest.mock import MagicMock

    Picamera2 = MagicMock()


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
        on_allowed: Callable[
            [str, Literal["arriving", "leaving"]], None
        ] = lambda _: None,
    ) -> None:
        self._diff_detector = diff_detector
        self._plate_detector = plate_detector
        self._ocr_detector = ocr_detector
        self._on_allowed = on_allowed

    def run(self) -> None:
        self._plate_detector.start_paused()
        self._ocr_detector.start_paused()
        self._diff_detector.start_paused()

        while True:
            # Wait for next diff
            self._diff_detector.resume()
            self._diff_detector.wait()

            # Start detecting plate and text
            self._plate_detector.resume()
            self._ocr_detector.resume()

            # Wait for plate text and direction
            with contextlib.suppress(Empty):
                allowed_plate = self._ocr_detector.detected_ocrs.get(timeout=10.0)
                self._ocr_detector.pause()
                direction = self._plate_detector.detected_directions.get(timeout=10.0)
                self._plate_detector.pause()
                logging.getLogger(__name__).info(
                    f"Allowed Plate '{allowed_plate}' in direction '{direction}'"
                )
                self._on_allowed(allowed_plate, direction)
            self._plate_detector.pause()
            self._ocr_detector.pause()


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
    ) -> None:
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
        plate_detector = PlateDetector(
            self.cam,
            cam_setting="main",
            debug=debug,
            on_notifying=self.on_plate_detected,
        )
        self.license_plate_processor = LicensePlateProcessor(
            diff_detector=DifferenceDetector(
                self.cam,
                cam_setting="lores",
                threshold=50,
                on_resume=self.on_idle,
                on_notifying=self.on_diff_detected,
            ),
            plate_detector=plate_detector,
            ocr_detector=OcrDetector(
                debug=debug,
                detected_plates=plate_detector.detected_plates,
                allowed_plates=allowed_plates,
            ),
            on_allowed=self.on_allowed,
        ).run()

    def on_idle(self) -> None:
        self.neopixel.clear()
        self.ir_light.turn_off()

    def on_diff_detected(self) -> None:
        self.ir_light.turn_on()

    def on_allowed(
        self,
        _plate: str,
        motion_direction: Literal["arriving", "leaving"],
    ) -> None:
        with self.ir_barrier:
            logging.getLogger(__name__).info(
                f"Checking gate {motion_direction} "
                f"closed: {self.gate.is_closed()} "
                f"opened: {self.gate.is_opened()} "
                f"ir_barrier: {self.ir_barrier.is_blocked}"
            )
            self.neopixel.roll(color=(255, 0, 0))
            time.sleep(2)
            if (
                motion_direction == "arriving"
                and self.gate.is_closed()
                and not self.ir_barrier.is_blocked
            ):
                logging.getLogger(__name__).info("Opening gate...")
                self.gate.open()
            if (
                motion_direction == "leaving"
                and self.gate.is_opened()
                and not self.ir_barrier.is_blocked
            ):
                logging.getLogger(__name__).info("Closing gate...")
                self.gate.close()

            self.neopixel.clear()
            # Pause entire processing for a while
            time.sleep(30)

    def on_plate_detected(self) -> None:
        self.neopixel.roll(color=(0, 0, 255), duration=1.0)

    def mqtt_receive(self, client, data, message: MQTTMessage) -> None:
        logging.getLogger(__name__).debug(
            f"mqtt_receive: {message.topic}, {message.payload}"
        )
        match message.topic:
            case "pigarage/gate":
                if message.payload == b"open":
                    self.gate.open()
                elif message.payload == b"close":
                    self.gate.close()
