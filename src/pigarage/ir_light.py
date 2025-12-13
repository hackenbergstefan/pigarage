import logging

try:
    from RPi import GPIO
except ImportError:
    from unittest.mock import MagicMock

    GPIO = MagicMock()


class IRLight:
    def __init__(self, gpio_ir_light: int) -> None:
        self.gpio_ir_light = gpio_ir_light

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.gpio_ir_light, GPIO.OUT, initial=GPIO.LOW)
        self._log = logging.getLogger(self.__class__.__name__)

    def turn_on(self) -> None:
        self._log.debug("")
        GPIO.output(self.gpio_ir_light, GPIO.HIGH)

    def turn_off(self) -> None:
        self._log.debug("")
        GPIO.output(self.gpio_ir_light, GPIO.LOW)
