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

    def turn_on(self) -> None:
        logging.getLogger(__name__).debug("IRLight turn_on")
        GPIO.output(self.gpio_ir_light, GPIO.HIGH)

    def turn_off(self) -> None:
        logging.getLogger(__name__).debug("IRLight turn_off")
        GPIO.output(self.gpio_ir_light, GPIO.LOW)
