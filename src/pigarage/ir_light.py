import logging

import RPi.GPIO as GPIO


class IRLight:
    def __init__(self, gpio_ir_light: int):
        self.gpio_ir_light = gpio_ir_light

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.gpio_ir_light, GPIO.OUT, initial=GPIO.LOW)

    def turn_on(self):
        logging.getLogger(__name__).debug("IRLight turn_on")
        GPIO.output(self.gpio_ir_light, GPIO.HIGH)

    def turn_off(self):
        logging.getLogger(__name__).debug("IRLight turn_off")
        GPIO.output(self.gpio_ir_light, GPIO.LOW)
