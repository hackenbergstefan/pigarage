import RPi.GPIO as GPIO


class IRLight:
    def __init__(self, gpio_ir_light: int):
        self.gpio_ir_light = gpio_ir_light

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.gpio_ir_light, GPIO.OUT, initial=GPIO.LOW)

    def turn_on(self):
        GPIO.output(self.gpio_ir_light, GPIO.HIGH)

    def turn_off(self):
        GPIO.output(self.gpio_ir_light, GPIO.LOW)
