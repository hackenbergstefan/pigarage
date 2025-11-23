import time

from RPi import GPIO


class Gate:
    def __init__(self, gpio_gate_closed, gpio_gate_opened, gpio_gate_button):
        self.gpio_gate_closed = gpio_gate_closed
        self.gpio_gate_opened = gpio_gate_opened
        self.gpio_gate_button = gpio_gate_button

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.gpio_gate_button, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.gpio_gate_closed, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        GPIO.setup(self.gpio_gate_opened, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

    def is_closed(self):
        return GPIO.input(self.gpio_gate_closed) == GPIO.HIGH

    def is_opened(self):
        return GPIO.input(self.gpio_gate_opened) == GPIO.HIGH

    def open(self):
        if not self.is_opened():
            GPIO.output(self.gpio_gate_button, GPIO.HIGH)
            time.sleep(0.3)
            GPIO.output(self.gpio_gate_button, GPIO.LOW)

    def close(self):
        if not self.is_closed():
            GPIO.output(self.gpio_gate_button, GPIO.HIGH)
            time.sleep(0.3)
            GPIO.output(self.gpio_gate_button, GPIO.LOW)
