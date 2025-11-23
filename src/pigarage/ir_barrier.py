import time

from RPi import GPIO


class IRBarrier:
    def __init__(self, gpio_ir_barrier_power, gpio_ir_barrier_sensor):
        self.gpio_ir_barrier_power = gpio_ir_barrier_power
        self.gpio_ir_barrier_sensor = gpio_ir_barrier_sensor

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.gpio_ir_barrier_power, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.gpio_ir_barrier_sensor, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

    def power_on(self):
        GPIO.output(self.gpio_ir_barrier_power, GPIO.HIGH)

    def power_off(self):
        GPIO.output(self.gpio_ir_barrier_power, GPIO.LOW)

    @property
    def is_blocked(self):
        self.power_on()
        time.sleep(0.3)  # wait for sensor to stabilize
        state = GPIO.input(self.gpio_ir_barrier_sensor) == GPIO.LOW
        self.power_off()
        return state

    def __enter__(self):
        self.power_on()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.power_off()
