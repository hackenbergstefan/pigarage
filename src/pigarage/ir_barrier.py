import time
import types
from typing import Self

try:
    from RPi import GPIO
except ImportError:
    from unittest.mock import MagicMock

    GPIO = MagicMock()


class IRBarrier:
    def __init__(self, gpio_ir_barrier_power: int, gpio_ir_barrier_sensor: int) -> None:
        self.gpio_ir_barrier_power = gpio_ir_barrier_power
        self.gpio_ir_barrier_sensor = gpio_ir_barrier_sensor

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.gpio_ir_barrier_power, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.gpio_ir_barrier_sensor, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

    def power_on(self) -> None:
        GPIO.output(self.gpio_ir_barrier_power, GPIO.HIGH)

    def power_off(self) -> None:
        GPIO.output(self.gpio_ir_barrier_power, GPIO.LOW)

    @property
    def is_blocked(self) -> bool:
        self.power_on()
        time.sleep(0.3)  # wait for sensor to stabilize
        state = GPIO.input(self.gpio_ir_barrier_sensor) == GPIO.LOW
        self.power_off()
        return state

    def __enter__(self) -> Self:
        self.power_on()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        self.power_off()
