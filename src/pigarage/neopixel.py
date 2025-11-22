import threading
import time

try:
    import spidev
except ImportError:
    pass


def flatten(xss):
    return [x for xs in xss for x in xs]


def rgb_to_bits(rgb: tuple[int, int, int]):
    r, g, b = rgb
    return flatten(
        (0xF8 if bit == "1" else 0xC0 for bit in f"{color:08b}") for color in (g, r, b)
    )


class StoppableTask(threading.Thread):
    def __init__(self, func, *args, **kwargs):
        self._running = False
        self._func = func
        super().__init__(*args, **kwargs)

    def start(self):
        self._running = True
        return super().start()

    def stop(self):
        self._running = False

    def run(self):
        while True:
            if self._running is False:
                return
            self._func()


class NeopixelSpi:
    instance = None

    def __init__(self, bus: int, device: int, leds: int, spi_freq=800):
        NeopixelSpi.instance = self
        self.spi = spidev.SpiDev()
        self.spi.open(bus, device)
        self.spi.max_speed_hz = spi_freq * 1024 * 8

        self.state = leds * [(0, 0, 0)]
        self._task = None
        self.clear()

    def update(self, newstate: list[tuple[int, int, int]] = None):
        if newstate:
            self.state = newstate
        raw_data = sum((rgb_to_bits(led) for led in self.state), [])
        self.spi.xfer3(raw_data)

    def clear(self):
        self.stop()
        self.fill(0, 0, 0)

    def fill(self, red: int, green: int, blue: int):
        self.stop()
        self.state = len(self.state) * [(red % 256, green % 256, blue % 256)]
        self.update()

    def fade(self, color_from, color_to, duration=0.1, steps=20):
        for i in range(steps + 1):
            self.fill(
                *tuple(
                    round(c1 + (c2 - c1) / steps * i)
                    for c1, c2 in zip(color_from, color_to)
                )
            )
            time.sleep(duration / steps)

    def pulse_once(self, color, amplitude=1.0, duration=0.5, steps="auto"):
        if steps == "auto":
            steps = 20 * duration
        color_to = [(1 - amplitude) * c for c in color]
        self.fade(color, color_to, duration=duration / 2, steps=steps)
        self.fade(color_to, color, duration=duration / 2, steps=steps)

    def pulse(self, color, amplitude=1.0, duration=0.5, steps="auto"):
        self.stop()
        self._task = StoppableTask(
            func=lambda: self.pulse_once(
                color,
                amplitude=amplitude,
                duration=duration,
                steps=steps,
            )
        )
        self._task.start()

    def stop(self):
        if self._task is not None and threading.current_thread() is not self._task:
            self._task.stop()
            self._task.join()
            self._task = None

    def roll_once(self, color: tuple[int, int, int], duration: int = 2.0):
        self.stop()
        for i in range(len(self.state)):
            state = len(self.state) * [(0, 0, 0)]
            state[i] = color
            self.update(state)
            time.sleep(duration / len(self.state))

    def roll(self, color: tuple[int, int, int], duration: int = 2.0):
        self.stop()
        self._task = StoppableTask(
            func=lambda: self.roll_once(
                color,
                duration=duration,
            )
        )
        self._task.start()


neo = NeopixelSpi(bus=0, device=0, leds=12)
neo.clear()
# neo.fill(235, 90, 7)
