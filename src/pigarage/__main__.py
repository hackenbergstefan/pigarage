import logging
from configparser import ConfigParser

from picamera2 import Picamera2

from . import PiGarage

Picamera2.set_logging(logging.ERROR)


class Config:
    def __init__(self):
        self.config = ConfigParser()
        self.config.read("pigarage.ini")

    @property
    def mqtt(self):
        return self.config["mqtt"]

    @property
    def gpio(self):
        return self.config["gpio"]

    @property
    def logging(self):
        return self.config["logging"]

    @property
    def pigarage(self):
        return self.config["pigarage"]


def allowed_detected(plate: str):
    pass


def main():
    config = Config()

    logging.basicConfig(
        level=getattr(logging, config.logging.get("level", "INFO")),
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    PiGarage(
        gpio_ir_barrier_power=int(config.gpio["ir_barrier_power"]),
        gpio_ir_barrier_sensor=int(config.gpio["ir_barrier_sensor"]),
        gpio_ir_light_power=int(config.gpio["ir_light_power"]),
        gpio_gate_button=int(config.gpio["gate_button"]),
        gpio_gate_closed=int(config.gpio["gate_closed"]),
        gpio_gate_opened=int(config.gpio["gate_opened"]),
        mqtt_host=config.mqtt["host"],
        mqtt_username=config.mqtt["username"],
        mqtt_password=config.mqtt["password"],
        debug=config.logging.getboolean("debug", False),
        allowed_plates=[
            p
            for plate in str(config.pigarage["allowed_plates"]).split(",")
            if (p := plate.strip())
        ],
    )


if __name__ == "__main__":
    main()
