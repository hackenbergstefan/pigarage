from configparser import ConfigParser
from pathlib import Path


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

    @property
    def logdir(self) -> Path:
        return Path(self.logging.get("logdir", ".")).resolve()


config = Config()
