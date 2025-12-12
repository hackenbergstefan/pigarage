import logging
from threading import Condition, Event, Thread
from typing import Callable


class PausableNotifingThread(Thread):
    def __init__(
        self,
        on_resume: Callable[[], None] = lambda: None,
        on_notifying: Callable[[], None] = lambda: None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs, daemon=True)
        self._paused_condition = Condition()
        self._paused = False
        self._notification = Event()
        self._on_resume = on_resume
        self._on_notifying = on_notifying

    def start_paused(self):
        self._paused = True
        self.start()
        with self._paused_condition:
            self._paused_condition.notify()

    def pause(self):
        logging.getLogger(__name__).debug(f"{self.__class__.__name__} pause")
        self._paused = True
        with self._paused_condition:
            self._paused_condition.notify()

    def resume(self):
        logging.getLogger(__name__).debug(f"{self.__class__.__name__} resume")
        self._paused = False
        with self._paused_condition:
            self._paused_condition.notify()

    def run(self):
        while True:
            # Wait until running
            with self._paused_condition:
                while self._paused:
                    self._paused_condition.wait_for(lambda: self._paused is False)
                    self._on_resume()

            # process
            self.process()

    def wait(self, timeout=None):
        logging.getLogger(__name__).debug(f"{self.__class__.__name__} wait")
        self._notification.wait(timeout=timeout)

    def _notify_waiters(self):
        logging.getLogger(__name__).debug(f"{self.__class__.__name__} notify_waiters")
        self._notification.set()
        self._on_notifying()
        self._notification.clear()

    def process(self):
        raise NotImplementedError("Subclasses must implement the process method.")
