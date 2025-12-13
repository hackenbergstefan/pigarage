import logging
from collections.abc import Callable
from threading import Condition, Event, Thread
from typing import Never


class PausableNotifingThread(Thread):
    def __init__(
        self,
        on_resume: Callable[[], None] = lambda: None,
        on_notifying: Callable[[], None] = lambda: None,
        *args: object,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs, daemon=True)
        self._paused_condition = Condition()
        self._paused = False
        self._notification = Event()
        self._on_resume = on_resume
        self._on_notifying = on_notifying
        self._log = logging.getLogger(self.__class__.__name__)

    def start_paused(self) -> None:
        self._paused = True
        self.start()
        with self._paused_condition:
            self._paused_condition.notify()

    def pause(self) -> None:
        if self._paused:
            return
        self._log.debug("")
        self._paused = True
        with self._paused_condition:
            self._paused_condition.notify()

    def resume(self) -> None:
        if not self._paused:
            return
        self._log.debug("")
        self._paused = False
        with self._paused_condition:
            self._paused_condition.notify()

    def run(self) -> None:
        while True:
            # Wait until running
            with self._paused_condition:
                while self._paused:
                    self._paused_condition.wait_for(lambda: self._paused is False)
                    self._on_resume()

            # process
            self.process()

    def wait(self, timeout: float | None = None) -> None:
        self._log.debug("")
        self._notification.wait(timeout=timeout)

    def _notify_waiters(self) -> None:
        if self._paused:
            return
        self._log.debug("")
        self._notification.set()
        self._on_notifying()
        self._notification.clear()

    def process(self) -> Never:
        raise NotImplementedError("Subclasses must implement the process method.")
