import time

from pigarage.util import PausableNotifingThread


def test_start_paused():
    class T(PausableNotifingThread):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.has_run = False

        def process(self):
            self.has_run = True

    t = T()
    t.start_paused()
    assert t._paused is True
    assert t.has_run is False


def test_pause():
    class T(PausableNotifingThread):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.has_run = 0

        def process(self):
            self.has_run += 1

    t = T()
    t.start()
    t.pause()
    assert t._paused is True
    time.sleep(0.3)
    has_run = t.has_run
    assert has_run > 0
    time.sleep(0.3)
    assert t.has_run == has_run


def test_resume():
    class T(PausableNotifingThread):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.has_run = 0

        def process(self):
            self.has_run += 1
            self.pause()

    t = T()
    t.start()
    time.sleep(0.3)
    assert t._paused is True
    assert t.has_run == 1
    t.resume()
    time.sleep(0.3)
    assert t._paused is True
    assert t.has_run == 2


def test_notification():
    class T(PausableNotifingThread):
        def process(self):
            time.sleep(0.5)
            self._notify_waiters()

    t = T()
    start = time.time()
    t.start()
    t.wait()
    assert time.time() - start > 0.5
