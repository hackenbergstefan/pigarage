import time
from queue import Empty, Queue
from threading import Thread
from typing import Any

import cv2
from picamera2 import Picamera2


class DetectionThread(Thread):
    """
    Base class for detection threads that captures images from a camera and processes them.
    If `process` returns a non-None value, the thread pauses until `resume` is called.
    The actual value returned by `process` is stored in `self.detected`.
    """

    def __init__(self, cam: Picamera2, cam_setting="lowres", *args, **kwargs):
        super().__init__(*args, **kwargs, daemon=True)
        self.cam = cam
        self.cam_setting = cam_setting
        self.paused = False
        self.detected = Queue(maxsize=1)
        self.images = Queue(maxsize=1)

    def start_paused(self):
        self.paused = True
        self.start()

    def pause(self):
        self.paused = True

    def resume(self):
        while not self.detected.empty():
            self.detected.get()
        self.paused = False

    def next_image(self) -> cv2.typing.MatLike | None:
        """
        Capture the next image from the camera.
        If the camera is not set, it will get an image from the images queue.
        """
        if self.cam is not None:
            img = self.cam.capture_array(self.cam_setting)
            # if self.cam_setting == "lores":
            #     img = cv2.cvtColor(img, cv2.COLOR_YUV2RGB)
            return img
        else:
            try:
                return self.images.get(timeout=1)
            except Empty:
                return None

    def process(self, img: cv2.typing.MatLike) -> Any:
        """
        Process the captured image.
        """
        raise NotImplementedError("Subclasses must implement the process method.")

    def run(self):
        while True:
            while self.paused:
                time.sleep(0.5)

            # Capture the next image
            img = self.next_image()
            if img is None:
                continue

            # Process the image and put the result if it is not None
            result = self.process(img)
            if result is not None:
                self.detected.put(result)

            # Pause if queue is full
            if self.detected.full():
                self.pause()

    def next_detected(self, timeout: float = 0.0) -> Any:
        end_time = time.time() + timeout
        # print(self.__class__, "next_detected", time.time(), end_time)
        if self.paused:
            self.resume()
        while self.detected.empty() and (time.time() < end_time or timeout == 0.0):
            # print(self.__class__, "-> next_detected", time.time(), end_time)
            time.sleep(0.3)
        self.pause()
        if not self.detected.empty():
            return self.detected.get()
        return None
