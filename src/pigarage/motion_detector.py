import logging
from typing import Literal

import cv2
import numpy as np
from picamera2 import Picamera2

from .util import DetectionThread


class MotionDetector(DetectionThread):
    def __init__(
        self,
        cam: Picamera2,
        cam_setting="lores",
        history: int = 200,
        variance_threshold: int = 150,
        resize: int = 480,
        *,
        debug: bool = False,
    ):
        super().__init__(cam=cam, cam_setting=cam_setting)

        self.tracker = cv2.createBackgroundSubtractorMOG2(
            history=history,
            detectShadows=False,
            varThreshold=variance_threshold,
        )
        self.history = []
        self.history_length = 6
        self.resize = resize
        self.debug = debug

    def _center_of_mass(self, img: cv2.typing.MatLike) -> float | None:
        """
        Calculate y-coordinate of center of mass of the convex hull of all contours in the image.
        """
        # Find contours in the image
        img = cv2.medianBlur(img, 5)

        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Calculate the convex hull of the contours
        hull = cv2.convexHull(np.vstack(contours))

        # Calculate the center of mass of the convex hull
        moments = cv2.moments(hull)
        # Skip if area is smaller than 10% of the image
        if moments["m00"] < 0.1 * img.shape[0] * img.shape[1]:
            return None
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])

        if self.debug:
            # Draw the convex hull and center of mass on the image for debugging
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = cv2.drawContours(img, [hull], -1, color=(255, 0, 0), thickness=3)
            img = cv2.circle(
                img,
                (center_x, center_y),
                radius=7,
                color=(0, 0, 255),
                thickness=-1,
            )
            cv2.imwrite("/tmp/motion_detector.jpg", img)
        return center_y

    def process(self, img: cv2.typing.MatLike) -> None | Literal["arriving", "leaving"]:
        """
        Process the image to detect motion direction based on the center of mass of the convex hull
        of the contours in the foreground mask.
        """
        # img = cv2.resize(
        #     img,
        #     dsize=(self.resize, int(img.shape[0] / img.shape[1] * self.resize)),
        # )
        fg_mask = self.tracker.apply(img)
        center_y = self._center_of_mass(fg_mask)
        if center_y is not None:
            self.history.append(center_y)

        # Detect motion direction based on the history of center of mass positions
        if len(self.history) == self.history_length:
            ys = sorted(range(self.history_length), key=lambda i: self.history[i])
            # logging.getLogger(__name__).debug(f"ys: {ys}")
            self.history.clear()
            if ys == list(range(self.history_length)) and ys[0] != ys[-1]:
                direction = "arriving"
                logging.getLogger(__name__).debug(f"direction: {direction}")
                return direction
            if ys == list(reversed(range(self.history_length))) and ys[0] != ys[-1]:
                direction = "leaving"
                logging.getLogger(__name__).debug(f"direction: {direction}")
                return direction
        return None
