from urllib.request import urlopen

import cv2
import numpy as np


def download_image(url: str):
    return cv2.imdecode(
        np.asarray(
            bytearray(urlopen(url).read()),
            dtype=np.uint8,
        ),
        -1,
    )


def download_lnpr_image(id: str) -> cv2.typing.MatLike:
    return download_image(
        f"https://source.roboflow.com/zD7y6XOoQnh7WC160Ae7/{id}/original.jpg"
    )


def download_lnpr_plate(id: str) -> cv2.typing.MatLike:
    img = download_lnpr_image(id)
    mask = download_image(
        f"https://source.roboflow.com/zD7y6XOoQnh7WC160Ae7/{id}/annotation-license-plates.png"
    )
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    return img[y : y + h, x : x + w]
