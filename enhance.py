import base64

import cv2
import numpy as np


def gray_world(img):
    img = img.astype(np.float32)
    avg = img.reshape(-1, 3).mean(axis=0)
    avg = np.maximum(avg, 1e-6)
    gain = np.clip(avg.mean() / avg, 0.5, 2.0)
    return np.clip(img * gain, 0, 255).astype(np.uint8)


def enhance_underwater_image(image_bytes):
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")

    img = cv2.addWeighted(img, 0.4, gray_world(img), 0.6, 0)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.bilateralFilter(l, 7, 25, 7)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def to_base64(img):
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise ValueError("Could not encode image")
    return base64.b64encode(buf).decode("utf-8")