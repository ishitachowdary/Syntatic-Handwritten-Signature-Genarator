import cv2
import numpy as np

IMG_SIZE = 128

def preprocess_signature(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Light binarization (CEDAR-friendly, stroke-safe)
    _, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

    img = img.astype(np.float32) / 127.5 - 1.0
    return img
