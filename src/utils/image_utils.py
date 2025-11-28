from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_image_pil(image_path: str) -> Image.Image:
    """
    Load an image from disk as a RGB PIL Image.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(path).convert("RGB")
    return img


def compute_blur_score(image_path: str) -> float:
    """
    Simple blur / sharpness metric using variance of Laplacian.
    Higher value = sharper image.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise ValueError(f"Could not read image as grayscale: {image_path}")

    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    variance = float(laplacian.var())
    return variance
