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


def save_mask_image(mask: np.ndarray, stem: str) -> str:
    """
    Save a binary mask (0 / 255) as a PNG in outputs/masks and
    return the path as a string.
    """
    mask = mask.astype(np.uint8)

    # Ensure 2D
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]

    output_dir = Path("outputs/masks")
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"{stem}_mask.png"
    Image.fromarray(mask).save(out_path)

    return str(out_path)
