from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_image_pil(image_path: str) -> Image.Image:
    """
    Load an image from disk as an RGB PIL Image.

    Parameters
    ----------
    image_path:
        Path to the image file on disk.

    Returns
    -------
    PIL.Image.Image
        Loaded image converted to RGB.

    Raises
    ------
    FileNotFoundError
        If the provided path does not exist.
    """
    image_file = Path(image_path)
    if not image_file.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_file).convert("RGB")
    return image


def compute_blur_score(image_path: str) -> float:
    """
    Compute a simple blur / sharpness metric using variance of the Laplacian.

    Higher values indicate sharper images; lower values indicate blurrier images.

    Parameters
    ----------
    image_path:
        Path to the image file on disk.

    Returns
    -------
    float
        Variance of the Laplacian of the grayscale image.

    Raises
    ------
    FileNotFoundError
        If the provided path does not exist.
    ValueError
        If the image cannot be read as grayscale.
    """
    image_file = Path(image_path)
    if not image_file.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    grayscale_image = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
    if grayscale_image is None:
        raise ValueError(f"Could not read image as grayscale: {image_path}")

    laplacian = cv2.Laplacian(grayscale_image, cv2.CV_64F)
    variance = float(laplacian.var())
    return variance


def save_mask_image(mask: np.ndarray, output_stem: str) -> str:
    """
    Save a binary mask (values 0 or 255) as a PNG under outputs/masks
    and return the resulting file path.

    Parameters
    ----------
    mask:
        2D or 3D NumPy array representing the mask (0/255 values).
    output_stem:
        Base filename (without extension) used when saving the mask.

    Returns
    -------
    str
        Path to the saved mask image as a string.
    """
    mask_uint8 = mask.astype(np.uint8)

    # Ensure 2D (drop singleton channel dimension if present)
    if mask_uint8.ndim == 3 and mask_uint8.shape[2] == 1:
        mask_uint8 = mask_uint8[:, :, 0]

    output_dir = Path("outputs/masks")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{output_stem}_mask.png"
    Image.fromarray(mask_uint8).save(output_path)

    return str(output_path)
