from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as nnf
from PIL import Image as PILImage

from src.models.model_loader import (
    get_model_and_classes,
    get_segmentation_model_and_preprocess,
    get_cifar_model_and_preprocess,
)
from src.utils.image_utils import (
    load_image_pil,
    compute_blur_score,
    save_mask_image,
)

# Type alias for what each analysis function returns
AnalysisResult = Dict[str, Any]

# ---------------------------------------------------------------------------
# Model initialization (loaded once at import time for performance)
# ---------------------------------------------------------------------------

# ImageNet classifier
_imagenet_model, _imagenet_class_names, _imagenet_preprocess = get_model_and_classes()

# Segmentation model (DeepLabV3)
_segmentation_model, _segmentation_preprocess = get_segmentation_model_and_preprocess()

# CIFAR-10 classifier (might not exist if training was not run)
try:
    _cifar10_model, _cifar10_class_names, _cifar10_preprocess = get_cifar_model_and_preprocess()
except FileNotFoundError:
    _cifar10_model = None
    _cifar10_class_names: List[str] = []
    _cifar10_preprocess = None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _generate_segmentation_mask(image: PILImage.Image, output_stem: str) -> str:
    """
    Run DeepLabV3 to produce a simple foreground mask and save it as a PNG file.

    All pixels that are not background (class index 0) are considered foreground.

    Parameters
    ----------
    image:
        Input RGB image as a PIL Image.
    output_stem:
        Base filename (without extension) used when saving the mask.

    Returns
    -------
    str
        Path to the saved binary mask PNG on disk.
    """
    # Preprocess for the segmentation model
    segmentation_input = _segmentation_preprocess(image).unsqueeze(0)  # [1, 3, H, W]

    with torch.no_grad():
        segmentation_output = _segmentation_model(segmentation_input)["out"]  # [1, C, H, W]
        # Per-pixel predicted class indices
        predicted_classes = segmentation_output.argmax(1)[0]  # [H, W]

    # Foreground = not background (class index 0)
    foreground_mask_tensor = torch.zeros_like(predicted_classes, dtype=torch.uint8)
    foreground_mask_tensor[predicted_classes != 0] = 1  # 1 for foreground, 0 for background

    # Convert to 0/255 uint8 numpy array
    foreground_mask_np: np.ndarray = foreground_mask_tensor.cpu().numpy() * 255

    # Save mask and return its path
    mask_path = save_mask_image(foreground_mask_np, output_stem)
    return mask_path


# ---------------------------------------------------------------------------
# Public analysis functions
# ---------------------------------------------------------------------------

def analyze_image(image_path: str) -> AnalysisResult:
    """
    Perform full analysis using the ImageNet-based pipeline.

    The pipeline performs:
    - ImageNet classification (pretrained ResNet18)
    - Blur score computation (Laplacian variance)
    - Foreground segmentation mask generation (DeepLabV3)

    Parameters
    ----------
    image_path:
        Path to the input image on disk.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'top1_label': str
        - 'topk_labels': List[str]
        - 'blur_score': float
        - 'mask_path': str
    """
    image: PILImage.Image = load_image_pil(image_path)

    # ---- Classification (ImageNet) ----
    classifier_input = _imagenet_preprocess(image).unsqueeze(0)  # [1, 3, H, W]

    with torch.no_grad():
        classifier_output = _imagenet_model(classifier_input)
        probabilities = nnf.softmax(classifier_output[0], dim=0)

    # Top-1 prediction
    top1_index = torch.argmax(probabilities)
    top1_label = _imagenet_class_names[int(top1_index)]

    # Top-5 predictions
    _top5_probabilities, top5_indices = torch.topk(probabilities, 5)
    topk_labels: List[str] = [_imagenet_class_names[int(index)] for index in top5_indices]

    # ---- Blur score ----
    blur_score = compute_blur_score(image_path)

    # ---- Segmentation mask ----
    image_stem = Path(image_path).stem
    mask_path = _generate_segmentation_mask(image, image_stem)

    return {
        "top1_label": top1_label,
        "topk_labels": topk_labels,
        "blur_score": float(blur_score),
        "mask_path": mask_path,
    }


def analyze_image_cifar(image_path: str) -> AnalysisResult:
    """
    Analyze an image using the fine-tuned CIFAR-10 classifier.

    The pipeline performs:
    - CIFAR-10 classification (fine-tuned ResNet18)
    - Blur score computation
    - No segmentation mask (mask_path is an empty string)

    Parameters
    ----------
    image_path:
        Path to the input image on disk.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'top1_label': str
        - 'topk_labels': List[str]
        - 'blur_score': float
        - 'mask_path': str (always "")
    """
    if _cifar10_model is None or _cifar10_preprocess is None or not _cifar10_class_names:
        raise RuntimeError(
            "CIFAR-10 model not available. "
            "Run 'python -m src.training.train' first to create the checkpoint."
        )

    image: PILImage.Image = load_image_pil(image_path)

    # ---- Classification (CIFAR-10) ----
    classifier_input = _cifar10_preprocess(image).unsqueeze(0)  # [1, 3, 32, 32]

    with torch.no_grad():
        classifier_output = _cifar10_model(classifier_input)
        probabilities = nnf.softmax(classifier_output[0], dim=0)

    # Top-1 prediction
    top1_index = torch.argmax(probabilities)
    top1_label = _cifar10_class_names[int(top1_index)]

    # Top-5 predictions (or fewer if num_classes < 5)
    top_k = min(5, len(_cifar10_class_names))
    _topk_probabilities, topk_indices = torch.topk(probabilities, top_k)
    topk_labels: List[str] = [_cifar10_class_names[int(index)] for index in topk_indices]

    # ---- Blur score ----
    blur_score = compute_blur_score(image_path)

    return {
        "top1_label": top1_label,
        "topk_labels": topk_labels,
        "blur_score": float(blur_score),
        "mask_path": "",
    }
