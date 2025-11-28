from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as nnf
from PIL import Image as PILImage

from src.models.model_loader import (
    get_model_and_classes,
    get_segmentation_model_and_preprocess,
)
from src.utils.image_utils import (
    load_image_pil,
    compute_blur_score,
    save_mask_image,
)

# Load models once (for performance)
_cls_model, _class_names, _cls_preprocess = get_model_and_classes()
_seg_model, _seg_preprocess = get_segmentation_model_and_preprocess()


def _generate_segmentation_mask(image: PILImage.Image, stem: str) -> str:
    """
    Run DeepLabV3 to produce a simple foreground mask and save it as PNG.
    Anything that is not background (class 0) is considered foreground.
    """
    # Preprocess for segmentation model
    input_tensor = _seg_preprocess(image).unsqueeze(0)  # [1, 3, H, W]

    with torch.no_grad():
        output = _seg_model(input_tensor)["out"]  # [1, C, H, W]
        # Per-pixel class prediction
        seg_map = output.argmax(1)[0]  # [H, W]

    # Foreground = not background (class index 0)
    fg_mask_tensor = torch.zeros_like(seg_map, dtype=torch.uint8)
    fg_mask_tensor[seg_map != 0] = 1  # 1 for foreground, 0 for background

    # Convert to 0/255 uint8 numpy array
    fg_mask_np: np.ndarray = fg_mask_tensor.cpu().numpy() * 255

    # Save mask and return path
    mask_path = save_mask_image(fg_mask_np, stem)
    return mask_path


def analyze_image(image_path: str) -> Dict:
    """
    Perform full analysis:
    - Classification using ResNet18
    - Blur score using Laplacian variance
    - Foreground mask using DeepLabV3
    """
    image: PILImage.Image = load_image_pil(image_path)

    # ---- Classification ----
    input_tensor = _cls_preprocess(image).unsqueeze(0)  # [1, 3, H, W]

    with torch.no_grad():
        outputs = _cls_model(input_tensor)
        probabilities = nnf.softmax(outputs[0], dim=0)

    # Top-1
    top1_idx = torch.argmax(probabilities)
    top1_label = _class_names[top1_idx]

    # Top-5
    _top5_prob, top5_idx = torch.topk(probabilities, 5)
    topk_labels: List[str] = [_class_names[idx] for idx in top5_idx]

    # ---- Blur score ----
    blur_score = compute_blur_score(image_path)

    # ---- Segmentation mask ----
    stem = Path(image_path).stem
    mask_path = _generate_segmentation_mask(image, stem)

    return {
        "top1_label": top1_label,
        "topk_labels": topk_labels,
        "blur_score": float(blur_score),
        "mask_path": mask_path,
    }
