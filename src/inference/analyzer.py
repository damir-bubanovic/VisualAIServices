from typing import Dict, List

import torch
from PIL import Image

from src.models.model_loader import get_model_and_classes
from src.utils.image_utils import load_image_pil, compute_blur_score


# Load model once (good for performance)
_model, _class_names, _preprocess = get_model_and_classes()


def analyze_image(image_path: str) -> Dict:
    """
    Perform real inference using pretrained ResNet18.
    """
    image: Image.Image = load_image_pil(image_path)

    # Preprocess
    input_tensor = _preprocess(image).unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        outputs = _model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Top-1
    top1_prob, top1_idx = torch.max(probabilities, dim=0)
    top1_label = _class_names[top1_idx]

    # Top-5
    top5_prob, top5_idx = torch.topk(probabilities, 5)
    topk_labels = [_class_names[idx] for idx in top5_idx]

    # Blur score
    blur_score = compute_blur_score(image_path)

    return {
        "top1_label": top1_label,
        "topk_labels": topk_labels,
        "blur_score": float(blur_score),
    }
