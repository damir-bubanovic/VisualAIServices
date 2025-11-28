from typing import Dict, List

from src.utils.image_utils import compute_blur_score, load_image_pil


DUMMY_LABELS: List[str] = [
    "cat",
    "dog",
    "car",
    "tree",
]


def analyze_image(image_path: str) -> Dict:
    # Make sure the image can be opened
    _ = load_image_pil(image_path)

    blur_score = compute_blur_score(image_path)

    top1_label = DUMMY_LABELS[0]
    topk_labels = DUMMY_LABELS

    return {
        "top1_label": top1_label,
        "topk_labels": topk_labels,
        "blur_score": float(blur_score),
    }
