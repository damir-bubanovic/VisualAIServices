from typing import Any, List, Tuple

from torchvision import models
from torchvision.models import ResNet18_Weights
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
)


def get_model_and_classes() -> Tuple[Any, List[str], Any]:
    """
    Load a pretrained ResNet18 model and prepare ImageNet class labels.
    Also returns the preprocessing transform.
    """
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.eval()

    class_names = weights.meta["categories"]
    preprocess = weights.transforms()

    return model, class_names, preprocess


def get_segmentation_model_and_preprocess() -> Tuple[Any, Any]:
    """
    Load a pretrained DeepLabV3-ResNet50 model for semantic segmentation.
    Returns the model and the preprocessing transform.
    """
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = deeplabv3_resnet50(weights=weights)
    model.eval()

    preprocess = weights.transforms()

    return model, preprocess
