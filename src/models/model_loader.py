from typing import Any, List, Tuple

import torch
from torchvision import models, transforms


def get_model_and_classes() -> Tuple[Any, List[str], transforms.Compose]:
    """
    Load a pretrained ResNet18 model and prepare ImageNet class labels.
    Also returns the preprocessing transform.
    """
    # Load pretrained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()

    # ImageNet class names
    class_names = models.ResNet18_Weights.DEFAULT.meta["categories"]

    # Image preprocessing (resize -> center crop -> normalize)
    preprocess = models.ResNet18_Weights.DEFAULT.transforms()

    return model, class_names, preprocess
