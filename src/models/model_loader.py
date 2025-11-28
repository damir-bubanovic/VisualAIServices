from pathlib import Path
from typing import Any, List, Tuple

import torch
import torch.nn as nn
from torchvision import models, transforms
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

    class_names: List[str] = list(weights.meta["categories"])
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


def get_cifar_model_and_preprocess() -> Tuple[Any, List[str], Any]:
    """
    Load the fine-tuned ResNet18 CIFAR-10 model and its class names,
    plus the preprocessing transform for inference.

    Requires that you have already run: python -m src.training.train
    """
    ckpt_path = Path("artifacts/models/resnet18_cifar10.pth")
    classes_path = Path("artifacts/models/cifar10_classes.txt")

    if not ckpt_path.exists() or not classes_path.exists():
        raise FileNotFoundError(
            "CIFAR-10 model or class file not found. "
            "Run 'python -m src.training.train' first."
        )

    # Read class names
    class_names: List[str] = classes_path.read_text(encoding="utf-8").splitlines()

    # Recreate the model architecture and load weights
    num_classes = len(class_names)
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # Preprocessing similar to CIFAR-10 validation
    preprocess = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    return model, class_names, preprocess
