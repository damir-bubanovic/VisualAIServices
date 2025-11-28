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
    Load a pretrained ResNet18 classifier and its ImageNet class labels.

    Returns
    -------
    model:
        ResNet18 model in evaluation mode with ImageNet weights.
    class_names:
        List of ImageNet class names corresponding to the model outputs.
    preprocess:
        Torchvision transform used to preprocess input images for this model.
    """
    imagenet_weights = ResNet18_Weights.DEFAULT
    imagenet_model = models.resnet18(weights=imagenet_weights)
    imagenet_model.eval()

    class_names: List[str] = list(imagenet_weights.meta["categories"])
    preprocess = imagenet_weights.transforms()

    return imagenet_model, class_names, preprocess


def get_segmentation_model_and_preprocess() -> Tuple[Any, Any]:
    """
    Load a pretrained DeepLabV3-ResNet50 segmentation model and its preprocess transform.

    Returns
    -------
    model:
        DeepLabV3-ResNet50 model in evaluation mode with COCO-style weights.
    preprocess:
        Torchvision transform used to preprocess input images for this model.
    """
    segmentation_weights = DeepLabV3_ResNet50_Weights.DEFAULT
    segmentation_model = deeplabv3_resnet50(weights=segmentation_weights)
    segmentation_model.eval()

    preprocess = segmentation_weights.transforms()

    return segmentation_model, preprocess


def get_cifar_model_and_preprocess() -> Tuple[Any, List[str], Any]:
    """
    Load the fine-tuned ResNet18 CIFAR-10 classifier and its class names,
    plus the preprocessing transform used at inference time.

    This function assumes that the training script has been run:

        python -m src.training.train

    which produces:
        - artifacts/models/resnet18_cifar10.pth
        - artifacts/models/cifar10_classes.txt

    Returns
    -------
    model:
        ResNet18 model in evaluation mode, with the final layer adapted to CIFAR-10
        and weights loaded from the checkpoint.
    class_names:
        List of CIFAR-10 class names, one per output neuron.
    preprocess:
        Torchvision transform used to preprocess input images for this model.
    """
    checkpoint_path = Path("artifacts/models/resnet18_cifar10.pth")
    class_names_path = Path("artifacts/models/cifar10_classes.txt")

    if not checkpoint_path.exists() or not class_names_path.exists():
        raise FileNotFoundError(
            "CIFAR-10 model or class file not found. "
            "Run 'python -m src.training.train' first to create them."
        )

    # Load class names
    class_names: List[str] = class_names_path.read_text(encoding="utf-8").splitlines()

    # Recreate the model architecture and load fine-tuned weights
    num_classes = len(class_names)
    cifar_model = models.resnet18(weights=None)
    in_features = cifar_model.fc.in_features
    cifar_model.fc = nn.Linear(in_features, num_classes)

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    cifar_model.load_state_dict(state_dict)
    cifar_model.eval()

    # Preprocessing similar to CIFAR-10 validation pipeline
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

    return cifar_model, class_names, preprocess
