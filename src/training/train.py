from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def get_cifar10_dataloaders(
    data_root: Path,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Prepare CIFAR-10 training and validation dataloaders.

    Parameters
    ----------
    data_root:
        Root directory where CIFAR-10 will be downloaded/stored.
    batch_size:
        Number of samples per batch.

    Returns
    -------
    train_dataloader:
        DataLoader for the CIFAR-10 training split.
    val_dataloader:
        DataLoader for the CIFAR-10 test split (used as validation here).
    class_names:
        List of CIFAR-10 class labels.
    """
    # CIFAR-10 normalization statistics (approximate)
    train_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=str(data_root),
        train=True,
        download=True,
        transform=train_transforms,
    )
    val_dataset = datasets.CIFAR10(
        root=str(data_root),
        train=False,
        download=True,
        transform=val_transforms,
    )

    train_dataloader: DataLoader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    val_dataloader: DataLoader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    class_names: List[str] = list(train_dataset.classes)
    return train_dataloader, val_dataloader, class_names


def create_cifar10_model(num_classes: int) -> nn.Module:
    """
    Create a ResNet18 backbone and replace the final fully-connected layer
    for CIFAR-10 classification.

    Parameters
    ----------
    num_classes:
        Number of target classes (CIFAR-10 = 10).

    Returns
    -------
    model:
        ResNet18 model with the final layer adapted for CIFAR-10.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Run a single training epoch.

    Parameters
    ----------
    model:
        Model to train.
    dataloader:
        DataLoader yielding (images, labels) batches for training.
    criterion:
        Loss function.
    optimizer:
        Optimizer instance (e.g., Adam).
    device:
        Torch device (CPU or CUDA).

    Returns
    -------
    float
        Average training loss over the epoch.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        return 0.0

    epoch_loss = running_loss / float(total_samples)
    return epoch_loss


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate the model on a validation or test dataset.

    Parameters
    ----------
    model:
        Model to evaluate.
    dataloader:
        DataLoader yielding (images, labels) batches for evaluation.
    criterion:
        Loss function.
    device:
        Torch device (CPU or CUDA).

    Returns
    -------
    val_loss:
        Average loss over the dataset.
    val_accuracy:
        Classification accuracy in the range [0.0, 1.0].
    """
    model.eval()
    running_loss = 0.0
    total_correct = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * batch_size

            # Predicted class indices
            _, predictions = torch.max(outputs, 1)

            # Count correct predictions
            correct_batch = int(torch.eq(predictions, labels).sum().item())
            total_correct += correct_batch
            total_samples += batch_size

    if total_samples == 0:
        return 0.0, 0.0

    val_loss = running_loss / float(total_samples)
    val_accuracy = total_correct / float(total_samples)
    return val_loss, val_accuracy


def main() -> None:
    """
    Entry point for CIFAR-10 training.

    This script:
    - Downloads CIFAR-10 (if not already present).
    - Prepares train/validation dataloaders.
    - Fine-tunes a ResNet18 model on CIFAR-10.
    - Saves:
        * Best model checkpoint to artifacts/models/resnet18_cifar10.pth
        * Class names to artifacts/models/cifar10_classes.txt
    """
    data_root = Path("data/cifar10")
    artifacts_root = Path("artifacts/models")
    artifacts_root.mkdir(parents=True, exist_ok=True)

    batch_size = 64
    num_epochs = 2  # Increase for better accuracy if desired
    learning_rate = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare data
    train_dataloader, val_dataloader, class_names = get_cifar10_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
    )

    # Save class names for inference
    class_names_file = artifacts_root / "cifar10_classes.txt"
    class_names_file.write_text("\n".join(class_names), encoding="utf-8")
    print(f"Saved class names to: {class_names_file}")

    # Create model and move to device
    model = create_cifar10_model(num_classes=len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_accuracy = 0.0
    best_model_path = artifacts_root / "resnet18_cifar10.pth"

    for epoch_index in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch_index}/{num_epochs}")

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_accuracy = evaluate_model(
            model=model,
            dataloader=val_dataloader,
            criterion=criterion,
            device=device,
        )

        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val loss  : {val_loss:.4f}")
        print(f"  Val acc   : {val_accuracy:.4%}")

        # Save best model so far
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved to: {best_model_path}")

    print("\nTraining finished.")
    print(f"Best validation accuracy: {best_val_accuracy:.4%}")
    print(f"Checkpoint path: {best_model_path}")


if __name__ == "__main__":
    main()
