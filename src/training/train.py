from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def get_dataloaders(
    data_dir: Path,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Prepare CIFAR-10 train/val dataloaders.
    """
    # CIFAR-10 normalization (approximate)
    transform_train = transforms.Compose(
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

    transform_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    train_ds = datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform_train,
    )
    val_ds = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform_val,
    )

    train_loader: DataLoader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    val_loader: DataLoader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    class_names: List[str] = list(train_ds.classes)
    return train_loader, val_loader, class_names


def create_model(num_classes: int) -> nn.Module:
    """
    Create a ResNet18 and replace final layer for CIFAR-10.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    total_samples = 0

    for images, labels in loader:
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




def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    total_correct = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * batch_size

            _, preds = torch.max(outputs, 1)

            # Count correct predictions directly on the tensor comparison
            correct_batch = int(torch.eq(preds, labels).sum().item())

            total_correct += correct_batch
            total_samples += batch_size

    if total_samples == 0:
        return 0.0, 0.0

    epoch_loss = running_loss / float(total_samples)
    accuracy = total_correct / float(total_samples)
    return epoch_loss, accuracy




def main() -> None:
    """
    Simple training script:
    - Downloads CIFAR-10
    - Fine-tunes ResNet18
    - Saves best model checkpoint + class names

    This is for demonstrating training workflow; you can keep epochs small.
    """
    data_dir = Path("data/cifar10")
    output_dir = Path("artifacts/models")
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 64
    num_epochs = 2  # increase if you want better accuracy
    learning_rate = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, class_names = get_dataloaders(data_dir, batch_size)

    # Save class names
    class_file = output_dir / "cifar10_classes.txt"
    class_file.write_text("\n".join(class_names), encoding="utf-8")
    print(f"Saved class names to: {class_file}")

    model = create_model(num_classes=len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = 0.0
    best_model_path = output_dir / "resnet18_cifar10.pth"

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val loss  : {val_loss:.4f}")
        print(f"  Val acc   : {val_acc:.4%}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved to: {best_model_path}")

    print("\nTraining finished.")
    print(f"Best validation accuracy: {best_val_acc:.4%}")
    print(f"Checkpoint path: {best_model_path}")


if __name__ == "__main__":
    main()
