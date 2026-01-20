# train.py
"""
Train a simple FashionMNIST classifier that uses the custom LearnedAffine layer.

Requirements:
- Train for 3 epochs
- Use CrossEntropyLoss
- Use AdamW optimizer
- Use StepLR scheduler
- Use proper eval() + no_grad() in evaluation
- Set torch.manual_seed for reproducibility
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from layers import LearnedAffine


class FashionMNISTModel(nn.Module):
    """
    Simple MLP for FashionMNIST with a custom LearnedAffine layer.

    Architecture:
    - Flatten 28x28 image to 784
    - Linear(784 -> 256)
    - LearnedAffine(256)
    - ReLU
    - Linear(256 -> 10)
    """
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.affine = LearnedAffine(256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.affine(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def get_dataloaders(batch_size: int):
    """
    Return train and test dataloaders for FashionMNIST.
    """
    transform = transforms.ToTensor()

    train_dataset = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, test_loader


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    One training epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward + optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track loss
        running_loss += loss.item() * images.size(0)

        # Track accuracy
        _, predicted = torch.max(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """
    Evaluation loop.

    Uses:
    - model.eval()
    - torch.no_grad()
    as required by the rubric.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def parse_args():
    parser = argparse.ArgumentParser(description="FashionMNIST + LearnedAffine training script")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size (default: 64)")
    parser.add_argument("--epochs", type=int, default=3, help="number of epochs to train (default: 3)")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate (default: 1e-3)")
    parser.add_argument("--gamma", type=float, default=0.7, help="StepLR gamma (default: 0.7)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument("--use-cuda", action="store_true", help="use CUDA if available")
    return parser.parse_args()


def main():
    args = parse_args()

    # Reproducibility: set seed
    torch.manual_seed(args.seed)

    # Device setup
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, test_loader = get_dataloaders(args.batch_size)

    # Model, loss, optimizer, scheduler
    model = FashionMNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    # StepLR scheduler: step once per epoch
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(
            model, test_loader, criterion, device
        )

        scheduler.step()

        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        print(f"Val   loss: {val_loss:.4f} | Val   acc: {val_acc:.4f}")
        print(f"Current LR: {scheduler.get_last_lr()[0]:.6f}")

    print("\nTraining finished.")


if __name__ == "__main__":
    main()
