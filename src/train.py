"""Training and evaluation utilities."""

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass(frozen=True)
class TrainConfig:
    """Configurable training hyperparameters."""

    learning_rate: float = 0.001
    num_epochs: int = 10
    device: str = "cpu"


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Callable,
    optimizer: Optimizer,
    device: str,
) -> dict[str, float]:
    """Train for one epoch and return metrics."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        # Forward
        logits = model(images)
        loss = loss_fn(logits, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Callable,
    device: str,
) -> dict[str, float]:
    """Evaluate on a validation or test set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = loss_fn(logits, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }
