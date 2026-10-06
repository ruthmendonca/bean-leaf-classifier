"""Evaluation utilities including precision, recall, F1-score and confusion matrix."""

from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader


def evaluate_detailed(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    class_names: list[str] | None = None,
) -> dict:
    """Evaluate model with precision, recall, F1, and confusion matrix.

    Args:
        model: PyTorch model.
        loader: DataLoader for evaluation.
        device: Device to run evaluation on.
        class_names: List of class names for reporting.

    Returns:
        Dictionary with metrics: accuracy, precision, recall, f1, cm.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="weighted", zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "cm": confusion_matrix(all_labels, all_preds),
        "predictions": all_preds,
        "labels": all_labels,
    }

    if class_names:
        per_class = {}
        for i, name in enumerate(class_names):
            mask = all_labels == i
            if mask.sum() > 0:
                per_class[name] = {
                    "precision": precision_score(
                        all_labels,
                        all_preds,
                        labels=[i],
                        average="macro",
                        zero_division=0,
                    ),
                    "recall": recall_score(
                        all_labels,
                        all_preds,
                        labels=[i],
                        average="macro",
                        zero_division=0,
                    ),
                    "f1": f1_score(
                        all_labels,
                        all_preds,
                        labels=[i],
                        average="macro",
                        zero_division=0,
                    ),
                    "support": mask.sum(),
                }
        metrics["per_class"] = per_class

    return metrics


def print_metrics(metrics: dict, class_names: list[str] | None = None) -> None:
    """Pretty-print evaluation metrics."""
    print("Overall Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")

    if "per_class" in metrics and class_names:
        print("\nPer-class Metrics:")
        for class_name in class_names:
            if class_name in metrics["per_class"]:
                pc = metrics["per_class"][class_name]
                print(f"  {class_name}:")
                print(f"    Precision: {pc['precision']:.4f}, Recall: {pc['recall']:.4f}, F1: {pc['f1']:.4f}, Support: {pc['support']}")

    print("\nConfusion Matrix:")
    print(metrics["cm"])
