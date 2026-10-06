"""Model architecture and Transfer Learning utilities."""

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torchvision import models


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for MobileNetV3-Small backbone and classifier head."""

    num_classes: int = 3
    dropout_rate: float = 0.2
    backbone: Literal["mobilenetv3_small"] = "mobilenetv3_small"


def create_model(config: ModelConfig | None = None, pretrained: bool = True) -> nn.Module:
    """Create MobileNetV3-Small model with custom classifier for 3 classes.
    
    Args:
        config: Model configuration (num_classes, dropout_rate, backbone type).
        pretrained: If True, load ImageNet weights. If False, use random init.
    """
    config = config or ModelConfig()

    # Fail loudly when pretrained weights are requested but unavailable.
    # Falling back to random initialization would invalidate Transfer Learning.
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)

    # Replace classifier
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=config.dropout_rate),
        nn.Linear(256, config.num_classes),
    )

    return model


def freeze_backbone(model: nn.Module, freeze: bool = True) -> None:
    """Freeze or unfreeze the backbone (feature extractor) of the model."""
    for name, param in model.named_parameters():
        # All parameters except those in 'classifier' are in the backbone
        if "classifier" not in name:
            param.requires_grad = not freeze


def get_trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return only trainable parameters."""
    return [p for p in model.parameters() if p.requires_grad]


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in get_trainable_parameters(model))
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
