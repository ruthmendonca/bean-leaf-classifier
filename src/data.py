"""Dataset and preprocessing utilities for the Beans image dataset."""

from dataclasses import dataclass
from typing import Any

import torch
from datasets import DatasetDict
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DataConfig:
    """Configurable preprocessing and batching parameters."""

    image_size: int = 224
    batch_size: int = 16
    num_workers: int = 0
    seed: int = 42


def build_transforms(config: DataConfig) -> dict[str, transforms.Compose]:
    """Build train and evaluation transforms with the same final format."""
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    evaluation = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    training = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return {"train": training, "validation": evaluation, "test": evaluation}


class BeansDataset(Dataset[tuple[torch.Tensor, int]]):
    """Adapt one Hugging Face split to the PyTorch Dataset protocol."""

    def __init__(self, split: Any, transform: transforms.Compose) -> None:
        self.split = split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        item = self.split[index]
        image = item["image"].convert("RGB")
        tensor = self.transform(image)
        return tensor, int(item["labels"])


def create_dataloaders(
    dataset: DatasetDict,
    config: DataConfig | None = None,
) -> dict[str, DataLoader[tuple[torch.Tensor, int]]]:
    """Create train, validation and test loaders from loaded HF splits."""
    config = config or DataConfig()
    dataset_transforms = build_transforms(config)
    loaders = {}

    for split_name in ("train", "validation", "test"):
        split_dataset = BeansDataset(
            dataset[split_name],
            dataset_transforms[split_name],
        )
        loaders[split_name] = DataLoader(
            split_dataset,
            batch_size=config.batch_size,
            shuffle=split_name == "train",
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return loaders
