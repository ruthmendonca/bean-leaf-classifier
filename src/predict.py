"""Single-image inference for the trained LeafGuard classifier."""

from io import BytesIO
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO

import torch
from PIL import Image

from .data import DataConfig, build_transforms
from .model import ModelConfig, create_model


DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "artifacts" / "mobilenetv3_beans_best.pt"


@lru_cache(maxsize=4)
def load_checkpoint(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    device: str | torch.device | None = None,
) -> tuple[torch.nn.Module, list[str], torch.device]:
    """Load the trained classifier and its class names from a local checkpoint."""
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(
        checkpoint_path,
        map_location=target_device,
        weights_only=False,
    )
    class_names = checkpoint.get("class_names", ["angular_leaf_spot", "bean_rust", "healthy"])
    model_config = checkpoint.get("model_config", ModelConfig(num_classes=len(class_names)))
    model = create_model(model_config, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(target_device)
    model.eval()
    return model, list(class_names), target_device


def _to_image(image: str | Path | Image.Image | bytes | BinaryIO) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            return opened.convert("RGB")
    if isinstance(image, bytes):
        with Image.open(BytesIO(image)) as opened:
            return opened.convert("RGB")
    with Image.open(image) as opened:
        return opened.convert("RGB")


def predict(
    image: str | Path | Image.Image | bytes | BinaryIO,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    device: str | torch.device | None = None,
) -> dict[str, object]:
    """Predict the bean leaf class for one image."""
    model, class_names, target_device = load_checkpoint(checkpoint_path, device)
    transform = build_transforms(DataConfig())['test']
    tensor = transform(_to_image(image)).unsqueeze(0).to(target_device)

    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
        class_index = int(torch.argmax(probabilities).item())

    return {
        "class_name": class_names[class_index],
        "confidence": float(probabilities[class_index].item()),
        "probabilities": {
            name: float(probabilities[index].item())
            for index, name in enumerate(class_names)
        },
    }