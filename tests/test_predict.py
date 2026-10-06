from io import BytesIO

from PIL import Image

from src.predict import predict


def _image_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (500, 500), color=(80, 140, 60)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_predict_returns_valid_classification() -> None:
    result = predict(_image_bytes(), device="cpu")

    assert result["class_name"] in {"angular_leaf_spot", "bean_rust", "healthy"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert set(result["probabilities"]) == {
        "angular_leaf_spot",
        "bean_rust",
        "healthy",
    }
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-5