from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from api.main import app


client = TestClient(app)


def _image_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (32, 32), color=(80, 140, 60)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_web_interface_is_available() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "LeafGuard" in response.text


def test_predict_rejects_non_image_upload() -> None:
    response = client.post(
        "/predict",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 415


def test_predict_accepts_image_upload(monkeypatch) -> None:
    expected = {
        "class_name": "healthy",
        "confidence": 0.9,
        "probabilities": {"angular_leaf_spot": 0.02, "bean_rust": 0.08, "healthy": 0.9},
    }
    monkeypatch.setattr("api.main.predict", lambda image: expected)

    response = client.post(
        "/predict",
        files={"file": ("leaf.png", _image_bytes(), "image/png")},
    )

    assert response.status_code == 200
    assert response.json() == expected