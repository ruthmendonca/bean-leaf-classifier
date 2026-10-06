"""HTTP API for LeafGuard bean leaf classification."""

from typing import Annotated
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from src.predict import predict


app = FastAPI(
    title="LeafGuard API",
    description="Classify bean leaf images with a MobileNetV3 model.",
    version="1.0.0",
)

WEB_INDEX = Path(__file__).parent / "static" / "index.html"


@app.get("/", include_in_schema=False)
def web_interface() -> FileResponse:
    return FileResponse(WEB_INDEX)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict_endpoint(file: Annotated[UploadFile, File(...)]) -> dict[str, object]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="O arquivo precisa ser uma imagem.")

    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="O arquivo enviado esta vazio.")
        return predict(image_bytes)
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Nao foi possivel processar a imagem: {error}",
        ) from error