from tempfile import NamedTemporaryFile
from typing import List

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from src.inference.analyzer import analyze_image, analyze_image_cifar


app = FastAPI(
    title="Visual AI Service",
    description=(
        "Microservice for image analysis: ImageNet classification + blur score + "
        "segmentation mask, and CIFAR-10 classification."
    ),
    version="0.2.0",
)


class HealthResponse(BaseModel):
    status: str


class AnalyzeResponse(BaseModel):
    filename: str
    top1_label: str
    topk_labels: List[str]
    blur_score: float
    mask_path: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/analyze-image", response_model=AnalyzeResponse)
async def analyze_image_endpoint(file: UploadFile = File(...)) -> AnalyzeResponse:
    """
    Accept an uploaded image file, save it temporarily,
    and run the ImageNet-based analyze_image() pipeline
    (classification + blur score + segmentation mask).
    """
    contents = await file.read()
    with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    result = analyze_image(tmp_path)

    return AnalyzeResponse(
        filename=file.filename,
        top1_label=result["top1_label"],
        topk_labels=result["topk_labels"],
        blur_score=result["blur_score"],
        mask_path=result["mask_path"],
    )


@app.post("/analyze-image-cifar", response_model=AnalyzeResponse)
async def analyze_image_cifar_endpoint(file: UploadFile = File(...)) -> AnalyzeResponse:
    """
    Accept an uploaded image file, save it temporarily,
    and run the CIFAR-10 analyze_image_cifar() pipeline
    (CIFAR-10 classification + blur score, no mask).
    """
    contents = await file.read()
    with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    result = analyze_image_cifar(tmp_path)

    return AnalyzeResponse(
        filename=file.filename,
        top1_label=result["top1_label"],
        topk_labels=result["topk_labels"],
        blur_score=result["blur_score"],
        mask_path=result["mask_path"],
    )
