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
    """Simple response model for the health check endpoint."""

    status: str


class AnalyzeResponse(BaseModel):
    """
    Common response model for image analysis endpoints.

    Fields
    ------
    filename:
        Name of the uploaded file.
    top1_label:
        Most probable predicted class.
    topk_labels:
        List of top-k predicted class names.
    blur_score:
        Numeric sharpness / blur metric (higher is sharper).
    mask_path:
        Path to a generated segmentation mask on disk.
        Empty string when the endpoint does not produce a mask.
    """

    filename: str
    top1_label: str
    topk_labels: List[str]
    blur_score: float
    mask_path: str


def _save_upload_to_tempfile(upload: UploadFile) -> str:
    """
    Persist an uploaded file to a temporary location on disk.

    Returns
    -------
    Path to the created temporary file as a string.
    """
    contents = upload.file.read()
    with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(contents)
        return tmp_file.name


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """
    Health check endpoint used for liveness monitoring.

    Returns a static status payload.
    """
    return HealthResponse(status="ok")


@app.post("/analyze-image", response_model=AnalyzeResponse)
async def analyze_imagenet_image(file: UploadFile = File(...)) -> AnalyzeResponse:
    """
    Analyze an image using the ImageNet-based pipeline.

    The pipeline performs:
    - ImageNet classification (pretrained ResNet18)
    - Blur score computation
    - Foreground segmentation mask generation
    """
    # Save uploaded content to a temporary file on disk
    contents = await file.read()
    with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(contents)
        temp_path = tmp_file.name

    analysis_result = analyze_image(temp_path)

    return AnalyzeResponse(
        filename=file.filename,
        top1_label=analysis_result["top1_label"],
        topk_labels=analysis_result["topk_labels"],
        blur_score=analysis_result["blur_score"],
        mask_path=analysis_result["mask_path"],
    )


@app.post("/analyze-image-cifar", response_model=AnalyzeResponse)
async def analyze_cifar_image(file: UploadFile = File(...)) -> AnalyzeResponse:
    """
    Analyze an image using the fine-tuned CIFAR-10 pipeline.

    The pipeline performs:
    - CIFAR-10 classification (fine-tuned ResNet18)
    - Blur score computation
    - No segmentation mask (mask_path will be an empty string)
    """
    contents = await file.read()
    with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(contents)
        temp_path = tmp_file.name

    analysis_result = analyze_image_cifar(temp_path)

    return AnalyzeResponse(
        filename=file.filename,
        top1_label=analysis_result["top1_label"],
        topk_labels=analysis_result["topk_labels"],
        blur_score=analysis_result["blur_score"],
        mask_path=analysis_result["mask_path"],
    )
