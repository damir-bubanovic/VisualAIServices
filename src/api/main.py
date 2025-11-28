from tempfile import NamedTemporaryFile
from typing import List

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from src.inference.analyzer import analyze_image


app = FastAPI(
    title="Visual AI Service",
    description="Microservice for basic image analysis (classification + blur score).",
    version="0.1.0",
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
    and run the analyze_image() pipeline.
    """
    # Save uploaded content to a temporary file on disk
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
