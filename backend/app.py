"""YOLO dataset upload + train API."""
import zipfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import train as yolo

app = FastAPI(
    title="BlackLocust",
    description="An AI trainer powered by Yolo26",
)
yolo._ensure_dirs()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/models-static", StaticFiles(directory=str(yolo.MODELS_DIR)), name="models")


class TrainRequest(BaseModel):
    dataset_name: str
    epochs: int = Field(default=50, ge=1, le=500)
    imgsz: int = Field(default=224, ge=32, le=1280)
    base_model: str = "yolo26n-cls.pt"
    model_name: Optional[str] = None


DETECT_MODELS = {"yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"}
CLS_MODELS = {"yolo26n-cls.pt", "yolo26s-cls.pt", "yolo26m-cls.pt", "yolo26l-cls.pt", "yolo26x-cls.pt"}
ALLOWED = DETECT_MODELS | CLS_MODELS


def _model_allowed(name: str) -> bool:
    stem = Path(name).name
    if stem in ALLOWED:
        return True
    return (yolo.MODELS_DIR / stem).is_file()


@app.get("/api/datasets")
def list_datasets():
    return yolo.list_datasets()


@app.post("/api/datasets", status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
):
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Upload a YOLO dataset as a .zip file")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        return yolo.save_dataset_zip(file.filename, content, name=name)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid zip file")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/datasets/{ref}")
def delete_dataset(ref: str):
    try:
        return yolo.remove_dataset(ref)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/models")
def list_models():
    return yolo.list_models()


@app.delete("/api/models/{name}")
def delete_model(name: str):
    try:
        return yolo.remove_model(name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/jobs")
def list_jobs():
    return yolo.list_jobs()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = yolo.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/api/train", status_code=status.HTTP_202_ACCEPTED)
def start_train(body: TrainRequest):
    if not _model_allowed(body.base_model):
        raise HTTPException(
            status_code=400,
            detail=f"base_model must be a pretrained YOLO weight or a file in models/ ({sorted(ALLOWED)})",
        )
    try:
        return yolo.start_train(
            dataset_name=body.dataset_name,
            epochs=body.epochs,
            imgsz=body.imgsz,
            base_model=body.base_model,
            model_name=body.model_name,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Form(...),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Upload an image")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    stem = Path(model).name
    local = yolo.MODELS_DIR / stem
    if not local.is_file() and stem not in ALLOWED:
        raise HTTPException(status_code=404, detail=f"Model not found: {stem}")
    try:
        return yolo.predict_image(stem, content, filename=file.filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8010, reload=True)
