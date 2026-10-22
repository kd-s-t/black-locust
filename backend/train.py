"""YOLO dataset upload and training jobs (Ultralytics)."""
from __future__ import annotations

import re
import shutil
import threading
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import db

ROOT = Path(__file__).resolve().parent
DATASETS_DIR = ROOT / "datasets"
MODELS_DIR = ROOT / "models"
RUNS_DIR = ROOT / "runs"

_jobs: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()


def _ensure_dirs() -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    db.init_db()
    _sync_existing_folders()


def _sync_existing_folders() -> None:
    """Register on-disk folders that are not yet in the DB (e.g. dropped manually)."""
    for entry in sorted(DATASETS_DIR.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        if db.get_dataset_by_folder(entry.name):
            continue
        yaml_path = _find_data_yaml(entry)
        task = _dataset_task(entry)
        now = datetime.now(timezone.utc).isoformat()
        db.insert_dataset(
            {
                "id": uuid.uuid4().hex,
                "name": entry.name,
                "folder": entry.name,
                "path": str(entry),
                "task": task,
                "has_data_yaml": yaml_path is not None,
                "data_yaml": str(yaml_path) if yaml_path else None,
                "ready": task in ("classify", "detect"),
                "original_filename": None,
                "created_at": now,
            }
        )


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_\-]+", "_", name.strip())
    return cleaned[:64] or f"dataset_{uuid.uuid4().hex[:8]}"


def _find_data_yaml(root: Path) -> Optional[Path]:
    direct = root / "data.yaml"
    if direct.is_file():
        return direct
    for path in root.rglob("data.yaml"):
        return path
    for path in root.rglob("data.yml"):
        return path
    return None


def _is_classification_layout(root: Path) -> bool:
    train = root / "train"
    if not train.is_dir():
        return False
    class_dirs = [p for p in train.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if len(class_dirs) < 1:
        return False
    img_ext = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    for d in class_dirs:
        if any(f.suffix.lower() in img_ext for f in d.iterdir() if f.is_file()):
            return True
    return False


def _dataset_task(root: Path) -> str:
    if _is_classification_layout(root):
        return "classify"
    if _find_data_yaml(root):
        return "detect"
    return "unknown"


def list_datasets() -> List[Dict[str, Any]]:
    _ensure_dirs()
    items: List[Dict[str, Any]] = []
    for row in db.list_datasets():
        folder = DATASETS_DIR / row["folder"]
        if not folder.is_dir():
            continue
        # Keep disk as source of truth for readiness if files changed
        yaml_path = _find_data_yaml(folder)
        task = _dataset_task(folder)
        items.append(
            {
                **row,
                "name": row["name"],
                "folder": row["folder"],
                "path": str(folder),
                "task": task,
                "has_data_yaml": yaml_path is not None,
                "data_yaml": str(yaml_path) if yaml_path else None,
                "ready": task in ("classify", "detect"),
            }
        )
    return items


def remove_dataset(ref: str) -> Dict[str, Any]:
    """Delete dataset DB row and folder under datasets/."""
    _ensure_dirs()
    row = db.get_dataset(ref) or db.get_dataset_by_folder(ref)
    if not row:
        # Match by display name (latest)
        for item in db.list_datasets():
            if item["name"] == _safe_name(ref) or item["folder"] == ref:
                row = item
                break
    if not row:
        raise FileNotFoundError(f"Dataset not found: {ref}")

    folder = DATASETS_DIR / row["folder"]
    # Safety: only delete inside datasets/
    if folder.resolve().parent != DATASETS_DIR.resolve():
        raise ValueError("Invalid dataset path")
    if folder.is_dir():
        shutil.rmtree(folder)
    db.delete_dataset(row["id"])
    return {"deleted": True, "id": row["id"], "folder": row["folder"], "name": row["name"]}


def save_dataset_zip(filename: str, content: bytes, name: Optional[str] = None) -> Dict[str, Any]:
    _ensure_dirs()
    display = _safe_name(name or Path(filename).stem)
    # Drop synced on-disk twin (folder name == display, no upload suffix)
    for row in db.list_datasets():
        if row["name"] == display and row["folder"] == display:
            twin = DATASETS_DIR / row["folder"]
            if twin.is_dir():
                shutil.rmtree(twin)
            db.delete_dataset(row["id"])
    dataset_id = uuid.uuid4().hex
    folder = f"{display}_{dataset_id[:8]}"
    target = DATASETS_DIR / folder
    target.mkdir(parents=True, exist_ok=True)

    zip_path = target / "_upload.zip"
    zip_path.write_bytes(content)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target)
    zip_path.unlink(missing_ok=True)

    children = [p for p in target.iterdir() if p.name != "__MACOSX"]
    if len(children) == 1 and children[0].is_dir() and not _find_data_yaml(target) and not _is_classification_layout(target):
        nested = children[0]
        for item in nested.iterdir():
            dest = target / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(item), str(dest))
        shutil.rmtree(nested)

    yaml_path = _find_data_yaml(target)
    task = _dataset_task(target)
    created_at = datetime.now(timezone.utc).isoformat()
    return db.insert_dataset(
        {
            "id": dataset_id,
            "name": display,
            "folder": folder,
            "path": str(target),
            "task": task,
            "has_data_yaml": yaml_path is not None,
            "data_yaml": str(yaml_path) if yaml_path else None,
            "ready": task in ("classify", "detect"),
            "original_filename": filename,
            "created_at": created_at,
        }
    )


def list_models() -> List[Dict[str, Any]]:
    _ensure_dirs()
    items: List[Dict[str, Any]] = []
    for path in MODELS_DIR.glob("*.pt"):
        stat = path.stat()
        items.append(
            {
                "name": path.name,
                "path": str(path),
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "_mtime": stat.st_mtime,
            }
        )
    items.sort(key=lambda m: m["_mtime"], reverse=True)
    for m in items:
        del m["_mtime"]
    return items


def remove_model(name: str) -> Dict[str, Any]:
    """Delete a .pt file from models/."""
    _ensure_dirs()
    stem = Path(name).name
    if not stem.endswith(".pt") or "/" in stem or "\\" in stem or stem.startswith("."):
        raise ValueError("Invalid model name")
    path = MODELS_DIR / stem
    if path.resolve().parent != MODELS_DIR.resolve():
        raise ValueError("Invalid model path")
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {stem}")
    path.unlink()
    return {"deleted": True, "name": stem}


def predict_image(model_name: str, image_bytes: bytes, filename: str = "image.jpg") -> Dict[str, Any]:
    """Run inference with a trained (or pretrained) model on one image."""
    import base64
    import io
    import tempfile

    _ensure_dirs()
    name = Path(model_name).name
    local = MODELS_DIR / name
    if local.is_file():
        weights = str(local.resolve())
    else:
        weights = name

    suffix = Path(filename).suffix.lower() or ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        suffix = ".jpg"

    from ultralytics import YOLO

    model = YOLO(weights)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name
    try:
        results = model.predict(source=tmp_path, verbose=False)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not results:
        raise ValueError("No prediction results")

    r0 = results[0]
    task = getattr(model, "task", None) or "unknown"
    predictions: List[Dict[str, Any]] = []
    image_b64: Optional[str] = None

    # Classification
    probs = getattr(r0, "probs", None)
    if probs is not None:
        names = r0.names or {}
        top5 = probs.top5
        top5conf = probs.top5conf.tolist() if hasattr(probs.top5conf, "tolist") else list(probs.top5conf)
        for idx, conf in zip(top5, top5conf):
            label = names.get(int(idx), str(idx)) if isinstance(names, dict) else str(idx)
            predictions.append({"label": label, "confidence": float(conf)})
        task = "classify"
    else:
        # Detection
        boxes = getattr(r0, "boxes", None)
        names = r0.names or {}
        if boxes is not None and len(boxes):
            for box in boxes:
                cls_id = int(box.cls[0]) if box.cls is not None else -1
                conf = float(box.conf[0]) if box.conf is not None else 0.0
                label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                xyxy = box.xyxy[0].tolist() if box.xyxy is not None else []
                predictions.append(
                    {
                        "label": label,
                        "confidence": conf,
                        "box": [float(x) for x in xyxy] if xyxy else None,
                    }
                )
        task = "detect"
        try:
            plotted = r0.plot()  # BGR numpy
            from PIL import Image

            rgb = plotted[:, :, ::-1]
            buf = io.BytesIO()
            Image.fromarray(rgb).save(buf, format="JPEG", quality=90)
            image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            image_b64 = None

    return {
        "model": name,
        "task": task,
        "predictions": predictions,
        "image_base64": image_b64,
    }


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def list_jobs() -> List[Dict[str, Any]]:
    with _lock:
        return [dict(j) for j in sorted(_jobs.values(), key=lambda x: x.get("created_at", ""), reverse=True)]


def _update_job(job_id: str, **kwargs: Any) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


def _output_model_name(requested: Optional[str], job_id: str) -> str:
    raw = (requested or "").strip() or job_id
    stem = _safe_name(Path(raw).stem)
    if not stem:
        stem = job_id
    return f"{stem}.pt"


def _assert_model_name_available(requested: Optional[str], job_id: str) -> str:
    saved = _output_model_name(requested, job_id)
    dest = MODELS_DIR / saved
    if dest.is_file():
        raise ValueError(f"Model name already exists: {saved}")
    return saved


def _run_train(
    job_id: str,
    data_path: str,
    base_model: str,
    epochs: int,
    imgsz: int,
    output_name: Optional[str] = None,
) -> None:
    _update_job(job_id, status="running", message="Training started", current_epoch=0)
    try:
        from ultralytics import YOLO

        model = YOLO(base_model)

        def on_train_epoch_end(trainer: Any) -> None:
            current = int(getattr(trainer, "epoch", 0)) + 1
            total = int(getattr(trainer, "epochs", epochs) or epochs)
            _update_job(
                job_id,
                current_epoch=current,
                message=f"Epoch {current}/{total}",
            )

        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        results = model.train(
            data=data_path,
            epochs=epochs,
            imgsz=imgsz,
            project=str(RUNS_DIR),
            name=job_id,
            exist_ok=True,
            verbose=False,
        )
        best = Path(getattr(results, "save_dir", RUNS_DIR / job_id)) / "weights" / "best.pt"
        if not best.is_file():
            best = RUNS_DIR / job_id / "weights" / "best.pt"
        if best.is_file():
            saved_name = _output_model_name(output_name, job_id)
            dest = MODELS_DIR / saved_name
            if dest.exists():
                raise RuntimeError(f"Model name already exists: {saved_name}")
            shutil.copy2(best, dest)
            _update_job(
                job_id,
                status="completed",
                message="Training completed",
                model_path=str(dest),
                model_name=saved_name,
            )
        else:
            _update_job(
                job_id,
                status="completed",
                message="Training finished but best.pt not found",
            )
    except Exception as e:
        _update_job(job_id, status="failed", message=str(e), error=str(e))


def _resolve_dataset_dir(dataset_ref: str) -> Path:
    """Resolve by DB id, folder, or display name (latest match)."""
    row = db.get_dataset(dataset_ref)
    if row:
        return DATASETS_DIR / row["folder"]
    by_folder = db.get_dataset_by_folder(dataset_ref)
    if by_folder:
        return DATASETS_DIR / by_folder["folder"]
    # Fallback: exact folder on disk
    direct = DATASETS_DIR / _safe_name(dataset_ref)
    if direct.is_dir():
        return direct
    # Latest upload with this display name
    for item in db.list_datasets():
        if item["name"] == _safe_name(dataset_ref) or item["folder"] == dataset_ref:
            return DATASETS_DIR / item["folder"]
    raise FileNotFoundError(f"Dataset not found: {dataset_ref}")


def _resolve_base_model(base_model: str) -> tuple[str, bool]:
    """Return (path_or_name for YOLO, is_local_trained)."""
    name = Path(base_model).name
    local = MODELS_DIR / name
    if local.is_file():
        return str(local.resolve()), True
    return name, False


def start_train(
    dataset_name: str,
    epochs: int = 50,
    imgsz: int = 224,
    base_model: str = "yolo26n-cls.pt",
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    _ensure_dirs()
    dataset_dir = _resolve_dataset_dir(dataset_name)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset not found: {dataset_name}")

    model_ref, is_local = _resolve_base_model(base_model)
    base_name = Path(base_model).name
    output_name = model_name

    task = _dataset_task(dataset_dir)
    if task == "classify":
        data_path = str(dataset_dir.resolve())
        if not is_local and "-cls" not in base_name:
            raise ValueError("Classification datasets need a *-cls.pt base model (e.g. yolo26n-cls.pt)")
    elif task == "detect":
        yaml_path = _find_data_yaml(dataset_dir)
        if not yaml_path:
            raise ValueError("Detection dataset needs data.yaml")
        data_path = str(yaml_path.resolve())
        if not is_local and "-cls" in base_name:
            raise ValueError("Detection datasets need a detect model (e.g. yolo26n.pt), not *-cls")
    else:
        raise ValueError(
            "Dataset must be classify (train/cat, train/dog) or detect (images + labels + data.yaml)"
        )

    with _lock:
        for job in _jobs.values():
            if job.get("status") in ("queued", "running"):
                raise RuntimeError("A training job is already in progress")

    meta = db.get_dataset_by_folder(dataset_dir.name)
    job_id = uuid.uuid4().hex[:12]
    saved_model_name = _assert_model_name_available(output_name, job_id)
    job = {
        "id": job_id,
        "dataset_id": meta["id"] if meta else None,
        "dataset_name": meta["name"] if meta else dataset_dir.name,
        "dataset_folder": dataset_dir.name,
        "data_yaml": data_path,
        "task": task,
        "base_model": base_name,
        "epochs": epochs,
        "current_epoch": 0,
        "imgsz": imgsz,
        "status": "queued",
        "message": "Queued",
        "model_path": None,
        "model_name": saved_model_name,
        "error": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with _lock:
        _jobs[job_id] = job

    thread = threading.Thread(
        target=_run_train,
        args=(job_id, data_path, model_ref, epochs, imgsz, output_name),
        daemon=True,
    )
    thread.start()
    return dict(job)
