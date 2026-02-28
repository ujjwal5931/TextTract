"""
Two-stage local OCR pipeline: Text Detection + Text Recognition.
Zero-cloud, zero generative guesswork—only reads what is visibly in the image.
"""

import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Lazy init to avoid loading at import time
_reader = None

# Project-local model dir to avoid ~/.EasyOCR permission or corrupted temp.zip issues
def _model_storage_dir() -> str:
    base = Path(__file__).resolve().parent
    path = base / "models" / "easyocr"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _get_reader(lang: List[str] = None, gpu: bool = True):
    """Lazy-load EasyOCR reader (local only). Uses project-local model dir."""
    global _reader
    if _reader is None:
        import easyocr
        storage = _model_storage_dir()
        _reader = easyocr.Reader(
            lang or ["en"],
            gpu=gpu,
            verbose=False,
            model_storage_directory=storage,
        )
    return _reader


def run_ocr(
    image: np.ndarray,
    lang: List[str] = None,
    gpu: bool = True,
) -> Tuple[List[dict], float]:
    """
    Run detection + recognition on image. Returns list of detections and inference time.
    Each detection: {"bbox": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], "text": str, "conf": float}
    No post-processing or LLM correction—raw model output only.
    """
    reader = _get_reader(lang=lang, gpu=gpu)
    t0 = time.perf_counter()
    results = reader.readtext(image)
    elapsed = time.perf_counter() - t0

    detections = []
    for (pts, text, conf) in results:
        # pts: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] in order
        detections.append({
            "bbox": [list(map(float, p)) for p in pts],
            "text": text.strip() if text else "",
            "conf": float(conf),
        })
    return detections, elapsed


def bbox_to_xyxy(bbox: List[List[float]]) -> Tuple[int, int, int, int]:
    """Convert 4-point bbox to (x_min, y_min, x_max, y_max) for drawing."""
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
