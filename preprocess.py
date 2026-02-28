"""
OpenCV-based image pre-processing for OCR.
Prepares noisy/scanned images before passing to the detection & recognition pipeline.
No generative correction—only signal enhancement.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR/RGB to grayscale for consistent OCR input."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def denoise(image: np.ndarray) -> np.ndarray:
    """Reduce noise while preserving edges (non-generative)."""
    if len(image.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(image, None, 6, 6, 7, 21)
    return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)


def sharpen(image: np.ndarray) -> np.ndarray:
    """Light sharpening kernel to enhance character edges."""
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)


def adaptive_threshold(
    gray: np.ndarray,
    block_size: int = 11,
    c: int = 2,
) -> np.ndarray:
    """Adaptive binarization for uneven lighting (e.g. scanned docs)."""
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c
    )


def deskew(image: np.ndarray) -> np.ndarray:
    """Correct small rotation by detecting dominant line angle (optional)."""
    gray = to_grayscale(image) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    if lines is None or len(lines) == 0:
        return image
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 != 0:
            angles.append(np.degrees(np.arctan((y2 - y1) / (x2 - x1))))
    if not angles:
        return image
    median_angle = np.median(angles)
    if abs(median_angle) < 0.5:
        return image
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), median_angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def crop_to_document(
    image: np.ndarray,
    min_area_ratio: float = 0.05,
    padding: int = 10,
) -> np.ndarray:
    """
    Detect the main document/content region and crop to it.
    Use when the image is a screenshot containing UI—OCR will run only on the document.
    Returns original image if no clear document contour is found.
    """
    gray = to_grayscale(image)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(
        edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    h, w = image.shape[:2]
    area_thresh = (h * w) * min_area_ratio
    best_rect = None
    best_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < area_thresh:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            x, y, rw, rh = cv2.boundingRect(approx)
            if rw * rh > best_area and rw > 20 and rh > 20:
                best_area = rw * rh
                best_rect = (x, y, rw, rh)
    if best_rect is None:
        # Fallback: largest reasonable bounding rect
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < area_thresh:
                continue
            x, y, rw, rh = cv2.boundingRect(cnt)
            if rw * rh > best_area and rw > 20 and rh > 20:
                best_area = rw * rh
                best_rect = (x, y, rw, rh)
    if best_rect is None:
        return image
    x, y, rw, rh = best_rect
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w, x + rw + padding)
    y2 = min(h, y + rh + padding)
    return image[y1:y2, x1:x2].copy()


def resize_if_large(
    image: np.ndarray,
    max_side: int = 2000,
    interp: int = cv2.INTER_AREA,
) -> np.ndarray:
    """Downscale very large images to avoid OOM and speed up inference."""
    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        return image
    scale = max_side / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=interp)


def preprocess(
    image: np.ndarray,
    *,
    crop_document: bool = False,
    denoise_enable: bool = True,
    sharpen_enable: bool = False,
    binarize: bool = False,
    deskew_enable: bool = False,
    max_side: Optional[int] = 2000,
) -> np.ndarray:
    """
    Full pre-processing pipeline. Returns image ready for OCR.
    All steps are deterministic; no LLM or generative correction.
    """
    out = image.copy()
    if crop_document:
        out = crop_to_document(out)
    out = resize_if_large(out, max_side=max_side or 2000)
    if deskew_enable:
        out = deskew(out)
    if denoise_enable:
        out = denoise(out)
    if sharpen_enable:
        out = sharpen(out)
    if binarize:
        gray = to_grayscale(out)
        out = adaptive_threshold(gray)
        if len(image.shape) == 3:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    return out
