"""
TextTract – Local OCR Web Dashboard.
Upload image → optional pre-processing → detection + recognition → visualize bboxes & text.
"""

import json
import time

import cv2
import numpy as np
import streamlit as st

from metrics import compute_cer
from ocr_pipeline import bbox_to_xyxy, run_ocr
from preprocess import preprocess

st.set_page_config(page_title="TextTract OCR", page_icon="📄", layout="wide")

st.title("TextTract – Local OCR Pipeline")
st.caption("Two-stage text detection & recognition · Zero-cloud, zero hallucination")

# Sidebar: options
with st.sidebar:
    st.header("Options")
    crop_document = st.checkbox(
        "Crop to document only",
        value=False,
        help="Detect and crop to the main document/receipt. Use when the image is a screenshot so UI text (e.g. browser buttons) is excluded from OCR.",
    )
    use_preprocess = st.checkbox("Use OpenCV pre-processing", value=False)
    if use_preprocess:
        denoise = st.checkbox("Denoise", value=True)
        sharpen = st.checkbox("Sharpen", value=False)
        binarize = st.checkbox("Binarize (adaptive threshold)", value=False)
        deskew = st.checkbox("Deskew", value=False)
        max_side = st.number_input("Max side (px)", min_value=256, value=2000, step=256)
    else:
        denoise = sharpen = binarize = deskew = False
        max_side = 2000
    gpu = st.checkbox("Use GPU (if available)", value=True)
    # Optional: CER if user provides ground truth
    st.divider()
    st.subheader("Evaluation (optional)")
    ground_truth = st.text_area("Ground truth (for CER)", height=100, placeholder="Paste reference text to compute CER")

# Main: upload
uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "webp"])
st.caption("Tip: For best results upload only the receipt/document. If the image is a screenshot with UI, enable **Crop to document only** in the sidebar.")
if not uploaded:
    st.info("Upload an image to run the OCR pipeline.")
    st.stop()

# Decode image
file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
if image is None:
    st.error("Could not decode image.")
    st.stop()

# Preprocess (optional): crop to document first, then other steps
if crop_document or use_preprocess:
    image_for_ocr = preprocess(
        image,
        crop_document=crop_document,
        denoise_enable=denoise if use_preprocess else False,
        sharpen_enable=sharpen,
        binarize=binarize,
        deskew_enable=deskew,
        max_side=max_side,
    )
else:
    image_for_ocr = image.copy()

# Run OCR
with st.spinner("Running local OCR (detection + recognition)…"):
    t0 = time.perf_counter()
    detections, pipeline_time = run_ocr(image_for_ocr, gpu=gpu)
    total_time = time.perf_counter() - t0

# CER (only if ground truth provided)
cer_value = None
if ground_truth and ground_truth.strip():
    hypothesis = " ".join(d.get("text", "") for d in detections)
    cer_value = compute_cer(ground_truth.strip(), hypothesis)

# Visualization: draw bboxes on original (or preprocessed) image
vis_image = image_for_ocr.copy()
if len(vis_image.shape) == 2:
    vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)

for i, d in enumerate(detections):
    x1, y1, x2, y2 = bbox_to_xyxy(d["bbox"])
    color = (0, 255, 0)
    cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
    label = d["text"][:30] + "…" if len(d.get("text", "")) > 30 else d.get("text", "")
    if label:
        cv2.putText(
            vis_image, label, (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )

# Layout: image | results
col1, col2 = st.columns(2)
with col1:
    st.subheader("Image with bounding boxes")
    # Streamlit expects RGB
    vis_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
    st.image(vis_rgb, use_container_width=True)

with col2:
    st.subheader("Extracted text & metadata")
    st.metric("Inference time", f"{pipeline_time:.3f} s")
    st.metric("Total time", f"{total_time:.3f} s")
    st.metric("Detections", len(detections))
    if cer_value is not None:
        st.metric("CER", f"{cer_value:.4f}")
        with st.expander("CER formula"):
            st.latex(r"\mathrm{CER} = \frac{d(r, h)}{|r|}")
            st.latex(r"d(r,h) = \text{LevenshteinDistance}(\,r, h\,)")
            st.markdown(
                "- **r**: reference (ground truth)\n"
                "- **h**: hypothesis (OCR output)\n"
                "- **|r|**: number of characters in reference\n"
                "- **d(r,h)**: character-level edit distance\n"
            )

    # List of text per bbox
    if detections:
        for i, d in enumerate(detections):
            st.text(f"[{i+1}] {d.get('text', '')} (conf: {d.get('conf', 0):.2f})")
        st.divider()
        # JSON output
        st.subheader("JSON output")
        detections_json = [
            {"index": i + 1, "bbox": d["bbox"], "text": d.get("text", ""), "confidence": d.get("conf", 0)}
            for i, d in enumerate(detections)
        ]
        st.code(json.dumps(detections_json, indent=2), language="json")

        st.subheader("Run summary (includes CER if provided)")
        summary = {
            "inference_time_s": round(pipeline_time, 6),
            "total_time_s": round(total_time, 6),
            "detections": len(detections),
        }
        if cer_value is not None:
            summary["cer"] = float(cer_value)
        st.code(json.dumps(summary, indent=2), language="json")
    else:
        st.write("No text detected.")

if cer_value is not None:
    st.caption("CER: lower is better. Ref = ground truth; hyp = concatenated extracted text.")
