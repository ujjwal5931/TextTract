<<<<<<< HEAD
# TextTract
=======
# TextTract – Local OCR Pipeline

Two-stage OCR (text detection + text recognition) with **zero cloud** and **zero hallucination**: all processing runs locally using EasyOCR. Optional OpenCV pre-processing for noisy or scanned images.

## Structure

```
texttract/
├── app.py           # Streamlit web dashboard
├── ocr_pipeline.py  # Detection + recognition (EasyOCR)
├── preprocess.py    # OpenCV pre-processing (optional)
├── metrics.py       # CER computation & logging
├── benchmark.py     # Dataset evaluation script
├── models/
├── utils/
│   └── io.py        # Image/GT loading for benchmarking
└── requirements.txt
```

## Setup

```bash
cd texttract
pip install -r requirements.txt
```

First run will download EasyOCR language weights (English by default).

## Web Dashboard

```bash
streamlit run app.py
```

- **Upload** an image.
- Toggle **OpenCV pre-processing** (denoise, sharpen, binarize, deskew).
- View **image with bounding boxes** and **extracted text** (list + JSON).
- **Inference time** is shown.
- Optional: paste **ground truth** to see **Character Error Rate (CER)**.

## Benchmarking (CER on a dataset)

Place images in one folder and ground-truth `.txt` files (same base name as each image) in another, then:

```bash
python benchmark.py --images_dir ./images --gt_dir ./ground_truth [--preprocess] [--no-gpu]
```

Logs per-image CER and mean CER.

## Design Notes

- **Local only**: No images sent to cloud APIs; EasyOCR runs on your machine (CPU/GPU).
- **No generative correction**: Output is raw model text only; no LLM autocorrect or guessing.
- **Spatial accuracy**: Bounding boxes are returned as 4-point polygons; CER is computed when ground truth is available.
>>>>>>> be10065 (Pushed to main)
