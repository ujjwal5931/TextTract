"""
Benchmark script: run OCR on a dataset and log CER.
Usage:
  python benchmark.py --images_dir ./images --gt_dir ./ground_truth [--preprocess]
"""

import argparse
import logging
from pathlib import Path

from metrics import aggregate_cer, compute_cer, setup_logging
from ocr_pipeline import run_ocr
from preprocess import preprocess
from utils.io import load_image, list_image_ground_truth_pairs

setup_logging()
logger = logging.getLogger("benchmark")


def main():
    parser = argparse.ArgumentParser(description="Benchmark OCR pipeline and compute CER")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory of images")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory of ground-truth .txt files (same stem as image)")
    parser.add_argument("--preprocess", action="store_true", help="Apply OpenCV pre-processing")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU")
    args = parser.parse_args()

    pairs = list_image_ground_truth_pairs(args.images_dir, args.gt_dir)
    if not pairs:
        logger.warning("No (image, gt) pairs found. Check paths and extensions.")
        return

    references = []
    hypotheses = []
    total_time = 0.0

    for img_path, gt_path in pairs:
        image = load_image(img_path)
        if args.preprocess:
            image = preprocess(image)
        with open(gt_path, "r", encoding="utf-8", errors="replace") as f:
            ref = f.read().strip()
        detections, elapsed = run_ocr(image, gpu=not getattr(args, "no_gpu", False))
        hyp = " ".join(d.get("text", "") for d in detections)
        references.append(ref)
        hypotheses.append(hyp)
        total_time += elapsed
        cer_sample = compute_cer(ref, hyp)
        logger.info("%s CER=%.4f time=%.3fs", Path(img_path).name, cer_sample, elapsed)

    mean_cer = aggregate_cer(references, hypotheses)
    logger.info("Mean CER = %.4f over %d samples", mean_cer, len(pairs))
    logger.info("Total inference time = %.3fs", total_time)


if __name__ == "__main__":
    main()
