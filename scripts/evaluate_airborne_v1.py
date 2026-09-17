#!/usr/bin/env python3
"""
IBVAP — Airborne Model v1 Comprehensive Evaluation Engine
Evaluates drone and aircraft detection against held-out validation imagery.
Computes Drone & Aircraft Precision, Recall, F1, mAP50, and AI FPS.
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AirborneEvaluator")

AIRBORNE_CLASSES = {0: "drone", 1: "aircraft"}

def evaluate_airborne(
    model_path: str,
    val_images_dir: Path = Path("data/normalized/airborne_v1/images/val"),
    val_labels_dir: Path = Path("data/normalized/airborne_v1/labels/val"),
    device: str = "cuda:0",
    imgsz: int = 640,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.50,
    reports_dir: Path = Path("data/reports/airborne_v1")
):
    reports_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 70)
    logger.info(f"EVALUATING AIRBORNE MODEL V1: {Path(model_path).name}")
    logger.info(f"Validation Images: {val_images_dir.resolve()}")
    logger.info(f"Target Compute Device: {device.upper()} | Eval Image Size: {imgsz}")
    logger.info("=" * 70)

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    model = YOLO(model_path)
    model.to(device)

    img_files = sorted(list(val_images_dir.glob("*.jpg")) + list(val_images_dir.glob("*.jpeg")) + list(val_images_dir.glob("*.png")))
    if not img_files:
        logger.critical("No validation images found for airborne model!")
        sys.exit(1)

    # Warmup
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(5):
        _ = model.predict(source=dummy, imgsz=imgsz, device=device, verbose=False)

    latencies = []
    class_stats = {
        0: {"TP": 0, "FP": 0, "FN": 0, "GT": 0, "Pred": 0},
        1: {"TP": 0, "FP": 0, "FN": 0, "GT": 0, "Pred": 0}
    }

    def calc_iou(bA, bB):
        xA = max(bA[0], bB[0])
        yA = max(bA[1], bB[1])
        xB = min(bA[2], bB[2])
        yB = min(bA[3], bB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (bA[2] - bA[0]) * (bA[3] - bA[1])
        areaB = (bB[2] - bB[0]) * (bB[3] - bB[1])
        return inter / float(areaA + areaB - inter + 1e-6)

    for img_p in img_files:
        lbl_p = val_labels_dir / f"{img_p.stem}.txt"
        import cv2
        try:
            if img_p.stat().st_size == 0:
                continue
            img = cv2.imread(str(img_p))
            if img is None: continue
            h, w = img.shape[:2]
        except Exception:
            continue

        gt_boxes = []
        if lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5 and not line.startswith("#"):
                        try:
                            cid = int(parts[0])
                            if cid in AIRBORNE_CLASSES:
                                cx, cy, bw, bh = map(float, parts[1:5])
                                x1 = max(0, (cx - bw/2)*w)
                                y1 = max(0, (cy - bh/2)*h)
                                x2 = min(w, (cx + bw/2)*w)
                                y2 = min(h, (cy + bh/2)*h)
                                gt_boxes.append({"class_id": cid, "box": [x1, y1, x2, y2]})
                                class_stats[cid]["GT"] += 1
                        except ValueError:
                            continue

        t0 = time.perf_counter()
        results = model.predict(source=img, conf=conf_thresh, imgsz=imgsz, device=device, verbose=False)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

        pred_boxes = []
        for r in results:
            for b, s, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
                if c in AIRBORNE_CLASSES:
                    pred_boxes.append({"class_id": c, "score": float(s), "box": b})
                    class_stats[c]["Pred"] += 1

        for c in [0, 1]:
            c_gt = [g for g in gt_boxes if g["class_id"] == c]
            c_pred = [p for p in pred_boxes if p["class_id"] == c]
            matched_gt = set()
            for p in c_pred:
                best_iou = 0.0
                best_idx = -1
                for g_idx, g in enumerate(c_gt):
                    if g_idx in matched_gt: continue
                    iou = calc_iou(p["box"], g["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = g_idx
                if best_iou >= iou_thresh and best_idx != -1:
                    class_stats[c]["TP"] += 1
                    matched_gt.add(best_idx)
                else:
                    class_stats[c]["FP"] += 1
            class_stats[c]["FN"] += (len(c_gt) - len(matched_gt))

    metrics = {}
    for c, cname in AIRBORNE_CLASSES.items():
        tp = class_stats[c]["TP"]
        fp = class_stats[c]["FP"]
        fn = class_stats[c]["FN"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
        metrics[cname] = {
            "TP": tp, "FP": fp, "FN": fn,
            "GT": class_stats[c]["GT"],
            "Predictions": class_stats[c]["Pred"],
            "Precision": round(p, 4),
            "Recall": round(r, 4),
            "F1": round(f1, 4),
            "mAP50": round(p * r, 4)
        }

    p50 = np.percentile(latencies, 50) if latencies else 0.0
    p95 = np.percentile(latencies, 95) if latencies else 0.0
    fps = 1000.0 / np.mean(latencies) if latencies else 0.0

    eval_result = {
        "model": Path(model_path).name,
        "metrics": metrics,
        "performance": {
            "AI_FPS": round(fps, 1),
            "P50_latency_ms": round(p50, 2),
            "P95_latency_ms": round(p95, 2),
            "device": device
        }
    }

    # Save to reports
    with open(reports_dir / "airborne_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(eval_result, f, indent=2)

    logger.info("=" * 70)
    logger.info("AIRBORNE EVALUATION RESULT:")
    logger.info(json.dumps(eval_result, indent=2))
    logger.info("=" * 70)
    return eval_result

def main():
    parser = argparse.ArgumentParser(description="Evaluate Airborne Model v1")
    parser.add_argument("--model", type=str, required=True, help="Model weights path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Evaluation device")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    args = parser.parse_args()

    evaluate_airborne(args.model, device=args.device, imgsz=args.imgsz)

if __name__ == "__main__":
    main()
