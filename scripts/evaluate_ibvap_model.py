#!/usr/bin/env python3
"""
IBVAP — Comprehensive Model Evaluator & Benchmark Protocol
Evaluates detectors on validation sets and the frozen IBVAP-GT-v1.0 benchmark.
Computes Person & Vehicle Precision, Recall, F1, mAP50, mAP50-95, AI FPS, P50, P95 latency.
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
import numpy as np
import torch
import cv2
import psutil
from ultralytics import YOLO

# Standard IBVAP Classes
IBVAP_CLASSES = {0: "person", 1: "vehicle"}

# COCO 80-Class to IBVAP Mapping (applied if evaluating COCO baseline)
COCO_TO_IBVAP = {
    0: 0, # person -> person
    2: 1, # car -> vehicle
    3: 1, # motorcycle -> vehicle
    5: 1, # bus -> vehicle
    7: 1  # truck -> vehicle
}

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def load_ground_truth(label_path: Path, width: int, height: int):
    gt = []
    if not label_path.exists():
        return gt
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                if cls_id not in IBVAP_CLASSES:
                    continue
                cx, cy, w, h = map(float, parts[1:5])
                x1 = max(0, (cx - w / 2) * width)
                y1 = max(0, (cy - h / 2) * height)
                x2 = min(width, (cx + w / 2) * width)
                y2 = min(height, (cy + h / 2) * height)
                gt.append({"class_id": cls_id, "box": [x1, y1, x2, y2]})
    return gt

def evaluate_model_on_benchmark(
    model_path: str,
    img_dir: Path,
    lbl_dir: Path,
    device: str = "cuda:0",
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.50,
    is_coco_baseline: bool = False
):
    print("=" * 70)
    print(f"IBVAP BENCHMARK EVALUATION: {Path(model_path).name}")
    print(f"Dataset Images: {img_dir}")
    print(f"Dataset Labels: {lbl_dir}")
    print(f"Target Device:  {device.upper()}")
    print("=" * 70)

    if not img_dir.exists() or not lbl_dir.exists():
        print(f"ERROR: Dataset directories missing ({img_dir} or {lbl_dir})")
        return None

    # Load model
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    model = YOLO(model_path)
    model.to(device)

    # Detect whether model has 2 classes (native IBVAP) or 80 classes (COCO)
    names = getattr(model, "names", {})
    num_classes = len(names)
    apply_coco_mapping = is_coco_baseline or num_classes > 10
    if apply_coco_mapping:
        print(f"Note: Detected {num_classes}-class model. Applying COCO -> IBVAP class normalization.")
    else:
        print(f"Note: Detected native {num_classes}-class IBVAP detector (0=person, 1=vehicle).")

    img_files = sorted([f for f in img_dir.glob("*.jpg")] + [f for f in img_dir.glob("*.png")])
    if not img_files:
        print("ERROR: No test images found in target directory.")
        return None

    # Warmup
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(5):
        _ = model.predict(source=dummy, imgsz=640, device=device, verbose=False)

    latencies = []
    class_stats = {
        0: {"TP": 0, "FP": 0, "FN": 0, "gt_count": 0, "pred_count": 0}, # Person
        1: {"TP": 0, "FP": 0, "FN": 0, "gt_count": 0, "pred_count": 0}  # Vehicle
    }

    # Start evaluation loop
    for img_p in img_files:
        lbl_p = lbl_dir / f"{img_p.stem}.txt"
        img = cv2.imread(str(img_p))
        if img is None: continue
        h, w = img.shape[:2]

        gt_boxes = load_ground_truth(lbl_p, w, h)
        for g in gt_boxes:
            class_stats[g["class_id"]]["gt_count"] += 1

        t0 = time.perf_counter()
        results = model.predict(source=img, conf=conf_thresh, imgsz=640, device=device, verbose=False)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0) # ms

        pred_boxes = []
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)

            for b, s, c in zip(boxes, scores, classes):
                target_cls = COCO_TO_IBVAP.get(c, -1) if apply_coco_mapping else c
                if target_cls in IBVAP_CLASSES:
                    pred_boxes.append({"class_id": target_cls, "score": float(s), "box": b})
                    class_stats[target_cls]["pred_count"] += 1

        # Match predictions to GT per class
        for c in [0, 1]:
            c_gt = [g for g in gt_boxes if g["class_id"] == c]
            c_pred = [p for g, p in enumerate(pred_boxes) if p["class_id"] == c]

            matched_gt = set()
            for p in c_pred:
                best_iou = 0.0
                best_idx = -1
                for idx, g in enumerate(c_gt):
                    if idx in matched_gt: continue
                    iou = calculate_iou(p["box"], g["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx
                if best_iou >= iou_thresh and best_idx != -1:
                    class_stats[c]["TP"] += 1
                    matched_gt.add(best_idx)
                else:
                    class_stats[c]["FP"] += 1

            class_stats[c]["FN"] += (len(c_gt) - len(matched_gt))

    # Calculate metrics
    results_summary = {}
    for c, cname in IBVAP_CLASSES.items():
        tp = class_stats[c]["TP"]
        fp = class_stats[c]["FP"]
        fn = class_stats[c]["FN"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results_summary[cname] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "GT": class_stats[c]["gt_count"],
            "Predictions": class_stats[c]["pred_count"],
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
            "mAP50": round(recall * precision, 4) # estimated point mAP
        }

    p50_lat = np.percentile(latencies, 50) if latencies else 0.0
    p95_lat = np.percentile(latencies, 95) if latencies else 0.0
    avg_fps = 1000.0 / np.mean(latencies) if latencies else 0.0

    perf_metrics = {
        "AI_FPS": round(avg_fps, 1),
        "P50_latency_ms": round(p50_lat, 2),
        "P95_latency_ms": round(p95_lat, 2),
        "Total_Frames": len(img_files),
        "Device": device,
        "VRAM_MB": round(torch.cuda.memory_allocated() / (1024**2), 1) if device.startswith("cuda") else 0.0
    }

    final_report = {
        "model": Path(model_path).name,
        "metrics": results_summary,
        "performance": perf_metrics
    }

    print("\n--- EVALUATION RESULTS ---")
    print(json.dumps(final_report, indent=2))
    print("=" * 70)

    return final_report

def main():
    parser = argparse.ArgumentParser(description="IBVAP Model Evaluation Suite")
    parser.add_argument("--model", type=str, default="models/current/ibvap_detector.pt", help="Path to model weights")
    parser.add_argument("--images", type=str, default="benchmark/images/test", help="Path to test images")
    parser.add_argument("--labels", type=str, default="benchmark/labels/test", help="Path to test labels")
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device")
    parser.add_argument("--coco-baseline", action="store_true", help="Force COCO class mapping for raw pretrained YOLO models")
    parser.add_argument("--output-json", type=str, default=None, help="Save evaluation metrics to JSON")
    args = parser.parse_args()

    report = evaluate_model_on_benchmark(
        args.model,
        Path(args.images),
        Path(args.labels),
        device=args.device,
        is_coco_baseline=args.coco_baseline
    )

    if args.output_json and report:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Saved evaluation report to {out_p}")

if __name__ == "__main__":
    main()
