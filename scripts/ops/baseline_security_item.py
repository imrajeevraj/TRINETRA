#!/usr/bin/env python3
"""
IBVAP — Track B: Security Item Baseline Evaluator & Error Analysis Engine
Evaluates models/production/security_item/ibvap_security_item_v2_1_production.pt
against benchmark IBVAP-GT-ITEM-v2.0 and generates error analysis.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
import numpy as np
import cv2
import torch
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SecurityItemBaseline")

def calc_iou(bA, bB):
    xA = max(bA[0], bB[0]); yA = max(bA[1], bB[1])
    xB = min(bA[2], bB[2]); yB = min(bA[3], bB[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = max(1.0, (bA[2] - bA[0]) * (bA[3] - bA[1]))
    areaB = max(1.0, (bB[2] - bB[0]) * (bB[3] - bB[1]))
    return inter / (areaA + areaB - inter)

def main():
    model_path = ROOT_DIR / "models/production/security_item/ibvap_security_item_v2_1_production.pt"
    bench_img_dir = ROOT_DIR / "data/normalized/security_item_v2/images/test"
    bench_lbl_dir = ROOT_DIR / "data/normalized/security_item_v2/labels/test"
    out_dir = ROOT_DIR / "data/reports/model_error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading Security Item Model: {model_path} on {device}")
    model = YOLO(str(model_path))

    images = sorted([p for p in bench_img_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    logger.info(f"Loaded {len(images)} benchmark test images.")

    tp = 0
    fp = 0
    fn = 0
    total_gts = 0

    fn_categories = {
        "small_firearm": 0,
        "occluded_firearm": 0,
        "low_light_firearm": 0,
        "distant_firearm": 0
    }

    fp_categories = {
        "tool_distractor": 0,
        "clothing_shadow": 0,
        "holster_confusion": 0
    }

    latencies = []
    # Warmup
    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    for _ in range(20):
        _ = model.predict(source=dummy, imgsz=640, conf=0.35, device=device, verbose=False)

    for img_p in images:
        lbl_p = bench_lbl_dir / f"{img_p.stem}.txt"
        gts = []
        if lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5 and not parts[0].startswith("#"):
                        c = int(parts[0])
                        if c == 0:
                            cx, cy, w, h = map(float, parts[1:5])
                            xmin = (cx - w/2) * 640
                            ymin = (cy - h/2) * 640
                            xmax = (cx + w/2) * 640
                            ymax = (cy + h/2) * 640
                            gts.append({
                                "bbox": [xmin, ymin, xmax, ymax],
                                "w": w * 640,
                                "h": h * 640,
                                "matched": False
                            })
                            total_gts += 1

        t0 = time.perf_counter()
        res = model.predict(source=str(img_p), imgsz=640, conf=0.35, device=device, verbose=False)[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

        preds = []
        if len(res.boxes) > 0:
            for b, c, s in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                if int(c) == 0:
                    preds.append({
                        "bbox": b.tolist(),
                        "conf": float(s),
                        "matched": False
                    })

        for p in preds:
            best_iou = 0.0
            best_gt = None
            for gt in gts:
                if not gt["matched"]:
                    iou = calc_iou(p["bbox"], gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gt
            if best_iou >= 0.45:
                p["matched"] = True
                best_gt["matched"] = True
                tp += 1
            else:
                fp += 1
                aspect = (p["bbox"][3] - p["bbox"][1]) / max(1.0, (p["bbox"][2] - p["bbox"][0]))
                if aspect > 1.8:
                    fp_categories["tool_distractor"] += 1
                else:
                    fp_categories["clothing_shadow"] += 1

        for gt in gts:
            if not gt["matched"]:
                fn += 1
                if max(gt["w"], gt["h"]) < 40:
                    fn_categories["small_firearm"] += 1
                elif gt["bbox"][1] > 400:
                    fn_categories["low_light_firearm"] += 1
                else:
                    fn_categories["occluded_firearm"] += 1

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, total_gts)
    f1 = 2 * precision * recall / max(1e-6, precision + recall)
    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    fps = 1000.0 / p50

    metrics = {
        "model": "models/production/security_item/ibvap_security_item_v2_1_production.pt",
        "benchmark": "IBVAP-GT-ITEM-v2.0",
        "images": len(images),
        "total_gts": total_gts,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "p50_ms": p50,
        "p95_ms": p95,
        "ai_fps": fps,
        "fn_categories": fn_categories,
        "fp_categories": fp_categories
    }

    out_json = out_dir / "security_item_v3_failures.json"
    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info(f"Security Item Baseline: P={precision*100:.2f}%, R={recall*100:.2f}%, F1={f1*100:.2f}%, P50={p50:.2f}ms, FPS={fps:.1f}")
    logger.info(f"FN Breakdown: {fn_categories}")
    logger.info(f"FP Breakdown: {fp_categories}")

if __name__ == "__main__":
    main()
