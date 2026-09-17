#!/usr/bin/env python3
"""
IBVAP — Track A: Ground Production Failure Mode Inventory Engine
Evaluates models/current/ibvap_detector.pt on IBVAP-GT-v1.0 and generates
an empirical inventory of False Positives and False Negatives.
"""

import os
import sys
import json
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
logger = logging.getLogger("GroundV5FailureInventory")

def calc_iou(bA, bB):
    xA = max(bA[0], bB[0]); yA = max(bA[1], bB[1])
    xB = min(bA[2], bB[2]); yB = min(bA[3], bB[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = max(1.0, (bA[2] - bA[0]) * (bA[3] - bA[1]))
    areaB = max(1.0, (bB[2] - bB[0]) * (bB[3] - bB[1]))
    return inter / (areaA + areaB - inter)

def main():
    model_path = ROOT_DIR / "models/current/ibvap_detector.pt"
    bench_img_dir = ROOT_DIR / "benchmark/images/test"
    bench_lbl_dir = ROOT_DIR / "benchmark/labels/test"
    out_dir = ROOT_DIR / "data/reports/model_error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading Ground Production Model: {model_path} on {device}")
    model = YOLO(str(model_path))

    images = sorted(list(bench_img_dir.glob("*.jpg")))
    logger.info(f"Evaluating across {len(images)} benchmark test keyframes...")

    fn_inventory = []
    fp_inventory = []
    tp_inventory = []

    fn_categories = {
        "small_person": 0,
        "distant_person": 0,
        "crowded_person": 0,
        "shadow_low_light": 0,
        "camera_perspective": 0,
        "vehicle_fn": 0
    }

    fp_categories = {
        "vertical_pole_fence": 0,
        "tree_shadow_foliage": 0,
        "structure_glare": 0,
        "vehicle_artifact": 0
    }

    for img_p in images:
        lbl_p = bench_lbl_dir / f"{img_p.stem}.txt"
        gts = []
        if lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        # convert to xyxy
                        xmin = (cx - w/2) * 1920
                        ymin = (cy - h/2) * 1080
                        xmax = (cx + w/2) * 1920
                        ymax = (cy + h/2) * 1080
                        gts.append({
                            "class_id": cls_id,
                            "bbox": [xmin, ymin, xmax, ymax],
                            "w": w * 1920,
                            "h": h * 1080,
                            "matched": False
                        })

        res = model.predict(source=str(img_p), imgsz=768, conf=0.25, device=device, verbose=False)[0]
        preds = []
        if len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            for b, c, s in zip(boxes, clss, confs):
                preds.append({
                    "class_id": c,
                    "bbox": b.tolist(),
                    "conf": float(s),
                    "matched": False
                })

        # Match preds to GTs
        for p in preds:
            best_iou = 0.0
            best_gt = None
            for gt in gts:
                if not gt["matched"] and gt["class_id"] == p["class_id"]:
                    iou = calc_iou(p["bbox"], gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gt
            if best_iou >= 0.50:
                p["matched"] = True
                best_gt["matched"] = True
                tp_inventory.append({
                    "image": img_p.name,
                    "class_id": p["class_id"],
                    "conf": p["conf"],
                    "iou": best_iou
                })
            else:
                # False Positive
                cls_name = "person" if p["class_id"] == 0 else "vehicle"
                # Classify FP
                if p["class_id"] == 0:
                    aspect = (p["bbox"][3] - p["bbox"][1]) / max(1.0, (p["bbox"][2] - p["bbox"][0]))
                    if aspect > 2.5:
                        cat = "vertical_pole_fence"
                    elif p["bbox"][1] > 700:
                        cat = "tree_shadow_foliage"
                    else:
                        cat = "structure_glare"
                else:
                    cat = "vehicle_artifact"
                fp_categories[cat] += 1
                fp_inventory.append({
                    "image": img_p.name,
                    "class_id": p["class_id"],
                    "conf": p["conf"],
                    "bbox": p["bbox"],
                    "category": cat
                })

        # Unmatched GTs are False Negatives
        for gt in gts:
            if not gt["matched"]:
                if gt["class_id"] == 0:
                    if gt["h"] < 60:
                        cat = "distant_person"
                    elif gt["h"] < 90:
                        cat = "small_person"
                    elif gt["bbox"][1] > 700:
                        cat = "shadow_low_light"
                    else:
                        cat = "camera_perspective"
                else:
                    cat = "vehicle_fn"
                fn_categories[cat] += 1
                fn_inventory.append({
                    "image": img_p.name,
                    "class_id": gt["class_id"],
                    "bbox": gt["bbox"],
                    "h": gt["h"],
                    "category": cat
                })

    logger.info(f"Inventory Summary: TP={len(tp_inventory)}, FP={len(fp_inventory)}, FN={len(fn_inventory)}")
    logger.info(f"FN Breakdown: {fn_categories}")
    logger.info(f"FP Breakdown: {fp_categories}")

    out_json = out_dir / "ground_v5_failure_inventory.json"
    summary_data = {
        "model": "models/current/ibvap_detector.pt",
        "benchmark": "IBVAP-GT-v1.0",
        "total_images": len(images),
        "counts": {
            "TP": len(tp_inventory),
            "FP": len(fp_inventory),
            "FN": len(fn_inventory)
        },
        "fn_categories": fn_categories,
        "fp_categories": fp_categories,
        "sample_false_negatives": fn_inventory[:50],
        "sample_false_positives": fp_inventory[:50]
    }
    def default_serializer(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Type {type(obj)} not serializable")

    out_json.write_text(json.dumps(summary_data, indent=2, default=default_serializer), encoding="utf-8")
    logger.info(f"Failure inventory saved to {out_json}")

if __name__ == "__main__":
    main()
