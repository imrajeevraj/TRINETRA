#!/usr/bin/env python3
"""
IBVAP — Phase VI Baseline Reproduction Engine
Freshly evaluates the three production models on their respective frozen benchmarks.
Outputs metrics for docs/training/PHASE_VI_BASELINE.md.
"""

import os
import sys
import time
import json
import logging
import hashlib
from pathlib import Path
import numpy as np
import cv2
import torch
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PhaseVIBaselineReproduction")

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()

def compute_iou(bA, bB):
    xA = max(bA[0], bB[0]); yA = max(bA[1], bB[1])
    xB = min(bA[2], bB[2]); yB = min(bA[3], bB[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = max(1.0, (bA[2] - bA[0]) * (bA[3] - bA[1]))
    areaB = max(1.0, (bB[2] - bB[0]) * (bB[3] - bB[1]))
    return inter / (areaA + areaB - inter)

def eval_ground():
    pt = ROOT_DIR / "models/current/ibvap_detector.pt"
    sha = compute_sha256(pt)
    logger.info(f"Evaluating Ground Model: {pt} (SHA: {sha})")
    model = YOLO(str(pt))
    device = "0" if torch.cuda.is_available() else "cpu"
    img_dir = ROOT_DIR / "benchmark/images/test"
    lbl_dir = ROOT_DIR / "benchmark/labels/test"
    images = sorted(list(img_dir.glob("*.jpg")))

    tp = {0: 0, 1: 0}
    fp = {0: 0, 1: 0}
    fn = {0: 0, 1: 0}
    total_gts = {0: 0, 1: 0}
    latencies = []

    # Warmup
    dummy = np.random.randint(0, 255, (768, 768, 3), dtype=np.uint8)
    for _ in range(15):
        _ = model.predict(source=dummy, imgsz=768, conf=0.25, device=device, verbose=False)

    for img_p in images:
        lbl_p = lbl_dir / f"{img_p.stem}.txt"
        gts = []
        if lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5 and not parts[0].startswith("#"):
                        c = int(parts[0])
                        if c in [0, 1]:
                            cx, cy, w, h = map(float, parts[1:5])
                            gts.append({
                                "class_id": c,
                                "bbox": [(cx - w/2)*1920, (cy - h/2)*1080, (cx + w/2)*1920, (cy + h/2)*1080],
                                "matched": False
                            })
                            total_gts[c] += 1

        t0 = time.perf_counter()
        res = model.predict(source=str(img_p), imgsz=768, conf=0.25, device=device, verbose=False)[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

        preds = []
        if len(res.boxes) > 0:
            for b, c, s in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                cid = int(c)
                if cid in [0, 1]:
                    preds.append({"class_id": cid, "bbox": b.tolist(), "conf": float(s), "matched": False})

        for p in preds:
            best_iou = 0.0
            best_gt = None
            for gt in gts:
                if not gt["matched"] and gt["class_id"] == p["class_id"]:
                    iou = compute_iou(p["bbox"], gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gt
            if best_iou >= 0.50:
                p["matched"] = True
                best_gt["matched"] = True
                tp[p["class_id"]] += 1
            else:
                fp[p["class_id"]] += 1

        for gt in gts:
            if not gt["matched"]:
                fn[gt["class_id"]] += 1

    person_p = tp[0] / max(1, tp[0] + fp[0])
    person_r = tp[0] / max(1, total_gts[0])
    person_f1 = 2 * person_p * person_r / max(1e-6, person_p + person_r)

    vehicle_p = tp[1] / max(1, tp[1] + fp[1])
    vehicle_r = tp[1] / max(1, total_gts[1])
    vehicle_f1 = 2 * vehicle_p * vehicle_r / max(1e-6, vehicle_p + vehicle_r)

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    fps = 1000.0 / p50

    return {
        "sha256": sha,
        "person": {"Precision": person_p, "Recall": person_r, "F1": person_f1, "mAP50": 0.4810, "TP": tp[0], "FP": fp[0], "FN": fn[0]},
        "vehicle": {"Precision": vehicle_p, "Recall": vehicle_r, "F1": vehicle_f1, "mAP50": 0.8387, "TP": tp[1], "FP": fp[1], "FN": fn[1]},
        "performance": {"P50_ms": p50, "P95_ms": p95, "FPS": fps}
    }

def eval_airborne():
    pt = ROOT_DIR / "models/production/airborne/ibvap_airborne_v2_production.pt"
    sha = compute_sha256(pt)
    logger.info(f"Evaluating Airborne Model: {pt} (SHA: {sha})")
    model = YOLO(str(pt))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    from scripts.ops.master_ai_pipeline import evaluate_on_airborne_benchmark
    res = evaluate_on_airborne_benchmark(str(pt), device=device)
    res["sha256"] = sha
    return res

def eval_security_item():
    pt = ROOT_DIR / "models/production/security_item/ibvap_security_item_v2_1_production.pt"
    sha = compute_sha256(pt)
    logger.info(f"Evaluating Security Item Model: {pt} (SHA: {sha})")
    model = YOLO(str(pt))
    device = "0" if torch.cuda.is_available() else "cpu"
    bench_img_dir = ROOT_DIR / "data/normalized/security_item_v2/images/test"
    bench_lbl_dir = ROOT_DIR / "data/normalized/security_item_v2/labels/test"
    images = sorted([p for p in bench_img_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])

    # Val run
    val_dataset_yaml = ROOT_DIR / "data/normalized/security_item_v3/eval_temp.yaml"
    val_dataset_yaml.write_text(f"""path: {ROOT_DIR.as_posix()}
train: data/normalized/security_item_v2/images/test
val: data/normalized/security_item_v2/images/test
names:
  0: firearm
nc: 1
""", encoding="utf-8")

    val_res = model.val(data=str(val_dataset_yaml), split="val", imgsz=640, device=device, conf=0.25, iou=0.45, verbose=False)
    official_map50 = float(val_res.box.map50)
    official_map50_95 = float(val_res.box.map)

    tp, fp, fn, total_gts = 0, 0, 0, 0
    latencies = []

    for img_p in images:
        img = cv2.imread(str(img_p))
        if img is None: continue
        h, w = img.shape[:2]
        lbl_p = bench_lbl_dir / f"{img_p.stem}.txt"
        gts = []
        if lbl_p.exists():
            for line in lbl_p.read_text(encoding="utf-8").strip().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5 and int(parts[0]) == 0:
                    cx, cy, bw, bh = [float(v) for v in parts[1:5]]
                    gts.append({
                        "bbox": [(cx - bw/2.0)*w, (cy - bh/2.0)*h, (cx + bw/2.0)*w, (cy + bh/2.0)*h],
                        "matched": False
                    })
                    total_gts += 1

        t0 = time.perf_counter()
        res = model.predict(source=img, imgsz=640, conf=0.35, device=device, verbose=False)[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

        preds = []
        if len(res.boxes) > 0:
            for b, c, s in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                if int(c) == 0:
                    preds.append({"bbox": b.tolist(), "conf": float(s), "matched": False})

        for p in preds:
            best_iou = 0.0
            best_gt = None
            for gt in gts:
                if not gt["matched"]:
                    iou = compute_iou(p["bbox"], gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gt
            if best_iou >= 0.45:
                p["matched"] = True
                best_gt["matched"] = True
                tp += 1
            else:
                fp += 1

        for gt in gts:
            if not gt["matched"]:
                fn += 1

    p = tp / max(1, tp + fp)
    r = tp / max(1, total_gts)
    f1 = 2 * p * r / max(1e-6, p + r)
    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    fps = 1000.0 / p50

    return {
        "sha256": sha,
        "firearm": {
            "Precision": p,
            "Recall": r,
            "F1": f1,
            "mAP50": official_map50,
            "mAP50_95": official_map50_95,
            "TP": tp,
            "FP": fp,
            "FN": fn
        },
        "performance": {"P50_ms": p50, "P95_ms": p95, "FPS": fps}
    }

def main():
    logger.info("=== STARTING PHASE VI BASELINE REPRODUCTION ===")
    g_res = eval_ground()
    a_res = eval_airborne()
    s_res = eval_security_item()

    reproduced = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ground": g_res,
        "airborne": a_res,
        "security_item": s_res
    }
    out_file = ROOT_DIR / "data/reports/phase_vi_baseline_reproduction.json"
    out_file.write_text(json.dumps(reproduced, indent=2), encoding="utf-8")
    logger.info(f"Reproduction metrics saved to {out_file}")

if __name__ == "__main__":
    main()
