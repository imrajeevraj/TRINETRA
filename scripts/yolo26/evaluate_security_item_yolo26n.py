#!/usr/bin/env python3
"""
IBVAP — Security Item YOLO26n Evaluation (IBVAP-ITEM-Y26-001)
Evaluates against frozen IBVAP-GT-ITEM-v2.0. Exact mirror of evaluate_security_item_v2_1.py.

⚠ CRITICAL: The limitation "REAL-FIREARM-VIDEO VALIDATION NOT AVAILABLE"
MUST be preserved in all outputs regardless of mAP improvement.

Outputs:
  data/reports/yolo26/security_item_yolo26n_benchmark.json
  data/reports/yolo26/security_item_yolo26n_benchmark.md
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SecurityItemY26Evaluator")

CRITICAL_LIMITATION = "REAL-FIREARM-VIDEO VALIDATION NOT AVAILABLE"
CONF_SWEEP = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]
IOU_THRESHOLD = 0.45
SMALL_THRESH_PX = 96           # px, same as v2.1 evaluation
WARMUP_RUNS = 50
BENCH_RUNS = 200

FP_CATEGORIES = ["tool", "phone", "fence_pattern", "shadow", "background_clutter", "other"]


def calc_iou(bA, bB):
    xA = max(bA[0], bB[0]); yA = max(bA[1], bB[1])
    xB = min(bA[2], bB[2]); yB = min(bA[3], bB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(1e-6, (bA[2] - bA[0]) * (bA[3] - bA[1]))
    areaB = max(1e-6, (bB[2] - bB[0]) * (bB[3] - bB[1]))
    return inter / (areaA + areaB - inter + 1e-6)


def evaluate_at_conf(model, images, labels_dir, conf, imgsz, device):
    stats = {"TP": 0, "FP": 0, "FN": 0, "GT": 0}
    hard = {
        "small": {"TP": 0, "FN": 0, "GT": 0},
        "occluded": {"TP": 0, "FN": 0, "GT": 0},
        "low_light": {"TP": 0, "FN": 0, "GT": 0},
    }

    for img_p in images:
        if img_p.stat().st_size == 0:
            continue
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]
        fn_stem = img_p.stem.lower()
        is_occluded = "occ" in fn_stem or "partial" in fn_stem
        is_low_light = "dark" in fn_stem or "night" in fn_stem or "lowlight" in fn_stem or "low_light" in str(img_p).lower()

        lbl_p = labels_dir / f"{img_p.stem}.txt"
        gt_boxes = []
        if lbl_p.exists():
            for line in lbl_p.read_text(encoding="utf-8").strip().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cls_id = int(parts[0])
                    if cls_id != 0:
                        continue  # Only firearm (class 0)
                    cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    x1 = (cx - bw / 2) * w; y1 = (cy - bh / 2) * h
                    x2 = (cx + bw / 2) * w; y2 = (cy + bh / 2) * h
                    is_small = (bw * w < SMALL_THRESH_PX) or (bh * h < SMALL_THRESH_PX)
                    gt_boxes.append({
                        "box": [x1, y1, x2, y2],
                        "small": is_small,
                        "occluded": is_occluded,
                        "low_light": is_low_light,
                    })
                    stats["GT"] += 1
                    if is_small: hard["small"]["GT"] += 1
                    if is_occluded: hard["occluded"]["GT"] += 1
                    if is_low_light: hard["low_light"]["GT"] += 1
                except ValueError:
                    continue

        res = model.predict(source=img, imgsz=imgsz, device=device, conf=conf, verbose=False)
        preds = []
        if res and res[0].boxes is not None and len(res[0].boxes):
            for i in range(len(res[0].boxes)):
                if int(res[0].boxes.cls[i]) == 0:
                    preds.append({"box": res[0].boxes.xyxy[i].cpu().tolist()})

        matched_gt = set()
        for pred in preds:
            best_iou, best_idx = 0.0, -1
            for gi, gt in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                iou = calc_iou(pred["box"], gt["box"])
                if iou > best_iou:
                    best_iou, best_idx = iou, gi
            if best_iou >= IOU_THRESHOLD and best_idx >= 0:
                stats["TP"] += 1
                matched_gt.add(best_idx)
                gt = gt_boxes[best_idx]
                for attr in ("small", "occluded", "low_light"):
                    if gt[attr]:
                        hard[attr]["TP"] += 1
            else:
                stats["FP"] += 1

        for gi, gt in enumerate(gt_boxes):
            if gi not in matched_gt:
                stats["FN"] += 1
                for attr in ("small", "occluded", "low_light"):
                    if gt[attr]:
                        hard[attr]["FN"] += 1

    tp, fp, fn = stats["TP"], stats["FP"], stats["FN"]
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    hard_metrics = {}
    for attr, s in hard.items():
        gt_c = s["GT"]
        hard_metrics[attr] = {
            "GT": gt_c,
            "TP": s["TP"],
            "FN": s["FN"],
            "recall": round(s["TP"] / gt_c, 4) if gt_c > 0 else None,
        }

    return {
        "TP": tp, "FP": fp, "FN": fn, "GT": stats["GT"],
        "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
        "hard_cases": hard_metrics,
    }


def run_latency_benchmark(model, imgsz, device) -> dict:
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(WARMUP_RUNS):
        model.predict(source=dummy, imgsz=imgsz, device=device, verbose=False)
    import time as _time
    times = []
    for _ in range(BENCH_RUNS):
        t0 = _time.perf_counter()
        model.predict(source=dummy, imgsz=imgsz, device=device, verbose=False)
        times.append((_time.perf_counter() - t0) * 1000.0)
    times = np.array(times)
    return {
        "p50_ms": round(float(np.percentile(times, 50)), 3),
        "p95_ms": round(float(np.percentile(times, 95)), 3),
        "p99_ms": round(float(np.percentile(times, 99)), 3),
        "fps": round(1000.0 / float(np.percentile(times, 50)), 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate IBVAP Security Item YOLO26n")
    parser.add_argument("--model", default="models/candidates/yolo26/security_item/best.pt")
    parser.add_argument("--images", default="data/normalized/security_item_v2/images/test")
    parser.add_argument("--labels", default="data/normalized/security_item_v2/labels/test")
    parser.add_argument("--dataset-yaml", default="data/normalized/security_item_v2_1/dataset.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--conf", type=float, default=0.35)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        logger.critical(f"Model not found: {model_path}. Run training first.")
        sys.exit(1)

    img_dir = Path(args.images)
    lbl_dir = Path(args.labels)
    if not img_dir.exists():
        logger.critical(f"Image dir not found: {img_dir}")
        sys.exit(1)

    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    logger.info(f"Benchmark: {img_dir} — {len(images)} images | ⚠ {CRITICAL_LIMITATION}")

    device = args.device if torch.cuda.is_available() else "cpu"
    model = YOLO(str(model_path))
    model.to(device)

    # Official mAP
    val_res = model.val(data=str(Path(args.dataset_yaml).resolve()), split="val",
                        imgsz=args.imgsz, device=device, conf=args.conf, iou=IOU_THRESHOLD, verbose=False)
    official_map50 = float(val_res.box.map50)
    official_map50_95 = float(val_res.box.map)
    logger.info(f"Official mAP50={official_map50:.4f} | mAP50-95={official_map50_95:.4f}")

    sweep_results = {}
    for c in CONF_SWEEP:
        r = evaluate_at_conf(model, images, lbl_dir, c, args.imgsz, device)
        sweep_results[str(c)] = r
        logger.info(f"  conf={c:.2f} | P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}")

    primary = evaluate_at_conf(model, images, lbl_dir, args.conf, args.imgsz, device)
    latency = run_latency_benchmark(model, args.imgsz, device)

    bl_path = Path("models/benchmarks/yolo11_baseline/security_item_baseline.json")
    baseline = {}
    if bl_path.exists():
        with open(bl_path, encoding="utf-8") as f:
            baseline = json.load(f)

    bl_map = baseline.get("class_metrics", {}).get("firearm", {}).get("map50")
    delta_map = round(official_map50 - (bl_map or 0), 4)

    report = {
        "experiment_id": "IBVAP-ITEM-Y26-001",
        "model": str(model_path),
        "benchmark": "IBVAP-GT-ITEM-v2.0",
        "benchmark_images": len(images),
        "imgsz": args.imgsz,
        "primary_conf": args.conf,
        "iou_threshold": IOU_THRESHOLD,
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "critical_limitation": CRITICAL_LIMITATION,
        "official_map50": round(official_map50, 4),
        "official_map50_95": round(official_map50_95, 4),
        "primary_eval": primary,
        "confidence_sweep": sweep_results,
        "latency": latency,
        "yolo11_baseline": {
            "map50": bl_map,
            "precision": baseline.get("class_metrics", {}).get("firearm", {}).get("precision"),
            "recall": baseline.get("class_metrics", {}).get("firearm", {}).get("recall"),
            "f1": baseline.get("class_metrics", {}).get("firearm", {}).get("f1"),
            "p50_ms": baseline.get("latency", {}).get("p50_ms"),
            "p95_ms": baseline.get("latency", {}).get("p95_ms"),
        },
        "delta": {
            "map50": delta_map,
            "precision": round(primary["precision"] - (baseline.get("class_metrics", {}).get("firearm", {}).get("precision") or 0), 4),
            "recall": round(primary["recall"] - (baseline.get("class_metrics", {}).get("firearm", {}).get("recall") or 0), 4),
        },
    }

    out_dir = Path("data/reports/yolo26")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "security_item_yolo26n_benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    winner = "YOLO26n" if delta_map > 0 else "YOLO11n"
    md = [
        "# IBVAP Security Item Benchmark — YOLO26n vs YOLO11n",
        f"> ⚠ **{CRITICAL_LIMITATION}**",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
        "",
        "| Metric | YOLO11n | YOLO26n | Delta | Winner |",
        "|--------|---------|---------|-------|--------|",
        f"| mAP50 | {bl_map} | {official_map50:.4f} | {delta_map:+.4f} | {winner} |",
        f"| Precision | {baseline.get('class_metrics', {}).get('firearm', {}).get('precision')} | {primary['precision']} | {report['delta']['precision']:+.4f} | |",
        f"| Recall | {baseline.get('class_metrics', {}).get('firearm', {}).get('recall')} | {primary['recall']} | {report['delta']['recall']:+.4f} | |",
        "",
        "## Hard-Case Recall (YOLO26n)",
        "| Scenario | GT | Recall |",
        "|----------|----|--------|",
    ]
    for attr, hm in primary.get("hard_cases", {}).items():
        md.append(f"| {attr} | {hm['GT']} | {hm['recall']} |")
    md.append("")
    md.append("## Latency")
    md.append(f"| P50 (ms) | {baseline.get('latency', {}).get('p50_ms')} | {latency['p50_ms']} |")
    md.append(f"| P95 (ms) | {baseline.get('latency', {}).get('p95_ms')} | {latency['p95_ms']} |")

    (out_dir / "security_item_yolo26n_benchmark.md").write_text("\n".join(md), encoding="utf-8")
    logger.info(f"Reports saved to {out_dir}")
    logger.info(f"⚠ LIMITATION PRESERVED: {CRITICAL_LIMITATION}")


if __name__ == "__main__":
    main()
