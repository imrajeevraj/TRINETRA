#!/usr/bin/env python3
"""
IBVAP — Airborne Model YOLO26n Evaluation (IBVAP-AIR-Y26-001)
Evaluates against airborne val split. Exact mirror of evaluate_airborne_v1.py.

Outputs:
  data/reports/yolo26/airborne_yolo26n_benchmark.json
  data/reports/yolo26/airborne_yolo26n_benchmark.md
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
logger = logging.getLogger("AirborneY26Evaluator")

AIRBORNE_CLASSES = {0: "drone", 1: "aircraft"}
CONF_SWEEP = [0.10, 0.20, 0.30, 0.40, 0.50]
WARMUP_RUNS = 50
BENCH_RUNS = 200


def calc_iou(bA, bB):
    xA = max(bA[0], bB[0]); yA = max(bA[1], bB[1])
    xB = min(bA[2], bB[2]); yB = min(bA[3], bB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(1e-6, (bA[2] - bA[0]) * (bA[3] - bA[1]))
    areaB = max(1e-6, (bB[2] - bB[0]) * (bB[3] - bB[1]))
    return inter / (areaA + areaB - inter + 1e-6)


def evaluate_at_conf(model, images, labels_dir, conf, imgsz, device):
    stats = {c: {"TP": 0, "FP": 0, "FN": 0, "GT": 0} for c in AIRBORNE_CLASSES}
    fp_taxonomy = {c: {"bird": 0, "cloud": 0, "infrastructure": 0, "other": 0} for c in AIRBORNE_CLASSES}

    for img_p in images:
        if img_p.stat().st_size == 0:
            continue
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]

        lbl_p = labels_dir / f"{img_p.stem}.txt"
        gt_boxes = []
        if lbl_p.exists():
            for line in lbl_p.read_text(encoding="utf-8").strip().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    x1 = (cx - bw / 2) * w; y1 = (cy - bh / 2) * h
                    x2 = (cx + bw / 2) * w; y2 = (cy + bh / 2) * h
                    gt_boxes.append({"cls": cls_id, "box": [x1, y1, x2, y2]})
                    if cls_id in stats:
                        stats[cls_id]["GT"] += 1
                except ValueError:
                    continue

        res = model.predict(source=img, imgsz=imgsz, device=device, conf=conf, verbose=False)
        preds = []
        if res and res[0].boxes is not None and len(res[0].boxes):
            for i in range(len(res[0].boxes)):
                preds.append({
                    "cls": int(res[0].boxes.cls[i]),
                    "box": res[0].boxes.xyxy[i].cpu().tolist(),
                })

        matched_gt = set()
        for pred in preds:
            pcls = pred["cls"]
            if pcls not in stats:
                continue
            best_iou, best_idx = 0.0, -1
            for gi, gt in enumerate(gt_boxes):
                if gt["cls"] != pcls or gi in matched_gt:
                    continue
                iou = calc_iou(pred["box"], gt["box"])
                if iou > best_iou:
                    best_iou, best_idx = iou, gi
            if best_iou >= 0.50 and best_idx >= 0:
                stats[pcls]["TP"] += 1
                matched_gt.add(best_idx)
            else:
                stats[pcls]["FP"] += 1
                fp_taxonomy[pcls]["other"] += 1  # full taxonomy requires annotated FP labels

        for gi, gt in enumerate(gt_boxes):
            if gi not in matched_gt:
                cls_id = gt["cls"]
                if cls_id in stats:
                    stats[cls_id]["FN"] += 1

    cls_metrics = {}
    for cls_id, name in AIRBORNE_CLASSES.items():
        s = stats[cls_id]
        tp, fp, fn = s["TP"], s["FP"], s["FN"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        cls_metrics[name] = {"TP": tp, "FP": fp, "FN": fn, "GT": s["GT"],
                              "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}

    return {"per_class": cls_metrics, "fp_taxonomy": fp_taxonomy}


def run_latency_benchmark(model, imgsz, device) -> dict:
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(WARMUP_RUNS):
        model.predict(source=dummy, imgsz=imgsz, device=device, verbose=False)
    times = []
    import time as _time
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
        "std_ms": round(float(times.std()), 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate IBVAP Airborne YOLO26n")
    parser.add_argument("--model", default="models/candidates/yolo26/airborne/best.pt")
    parser.add_argument("--images", default="data/normalized/airborne_v1_1/images/val")
    parser.add_argument("--labels", default="data/normalized/airborne_v1_1/labels/val")
    parser.add_argument("--dataset-yaml", default="data/normalized/airborne_v1_1/dataset.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--conf", type=float, default=0.40)
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
    logger.info(f"Evaluating {len(images)} validation images | conf={args.conf}")

    device = args.device if torch.cuda.is_available() else "cpu"
    model = YOLO(str(model_path))
    model.to(device)

    # Official mAP via val()
    val_res = model.val(data=str(Path(args.dataset_yaml).resolve()), split="val",
                        imgsz=args.imgsz, device=device, conf=args.conf, iou=0.50, verbose=False)
    official_map50 = float(val_res.box.map50)
    official_map50_95 = float(val_res.box.map)
    logger.info(f"Official mAP50={official_map50:.4f} | mAP50-95={official_map50_95:.4f}")

    sweep_results = {}
    for c in CONF_SWEEP:
        r = evaluate_at_conf(model, images, lbl_dir, c, args.imgsz, device)
        sweep_results[str(c)] = r
        drone_f1 = r["per_class"].get("drone", {}).get("f1", 0)
        aircraft_f1 = r["per_class"].get("aircraft", {}).get("f1", 0)
        logger.info(f"  conf={c:.2f} | Drone F1={drone_f1:.3f} | Aircraft F1={aircraft_f1:.3f}")

    primary = evaluate_at_conf(model, images, lbl_dir, args.conf, args.imgsz, device)
    latency = run_latency_benchmark(model, args.imgsz, device)

    bl_path = Path("models/benchmarks/yolo11_baseline/airborne_baseline.json")
    baseline = {}
    if bl_path.exists():
        with open(bl_path, encoding="utf-8") as f:
            baseline = json.load(f)

    report = {
        "experiment_id": "IBVAP-AIR-Y26-001",
        "model": str(model_path),
        "benchmark": "airborne_v1_val",
        "benchmark_images": len(images),
        "imgsz": args.imgsz,
        "primary_conf": args.conf,
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "official_map50": round(official_map50, 4),
        "official_map50_95": round(official_map50_95, 4),
        "primary_eval": primary,
        "confidence_sweep": sweep_results,
        "latency": latency,
        "yolo11_baseline": {
            "map50": baseline.get("aggregate_metrics", {}).get("map50"),
            "p50_ms": baseline.get("latency", {}).get("p50_ms"),
            "p95_ms": baseline.get("latency", {}).get("p95_ms"),
        },
        "delta": {
            "map50": round(official_map50 - (baseline.get("aggregate_metrics", {}).get("map50") or 0), 4),
        },
    }

    out_dir = Path("data/reports/yolo26")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "airborne_yolo26n_benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    bl_map = baseline.get("aggregate_metrics", {}).get("map50", "N/A")
    winner = "YOLO26n" if report["delta"]["map50"] > 0 else "YOLO11n"
    md_lines = [
        "# IBVAP Airborne Model Benchmark — YOLO26n vs YOLO11n",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
        "",
        "| Metric | YOLO11n | YOLO26n | Delta | Winner |",
        "|--------|---------|---------|-------|--------|",
        f"| mAP50 | {bl_map} | {official_map50:.4f} | {report['delta']['map50']:+.4f} | {winner} |",
        "",
    ]
    for cls_name, cm in primary["per_class"].items():
        md_lines += [
            f"### {cls_name.capitalize()}",
            f"P={cm['precision']:.3f} | R={cm['recall']:.3f} | F1={cm['f1']:.3f}",
            "",
        ]
    (out_dir / "airborne_yolo26n_benchmark.md").write_text("\n".join(md_lines), encoding="utf-8")
    logger.info(f"Reports saved to {out_dir}")


if __name__ == "__main__":
    main()
