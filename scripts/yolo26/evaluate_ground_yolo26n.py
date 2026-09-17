#!/usr/bin/env python3
"""
IBVAP — Ground Model YOLO26n Evaluation (IBVAP-GROUND-Y26-001)
Evaluates against frozen benchmark IBVAP-GT-v1.0 (333 images).
Exact mirror of evaluate_ground_v2.py — same benchmark, same method.

Outputs:
  data/reports/yolo26/ground_yolo26n_benchmark.json
  data/reports/yolo26/ground_yolo26n_benchmark.md
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GroundY26Evaluator")

GROUND_CLASSES = {0: "person", 1: "vehicle"}
CONF_SWEEP = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
SMALL_THRESH_PX = 32          # objects with w or h < 32px treated as small
OCCLUDED_LABEL_SUFFIX = "_occ"
LOW_LIGHT_SUBDIR = "low_light"
DEFAULT_IMGSZ = 768            # same as YOLO11 Ground v2.0
DEFAULT_BENCHMARK_IMAGES = "benchmark/images/test"
DEFAULT_BENCHMARK_LABELS = "benchmark/labels/test"
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
    stats = {
        cls_id: {"TP": 0, "FP": 0, "FN": 0, "GT": 0}
        for cls_id in GROUND_CLASSES
    }
    hard_stats = {
        cls_id: {"small": {"TP": 0, "FN": 0, "GT": 0},
                 "occluded": {"TP": 0, "FN": 0, "GT": 0},
                 "low_light": {"TP": 0, "FN": 0, "GT": 0}}
        for cls_id in GROUND_CLASSES
    }

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
                    gt_boxes.append({
                        "cls": cls_id, "box": [x1, y1, x2, y2],
                        "small": (bw * w < SMALL_THRESH_PX or bh * h < SMALL_THRESH_PX),
                        "occluded": "_occ" in img_p.stem,
                        "low_light": LOW_LIGHT_SUBDIR in str(img_p),
                    })
                    if cls_id in stats:
                        stats[cls_id]["GT"] += 1
                except ValueError:
                    continue

        res = model.predict(source=img, imgsz=imgsz, device=device, conf=conf, verbose=False)
        preds = []
        if res and res[0].boxes is not None and len(res[0].boxes):
            boxes = res[0].boxes
            for i in range(len(boxes)):
                preds.append({
                    "cls": int(boxes.cls[i]),
                    "box": boxes.xyxy[i].cpu().tolist(),
                    "conf": float(boxes.conf[i]),
                })

        # Match predictions to GT
        matched_gt = set()
        for pred in preds:
            pcls = pred["cls"]
            if pcls not in stats:
                continue
            best_iou = 0.0
            best_gt_idx = -1
            for gi, gt in enumerate(gt_boxes):
                if gt["cls"] != pcls or gi in matched_gt:
                    continue
                iou = calc_iou(pred["box"], gt["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gi
            if best_iou >= 0.50 and best_gt_idx >= 0:
                stats[pcls]["TP"] += 1
                matched_gt.add(best_gt_idx)
                gt = gt_boxes[best_gt_idx]
                for attr in ("small", "occluded", "low_light"):
                    if gt[attr]:
                        hard_stats[pcls][attr]["TP"] += 1
            else:
                stats[pcls]["FP"] += 1

        for gi, gt in enumerate(gt_boxes):
            if gi not in matched_gt:
                pcls = gt["cls"]
                if pcls in stats:
                    stats[pcls]["FN"] += 1
                    for attr in ("small", "occluded", "low_light"):
                        if gt[attr]:
                            hard_stats[pcls][attr]["FN"] += 1
                            hard_stats[pcls][attr]["GT"] += 1

    # Compute metrics
    cls_metrics = {}
    for cls_id, name in GROUND_CLASSES.items():
        s = stats[cls_id]
        tp, fp, fn = s["TP"], s["FP"], s["FN"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        cls_metrics[name] = {"TP": tp, "FP": fp, "FN": fn, "GT": s["GT"],
                              "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}

    return {"per_class": cls_metrics, "hard_cases": hard_stats}


def run_latency_benchmark(model, imgsz, device) -> dict:
    logger.info(f"Latency benchmark: {WARMUP_RUNS} warmup + {BENCH_RUNS} steady-state runs")
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
        "mean_ms": round(float(times.mean()), 3),
        "p50_ms": round(float(np.percentile(times, 50)), 3),
        "p95_ms": round(float(np.percentile(times, 95)), 3),
        "p99_ms": round(float(np.percentile(times, 99)), 3),
        "fps": round(1000.0 / float(np.percentile(times, 50)), 1),
        "std_ms": round(float(times.std()), 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate IBVAP Ground YOLO26n")
    parser.add_argument("--model", default="models/candidates/yolo26/ground/best.pt")
    parser.add_argument("--images", default=DEFAULT_BENCHMARK_IMAGES)
    parser.add_argument("--labels", default=DEFAULT_BENCHMARK_LABELS)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--conf", type=float, default=0.25, help="Primary conf for detailed eval")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        logger.critical(f"Model not found: {model_path}. Run training first.")
        sys.exit(1)

    img_dir = Path(args.images)
    lbl_dir = Path(args.labels)
    if not img_dir.exists():
        logger.critical(f"Benchmark image dir not found: {img_dir}")
        sys.exit(1)

    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    logger.info(f"Benchmark: {img_dir} — {len(images)} images")

    device = args.device if torch.cuda.is_available() else "cpu"
    model = YOLO(str(model_path))
    model.to(device)

    # ── Official mAP via ultralytics val() ─────────────────────────────────
    from pathlib import Path as _P
    # Use the same exact validation split that YOLO11 baseline was run against
    dataset_yaml = _P("data/normalized/ground_v2_exp002/dataset.yaml")
    
    print(f"[{datetime.now().isoformat()}] [INFO] Starting validation...")
    val_res = model.val(data=str(dataset_yaml), split="val", imgsz=args.imgsz,
                        device=device, conf=args.conf, iou=0.50, verbose=False)
    official_map50 = float(val_res.box.map50)
    official_map50_95 = float(val_res.box.map)
    logger.info(f"Official mAP50: {official_map50:.4f} | mAP50-95: {official_map50_95:.4f}")

    # ── Confidence sweep ────────────────────────────────────────────────────
    logger.info("Confidence threshold sweep...")
    sweep_results = {}
    for c in CONF_SWEEP:
        r = evaluate_at_conf(model, images, lbl_dir, c, args.imgsz, device)
        sweep_results[str(c)] = r
        person_f1 = r["per_class"].get("person", {}).get("f1", 0)
        vehicle_f1 = r["per_class"].get("vehicle", {}).get("f1", 0)
        logger.info(f"  conf={c:.2f} | Person F1={person_f1:.3f} | Vehicle F1={vehicle_f1:.3f}")

    # ── Primary conf detailed eval ──────────────────────────────────────────
    primary = evaluate_at_conf(model, images, lbl_dir, args.conf, args.imgsz, device)
    latency = run_latency_benchmark(model, args.imgsz, device)

    # ── Baseline comparison ─────────────────────────────────────────────────
    bl_path = Path("models/benchmarks/yolo11_baseline/ground_baseline.json")
    baseline = {}
    if bl_path.exists():
        with open(bl_path, encoding="utf-8") as f:
            baseline = json.load(f)

    report = {
        "experiment_id": "IBVAP-GROUND-Y26-001",
        "model": str(model_path),
        "benchmark": "IBVAP-GT-v1.0",
        "benchmark_images": len(images),
        "imgsz": args.imgsz,
        "primary_conf": args.conf,
        "iou_threshold": 0.50,
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "official_map50": round(official_map50, 4),
        "official_map50_95": round(official_map50_95, 4),
        "primary_eval": primary,
        "confidence_sweep": sweep_results,
        "latency": latency,
        "yolo11_baseline": {
            "map50": baseline.get("aggregate_metrics", {}).get("mean_map50"),
            "p50_ms": baseline.get("latency", {}).get("p50_ms"),
            "p95_ms": baseline.get("latency", {}).get("p95_ms"),
            "fps": baseline.get("latency", {}).get("fps"),
        },
        "delta": {
            "map50": round(official_map50 - (baseline.get("aggregate_metrics", {}).get("mean_map50") or 0), 4),
            "p50_ms": round(latency["p50_ms"] - (baseline.get("latency", {}).get("p50_ms") or 0), 3),
        },
    }

    out_dir = Path("data/reports/yolo26")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "ground_yolo26n_benchmark.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(f"JSON report: {json_path}")

    # ── Markdown summary ────────────────────────────────────────────────────
    bl_map = baseline.get("aggregate_metrics", {}).get("mean_map50", "N/A")
    delta_map = report["delta"]["map50"]
    winner = "YOLO26n" if delta_map > 0 else ("TIE" if delta_map == 0 else "YOLO11n")
    md = [
        "# IBVAP Ground Model Benchmark — YOLO26n vs YOLO11n",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
        f"**Benchmark:** IBVAP-GT-v1.0 ({len(images)} images) | **Conf:** {args.conf} | **IoU:** 0.50",
        "",
        "## Accuracy",
        "| Metric | YOLO11n | YOLO26n | Delta | Winner |",
        "|--------|---------|---------|-------|--------|",
        f"| mAP50 | {bl_map} | {official_map50:.4f} | {delta_map:+.4f} | {winner} |",
        "",
        "## Per-Class (YOLO26n)",
    ]
    for cls_name, cm in primary["per_class"].items():
        md += [
            f"### {cls_name.capitalize()}",
            f"| P | R | F1 |",
            f"|---|---|-----|",
            f"| {cm['precision']:.3f} | {cm['recall']:.3f} | {cm['f1']:.3f} |",
            "",
        ]
    md += [
        "## Latency",
        "| Metric | YOLO11n | YOLO26n | Delta |",
        "|--------|---------|---------|-------|",
        f"| P50 (ms) | {baseline.get('latency', {}).get('p50_ms', 'N/A')} | {latency['p50_ms']} | {report['delta']['p50_ms']:+.3f} |",
        f"| P95 (ms) | {baseline.get('latency', {}).get('p95_ms', 'N/A')} | {latency['p95_ms']} | N/A |",
        f"| FPS | {baseline.get('latency', {}).get('fps', 'N/A')} | {latency['fps']} | N/A |",
        "",
        f"## Promotion Gate: {'PENDING — review full scorecard' }",
    ]
    md_path = out_dir / "ground_yolo26n_benchmark.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    logger.info(f"Markdown report: {md_path}")
    logger.info(f"mAP50 YOLO26n={official_map50:.4f} | YOLO11n={bl_map} | Delta={delta_map:+.4f} | {winner}")


if __name__ == "__main__":
    main()
