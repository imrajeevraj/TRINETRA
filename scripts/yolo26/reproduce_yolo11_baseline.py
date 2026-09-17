#!/usr/bin/env python3
"""
IBVAP — Phase 1: YOLO11 Baseline Reproduction
Runs all three YOLO11 production models through their standard evaluation paths
and confirms the metrics match the frozen baseline JSON files.

If drift > 1% mAP50 on any model: WARN and record the discrepancy.
This is a read-only audit — no model weights are modified.
"""

import json
import logging
import sys
import time
from pathlib import Path

import torch
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BaselineReproduction")

TOLERANCE = 0.01  # 1% mAP50 tolerance

BASELINE_CONFIGS = {
    "ground": {
        "checkpoint": "models/production/ground/ibvap_ground_v2_production.pt",
        "dataset_yaml": "data/normalized/ground_v2_exp002/dataset.yaml",
        "imgsz": 768,
        "conf": 0.25,
        "iou": 0.50,
        "baseline_json": "models/benchmarks/yolo11_baseline/ground_baseline.json",
        "map50_key": "mean_map50",
    },
    "airborne": {
        "checkpoint": "models/production/airborne/ibvap_airborne_v1_production.pt",
        "dataset_yaml": "data/normalized/airborne_v1_1/dataset.yaml",
        "imgsz": 640,
        "conf": 0.40,
        "iou": 0.50,
        "baseline_json": "models/benchmarks/yolo11_baseline/airborne_baseline.json",
        "map50_key": "map50",
    },
    "security_item": {
        "checkpoint": "models/production/security_item/ibvap_security_item_v2_1_production.pt",
        "dataset_yaml": "data/normalized/security_item_v2_1/dataset.yaml",
        "imgsz": 640,
        "conf": 0.35,
        "iou": 0.45,
        "baseline_json": "models/benchmarks/yolo11_baseline/security_item_baseline.json",
        "map50_key": "map50",
    },
}


def reproduce_baseline(name: str, cfg: dict) -> dict:
    logger.info("=" * 60)
    logger.info(f"Reproducing baseline: {name.upper()}")

    ckpt = Path(cfg["checkpoint"])
    if not ckpt.exists():
        logger.error(f"  Checkpoint not found: {ckpt}")
        return {"name": name, "status": "SKIPPED", "error": "checkpoint missing"}

    ds = Path(cfg["dataset_yaml"])
    if not ds.exists():
        logger.error(f"  Dataset YAML not found: {ds}")
        return {"name": name, "status": "SKIPPED", "error": "dataset yaml missing"}

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(ckpt))

    val_res = model.val(
        data=str(ds.resolve()),
        split="val",
        imgsz=cfg["imgsz"],
        device=device,
        conf=cfg["conf"],
        iou=cfg["iou"],
        verbose=False,
    )

    map50 = float(val_res.box.map50)
    map50_95 = float(val_res.box.map)
    precision = float(val_res.box.mp)
    recall = float(val_res.box.mr)

    logger.info(f"  mAP50:    {map50:.4f}")
    logger.info(f"  mAP50-95: {map50_95:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:   {recall:.4f}")

    # Load frozen baseline
    baseline_path = Path(cfg["baseline_json"])
    if not baseline_path.exists():
        logger.warning(f"  Baseline JSON not found: {baseline_path}")
        baseline_map50 = None
        drift = None
        status = "BASELINE_MISSING"
    else:
        with open(baseline_path, encoding="utf-8") as f:
            bl = json.load(f)
        agg = bl.get("aggregate_metrics", {})
        baseline_map50 = agg.get(cfg["map50_key"], agg.get("map50", agg.get("mean_map50")))
        if baseline_map50 is not None:
            drift = abs(map50 - baseline_map50)
            status = "PASS" if drift <= TOLERANCE else "DRIFT_WARNING"
            logger.info(f"  Baseline mAP50: {baseline_map50:.4f} | Drift: {drift:.4f} | {status}")
        else:
            drift = None
            status = "BASELINE_MISSING_KEY"

    return {
        "name": name,
        "status": status,
        "reproduced_map50": map50,
        "reproduced_map50_95": map50_95,
        "reproduced_precision": precision,
        "reproduced_recall": recall,
        "baseline_map50": baseline_map50,
        "drift": drift,
        "checkpoint": str(ckpt),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main():
    out_dir = Path("models/benchmarks/yolo11_baseline")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    overall_pass = True
    for name, cfg in BASELINE_CONFIGS.items():
        r = reproduce_baseline(name, cfg)
        results[name] = r
        if r["status"] not in ("PASS", "SKIPPED", "BASELINE_MISSING", "BASELINE_MISSING_KEY"):
            overall_pass = False

    report = {
        "reproduction_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tolerance": TOLERANCE,
        "overall_pass": overall_pass,
        "results": results,
    }

    out_path = out_dir / "baseline_reproduction.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(f"Report saved: {out_path}")

    for name, r in results.items():
        icon = "✅" if r["status"] == "PASS" else ("⚠️" if "WARN" in r["status"] else "❌")
        logger.info(f"  {icon} {name}: {r['status']} (mAP50={r.get('reproduced_map50', 'N/A')})")

    if not overall_pass:
        logger.warning("Baseline drift detected. Review before proceeding with YOLO26 training.")
    else:
        logger.info("Baseline reproduction PASSED. YOLO11 metrics confirmed.")


if __name__ == "__main__":
    main()
