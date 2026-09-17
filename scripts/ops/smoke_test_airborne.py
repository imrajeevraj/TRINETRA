#!/usr/bin/env python3
"""
IBVAP — Track A: Airborne Candidate Production Smoke & Validation Suite
Executes Steps A1 through A9 for candidate ibvap_airborne_v2_exp002.
"""

import os
import sys
import time
import json
import hashlib
import logging
from pathlib import Path
import numpy as np
import cv2
import torch
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AirborneSmokeTest")

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.ops.master_ai_pipeline import compute_sha256, evaluate_on_airborne_benchmark

def main():
    logger.info("=== STEP A1: VERIFY CHECKPOINT EXISTS ===")
    candidate_path = ROOT_DIR / "data/training/runs/ibvap_airborne_v2_exp002/weights/best.pt"
    assert candidate_path.exists(), f"Candidate checkpoint missing: {candidate_path}"
    logger.info(f"Checkpoint verified: {candidate_path} ({candidate_path.stat().st_size} bytes)")

    logger.info("=== STEP A2: CALCULATE SHA-256 ===")
    actual_sha = compute_sha256(candidate_path)
    logger.info(f"Computed SHA-256: {actual_sha}")

    logger.info("=== STEP A3: COMPARE CHECKSUM AGAINST MODEL REGISTRY ===")
    registry_path = ROOT_DIR / "models/model_registry.yaml"
    import yaml
    with open(registry_path, "r", encoding="utf-8") as f:
        reg_data = yaml.safe_load(f)
    registered_entry = None
    for m in reg_data.get("models", []):
        if m.get("id") == "ibvap-airborne-v2-exp002":
            registered_entry = m
            break
    assert registered_entry is not None, "Model ibvap-airborne-v2-exp002 not found in registry!"
    reg_sha = registered_entry.get("sha256", "").upper()
    assert actual_sha.upper() == reg_sha, f"SHA mismatch! Actual: {actual_sha}, Registry: {reg_sha}"
    logger.info("SHA-256 matches model registry entry perfectly!")

    logger.info("=== STEP A4 & A5: LOAD CHECKPOINT & VERIFY ARCHITECTURE ===")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(candidate_path))
    model.to(device)

    # Inspect class names and order
    classes = model.names
    logger.info(f"Model classes: {classes}")
    assert classes[0] == "drone", f"Expected class 0: drone, got: {classes.get(0)}"
    assert classes[1] == "aircraft", f"Expected class 1: aircraft, got: {classes.get(1)}"
    logger.info("Class order verified: {0: 'drone', 1: 'aircraft'}")

    logger.info("=== STEP A6 & A7: RUN AIRBORNE SMOKE TEST ===")
    # Create or sample test frames covering all required scenarios
    smoke_dir = ROOT_DIR / "data/reports/kaggle_validation/airborne_smoke_test"
    smoke_dir.mkdir(parents=True, exist_ok=True)

    test_scenarios = [
        ("drone", ROOT_DIR / "data/normalized/airborne_v2/images/val"),
        ("aircraft", ROOT_DIR / "data/normalized/airborne_v2/images/val"),
        ("bird_negative", ROOT_DIR / "data/raw/airborne/birds_negatives"),
        ("cloud_negative", ROOT_DIR / "data/raw/airborne/clouds_negatives"),
    ]

    sample_images = []
    # Sample from each scenario
    for tag, folder in test_scenarios:
        if folder.exists():
            files = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
            if files:
                sample_images.append((tag, files[0]))

    # Synthetic empty sky and distant scene
    empty_sky = np.full((640, 640, 3), (235, 206, 135), dtype=np.uint8) # BGR light blue
    empty_sky_path = smoke_dir / "smoke_empty_sky.jpg"
    cv2.imwrite(str(empty_sky_path), empty_sky)
    sample_images.append(("empty_sky", empty_sky_path))

    smoke_results = {}
    for tag, img_path in sample_images:
        res = model.predict(source=str(img_path), imgsz=640, conf=0.25, device=device, verbose=False)[0]
        boxes = len(res.boxes)
        det_classes = [classes[int(c)] for c in res.boxes.cls.tolist()] if boxes > 0 else []
        smoke_results[tag] = {
            "image": img_path.name,
            "detections": boxes,
            "classes": det_classes
        }
        logger.info(f"Smoke Test [{tag}]: {img_path.name} -> {boxes} detections ({det_classes})")

    logger.info("=== STEP A8: FRESH FROZEN BENCHMARK EVALUATION ===")
    eval_metrics = evaluate_on_airborne_benchmark(str(candidate_path), imgsz=640)
    logger.info(f"Fresh Benchmark Evaluation: {json.dumps(eval_metrics, indent=2)}")

    logger.info("=== STEP A9: PERFORMANCE & HARDWARE BENCHMARK ===")
    latencies = []
    # 50 warmup forward passes
    dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    for _ in range(50):
        _ = model.predict(source=dummy_input, imgsz=640, conf=0.25, device=device, verbose=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        vram_allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        vram_reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
    else:
        vram_allocated_mb = 0.0
        vram_reserved_mb = 0.0

    # 150 timed runs
    t0_total = time.perf_counter()
    for _ in range(150):
        t0 = time.perf_counter()
        _ = model.predict(source=dummy_input, imgsz=640, conf=0.25, device=device, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)
    total_time = time.perf_counter() - t0_total

    fps = 150.0 / total_time
    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))

    hardware_report = {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "fps": fps,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "vram_allocated_mb": vram_allocated_mb,
        "vram_reserved_mb": vram_reserved_mb,
    }
    logger.info(f"Hardware Performance: {hardware_report}")

    summary = {
        "step_a1_checkpoint": str(candidate_path),
        "step_a2_sha256": actual_sha,
        "step_a3_registry_match": True,
        "step_a5_classes": classes,
        "step_a6_smoke_tests": smoke_results,
        "step_a8_eval_metrics": eval_metrics,
        "step_a9_hardware": hardware_report
    }

    out_file = smoke_dir / "airborne_smoke_validation.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"Smoke test summary successfully saved to {out_file}")

if __name__ == "__main__":
    main()
