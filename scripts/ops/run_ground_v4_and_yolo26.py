#!/usr/bin/env python3
"""
IBVAP — Track B & C: Ground v4 Experiments and YOLO26 Fair Evaluation Engine
Executes:
1. Ground v4 Exp001 (640px)
2. Ground v4 Exp002 (768px)
3. Ground v4 Exp003 (768px + optimized small object)
4. Frozen benchmark evaluations for Ground v4 vs Production vs Ground v3
5. YOLO26 Ground and Airborne empirical benchmark evaluations
"""

import sys
import os
import json
import time
import logging
from pathlib import Path
import numpy as np
import torch
import cv2

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.ops.master_ai_pipeline import (
    run_training_experiment,
    evaluate_on_frozen_ground_benchmark,
    evaluate_on_airborne_benchmark,
    generate_visual_error_overlays,
    compute_sha256,
    PRODUCTION_GROUND,
    PRODUCTION_AIRBORNE
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GroundV4AndYOLO26Runner")

def main():
    ground_yaml = ROOT_DIR / "data/normalized/ground_v4/ground_v4.yaml"
    reports_dir = ROOT_DIR / "data/reports/kaggle_validation"
    error_dir = ROOT_DIR / "data/reports/model_error_analysis"
    reports_dir.mkdir(parents=True, exist_ok=True)
    error_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Track B: Ground v4 Experiments
    # -------------------------------------------------------------
    logger.info("=== RUNNING GROUND V4 EXP 001 (640px, 12 epochs) ===")
    g4_exp1 = run_training_experiment(
        "ibvap_ground_v4_exp001",
        ground_yaml,
        "yolo11n.pt",
        epochs=12,
        batch_size=16,
        imgsz=640,
        device="cuda:0",
        scale=0.1,
        mosaic=0.2
    )
    logger.info(f"G4 Exp 001 Complete: {g4_exp1}")

    logger.info("=== RUNNING GROUND V4 EXP 002 (768px, 15 epochs) ===")
    g4_exp2 = run_training_experiment(
        "ibvap_ground_v4_exp002",
        ground_yaml,
        "yolo11n.pt",
        epochs=15,
        batch_size=16,
        imgsz=768,
        device="cuda:0",
        scale=0.1,
        mosaic=0.2
    )
    logger.info(f"G4 Exp 002 Complete: {g4_exp2}")

    logger.info("=== RUNNING GROUND V4 EXP 003 (768px, 15 epochs, small-object tuned) ===")
    g4_exp3 = run_training_experiment(
        "ibvap_ground_v4_exp003",
        ground_yaml,
        "yolo11n.pt",
        epochs=15,
        batch_size=16,
        imgsz=768,
        device="cuda:0",
        scale=0.15,
        mosaic=0.25,
        close_mosaic=5
    )
    logger.info(f"G4 Exp 003 Complete: {g4_exp3}")

    # Weights paths
    g4_1_weights = ROOT_DIR / "data/training/runs/ibvap_ground_v4_exp001/weights/best.pt"
    g4_2_weights = ROOT_DIR / "data/training/runs/ibvap_ground_v4_exp002/weights/best.pt"
    g4_3_weights = ROOT_DIR / "data/training/runs/ibvap_ground_v4_exp003/weights/best.pt"

    # -------------------------------------------------------------
    # 2. Frozen Benchmark Evaluations for Ground v4
    # -------------------------------------------------------------
    logger.info("=== EVALUATING GROUND V4 CANDIDATES ON FROZEN BENCHMARK ===")
    eval_g4_1 = evaluate_on_frozen_ground_benchmark(str(g4_1_weights), imgsz=640)
    eval_g4_2 = evaluate_on_frozen_ground_benchmark(str(g4_2_weights), imgsz=768)
    eval_g4_3 = evaluate_on_frozen_ground_benchmark(str(g4_3_weights), imgsz=768)

    # -------------------------------------------------------------
    # 3. Track C: YOLO26 Empirical Evaluation
    # -------------------------------------------------------------
    logger.info("=== EVALUATING YOLO26 CANDIDATES ON FROZEN BENCHMARKS ===")
    y26_ground_path = ROOT_DIR / "models/candidates/yolo26/ground/best.pt"
    y26_air_path = ROOT_DIR / "models/candidates/yolo26/airborne/best.pt"

    eval_y26_ground = evaluate_on_frozen_ground_benchmark(str(y26_ground_path), imgsz=768)
    eval_y26_air = evaluate_on_airborne_benchmark(str(y26_air_path), imgsz=640)

    # Generate visual error overlays for Ground v4 and YOLO26
    generate_visual_error_overlays(str(g4_3_weights), error_dir, imgsz=768)

    # Compile comprehensive results dictionary
    results = {
        "ground_v4_candidates": {
            "exp001": {
                "metrics": eval_g4_1,
                "weights": str(g4_1_weights),
                "sha256": compute_sha256(g4_1_weights)
            },
            "exp002": {
                "metrics": eval_g4_2,
                "weights": str(g4_2_weights),
                "sha256": compute_sha256(g4_2_weights)
            },
            "exp003": {
                "metrics": eval_g4_3,
                "weights": str(g4_3_weights),
                "sha256": compute_sha256(g4_3_weights)
            }
        },
        "yolo26_evaluation": {
            "ground": {
                "metrics": eval_y26_ground,
                "weights": str(y26_ground_path),
                "sha256": compute_sha256(y26_ground_path)
            },
            "airborne": {
                "metrics": eval_y26_air,
                "weights": str(y26_air_path),
                "sha256": compute_sha256(y26_air_path)
            }
        }
    }

    out_file = reports_dir / "ground_v4_and_yolo26_results.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info(f"Track B & C execution complete! Results saved to {out_file}")

if __name__ == "__main__":
    main()
