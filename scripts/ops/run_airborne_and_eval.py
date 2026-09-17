#!/usr/bin/env python3
"""
IBVAP — Airborne Experiments & Unified Benchmark Evaluator
Runs Airborne v2 experiments and evaluates both Ground and Airborne candidates
against production baselines on the frozen benchmark.
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import time
from scripts.ops.master_ai_pipeline import (
    run_training_experiment,
    evaluate_on_frozen_ground_benchmark,
    evaluate_on_airborne_benchmark,
    generate_visual_error_overlays,
    compute_sha256,
    PRODUCTION_GROUND,
    PRODUCTION_AIRBORNE
)

ROOT_DIR = Path(__file__).resolve().parents[2]

def main():
    airborne_yaml = ROOT_DIR / "data/normalized/airborne_v2/airborne_v2.yaml"
    reports_base = ROOT_DIR / "data/reports/kaggle_validation"
    error_dir = ROOT_DIR / "data/reports/model_error_analysis"
    reports_base.mkdir(parents=True, exist_ok=True)
    error_dir.mkdir(parents=True, exist_ok=True)

    print("=== RUNNING AIRBORNE EXP 001 ===")
    a1 = run_training_experiment(
        "ibvap_airborne_v2_exp001",
        airborne_yaml,
        "yolo11n.pt",
        epochs=12,
        batch_size=16,
        imgsz=640,
        device="cuda:0",
        scale=0.1,
        mosaic=0.2
    )
    print("A1 DONE:", a1)

    print("=== RUNNING AIRBORNE EXP 002 ===")
    a2 = run_training_experiment(
        "ibvap_airborne_v2_exp002",
        airborne_yaml,
        "yolo11n.pt",
        epochs=15,
        batch_size=16,
        imgsz=640,
        device="cuda:0",
        scale=0.15,
        mosaic=0.25,
        close_mosaic=5
    )
    print("A2 DONE:", a2)

    # Weights paths
    g1_weights = ROOT_DIR / "data/training/runs/ibvap_ground_v3_exp001/weights/best.pt"
    g2_weights = ROOT_DIR / "data/training/runs/ibvap_ground_v3_exp002/weights/best.pt"
    a1_weights = ROOT_DIR / "data/training/runs/ibvap_airborne_v2_exp001/weights/best.pt"
    a2_weights = ROOT_DIR / "data/training/runs/ibvap_airborne_v2_exp002/weights/best.pt"

    print("=== EVALUATING ON FROZEN BENCHMARKS ===")
    prod_g = evaluate_on_frozen_ground_benchmark(PRODUCTION_GROUND["golden_path"], imgsz=768)
    cand_g1 = evaluate_on_frozen_ground_benchmark(str(g1_weights), imgsz=640)
    cand_g2 = evaluate_on_frozen_ground_benchmark(str(g2_weights), imgsz=768)

    prod_a = evaluate_on_airborne_benchmark(PRODUCTION_AIRBORNE["golden_path"], imgsz=640)
    cand_a1 = evaluate_on_airborne_benchmark(str(a1_weights), imgsz=640)
    cand_a2 = evaluate_on_airborne_benchmark(str(a2_weights), imgsz=640)

    # Visual errors for Ground
    generate_visual_error_overlays(str(g2_weights), error_dir, imgsz=768)

    results = {
        "production": {
            "ground": prod_g,
            "airborne": prod_a
        },
        "ground_candidates": {
            "exp001": cand_g1,
            "exp002": cand_g2,
            "exp001_sha": compute_sha256(g1_weights),
            "exp002_sha": compute_sha256(g2_weights)
        },
        "airborne_candidates": {
            "exp001": cand_a1,
            "exp002": cand_a2,
            "exp001_sha": compute_sha256(a1_weights),
            "exp002_sha": compute_sha256(a2_weights)
        }
    }

    out_json = reports_base / "master_benchmark_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Master evaluation complete! Saved to {out_json}")

if __name__ == "__main__":
    main()
