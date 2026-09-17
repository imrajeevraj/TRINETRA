#!/usr/bin/env python3
"""
IBVAP — Hyperparameter Tuning & Evolution Engine
Performs hyperparameter search over learning rate, momentum, and augmentation space
using validation split. Strict prohibition against tuning on IBVAP-GT-v1.0.
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
import yaml
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("IBVAPTuner")

def tune_hyperparameters(config_path: str, iterations: int = 10, epochs_per_iter: int = 5):
    logger.info("=" * 70)
    logger.info("IBVAP HYPERPARAMETER EVOLUTION")
    logger.info("=" * 70)

    cfg_file = Path(config_path)
    if not cfg_file.exists():
        logger.error(f"Config not found: {cfg_file}")
        sys.exit(1)

    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_yaml = cfg.get("dataset", {}).get("config_path", "data/training/datasets/IBVAP-TRAIN-v1.0/data.yaml")
    base_model = cfg.get("model", {}).get("base_architecture", "yolo11n.pt")
    device = cfg.get("training", {}).get("device", "cuda:0")

    logger.info(f"Base Model: {base_model} | Tuning Dataset: {data_yaml}")
    logger.info("NOTE: Hyperparameter search evaluates against validation split only.")
    logger.info("FROZEN BENCHMARK IBVAP-GT-v1.0 IS NEVER ACCESSED DURING TUNING.")

    model = YOLO(base_model)
    tune_results = model.tune(
        data=data_yaml,
        epochs=epochs_per_iter,
        iterations=iterations,
        optimizer="AdamW",
        plots=False,
        save=False,
        val=True,
        device=device
    )

    logger.info("Hyperparameter evolution complete.")
    logger.info(f"Optimal parameters saved to runs/detect/tune/")
    return tune_results

def main():
    parser = argparse.ArgumentParser(description="IBVAP Hyperparameter Evolution")
    parser.add_argument("--config", type=str, default="configs/training.yaml", help="Path to training config")
    parser.add_argument("--iterations", type=int, default=5, help="Number of search iterations")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs per iteration")
    args = parser.parse_args()

    tune_hyperparameters(args.config, iterations=args.iterations, epochs_per_iter=args.epochs)

if __name__ == "__main__":
    main()
