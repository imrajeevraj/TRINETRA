#!/usr/bin/env python3
"""
IBVAP — Model Training & Transfer Learning Engine
Trains YOLO11n on the normalized IBVAP surveillance dataset with GPU validation,
automatic logging, SHA-256 calculation, and artifact registration.
"""

import os
import sys
import argparse
import logging
import hashlib
import json
import time
from pathlib import Path
import torch
import yaml
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("IBVAPTrainer")

def calculate_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().upper()

def train_ibvap(config_path: str, epochs_override: int | None = None, device_override: str | None = None):
    logger.info("=" * 70)
    logger.info("IBVAP MODEL TRAINING PIPELINE")
    logger.info("=" * 70)

    # 1. Load configuration
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        logger.error(f"Configuration file not found: {cfg_file}")
        sys.exit(1)

    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2. Hardware & Device Verification (Step 19)
    requested_device = device_override or cfg.get("training", {}).get("device", "cuda:0")
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    else:
        device = requested_device if torch.cuda.is_available() else "cpu"

    logger.info(f"Target Compute Device: {device.upper()}")
    if device.startswith("cuda"):
        logger.info(f"GPU Hardware: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        logger.info(f"Total VRAM: {vram:.1f} MB")
        logger.info(f"PyTorch Version: {torch.__version__}")

    # 3. Validate Dataset (Step 21.1)
    dataset_cfg = cfg.get("dataset", {}).get("config_path", "data/training/datasets/IBVAP-TRAIN-v1.0/data.yaml")
    data_yaml_path = Path(dataset_cfg)
    if not data_yaml_path.exists():
        logger.error(f"Dataset YAML descriptor not found: {data_yaml_path}")
        logger.error("Please assemble the dataset first or verify data.yaml path.")
        sys.exit(1)

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data_info = yaml.safe_load(f)
    logger.info(f"Dataset Verified: {data_yaml_path}")
    logger.info(f"Dataset Classes: {data_info.get('names', {})}")

    # 4. Load Base Model (Step 17 & 21.3)
    base_model = cfg.get("model", {}).get("base_architecture", "yolo11n.pt")
    logger.info(f"Loading Base Architecture: {base_model}")
    model = YOLO(base_model)

    # 5. Determine Hyperparameters
    epochs = epochs_override if epochs_override is not None else cfg.get("training", {}).get("epochs", 10)
    batch_size = cfg.get("training", {}).get("batch_size", 16)
    imgsz = cfg.get("training", {}).get("image_size", 640)
    workers = cfg.get("training", {}).get("workers", 4)
    run_name = cfg.get("run_name", f"ibvap_train_{int(time.time())}")
    runs_dir = Path(cfg.get("output", {}).get("runs_dir", "data/training/runs"))
    runs_dir.mkdir(parents=True, exist_ok=True)

    aug = cfg.get("augmentation", {})
    opt = cfg.get("optimizer", {})

    logger.info("=" * 70)
    logger.info(f"STARTING TRAINING RUN: {run_name}")
    logger.info(f"Epochs: {epochs} | Batch: {batch_size} | Image Size: {imgsz} | Device: {device}")
    logger.info("=" * 70)

    start_time = time.time()
    results = model.train(
        data=str(data_yaml_path.resolve()),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        workers=workers,
        project=str(runs_dir.resolve()),
        name=run_name,
        seed=cfg.get("training", {}).get("seed", 42),
        optimizer=opt.get("name", "AdamW"),
        lr0=opt.get("learning_rate", 0.002),
        lrf=opt.get("lr_final", 0.0001),
        weight_decay=opt.get("weight_decay", 0.0005),
        hsv_h=aug.get("hsv_h", 0.015),
        hsv_s=aug.get("hsv_s", 0.4),
        hsv_v=aug.get("hsv_v", 0.4),
        translate=aug.get("translate", 0.1),
        scale=aug.get("scale", 0.3),
        fliplr=aug.get("fliplr", 0.5),
        mosaic=aug.get("mosaic", 0.5),
        verbose=True,
        save=True
    )
    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed:.2f} seconds ({elapsed/60:.1f} minutes).")

    # 6. Locate Best Weights and Checksum
    run_output_dir = runs_dir / run_name
    best_weights = run_output_dir / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = run_output_dir / "weights" / "last.pt"

    sha256_hash = "UNKNOWN"
    if best_weights.exists():
        sha256_hash = calculate_sha256(best_weights)
        logger.info(f"Saved Best Model Weights: {best_weights}")
        logger.info(f"Model Checksum (SHA-256): {sha256_hash}")

    # 7. Record Training Metadata
    summary = {
        "run_name": run_name,
        "base_model": base_model,
        "dataset_version": cfg.get("dataset", {}).get("dataset_version", "IBVAP-TRAIN-v1.0"),
        "dataset_yaml": str(data_yaml_path),
        "epochs": epochs,
        "batch_size": batch_size,
        "image_size": imgsz,
        "device": device,
        "elapsed_seconds": round(elapsed, 2),
        "best_weights_path": str(best_weights),
        "sha256": sha256_hash,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    
    summary_file = run_output_dir / "training_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Training summary saved: {summary_file}")

    return summary

def main():
    parser = argparse.ArgumentParser(description="IBVAP Model Training Pipeline")
    parser.add_argument("--config", type=str, default="configs/training.yaml", help="Path to training config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs count")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu or cuda:0)")
    args = parser.parse_args()

    train_ibvap(args.config, epochs_override=args.epochs, device_override=args.device)

if __name__ == "__main__":
    main()
