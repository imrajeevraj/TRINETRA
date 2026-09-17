#!/usr/bin/env python3
"""
IBVAP — Ground Model v2 Training Engine (Experiment 001)
Executes controlled transfer learning using YOLO11n on the normalized Ground Model v2
dataset. Automatically profiles compute hardware, logs metrics, calculates SHA-256,
and produces experiment_manifest.json.
"""

import os
import sys
import json
import time
import hashlib
import argparse
import logging
from pathlib import Path
import torch
import yaml
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GroundV2Trainer")

def calculate_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().upper()

def train_ground_v2(
    dataset_yaml: str = "data/normalized/ground_v2_exp002/dataset.yaml",
    base_model: str = "yolo11n.pt",
    epochs: int = 40,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = "cuda:0",
    run_name: str = "ibvap_ground_v2_exp002a",
    seed: int = 42,
    lr0: float = 0.002,
    lrf: float = 0.0001,
    cos_lr: bool = True,
    scale: float = 0.1,
    mosaic: float = 0.2,
    close_mosaic: int = 10,
    translate: float = 0.05
):
    logger.info("=" * 70)
    logger.info(f"IBVAP GROUND MODEL V2 — CONTROLLED TRAINING RUN: {run_name}")
    logger.info("=" * 70)

    yaml_p = Path(dataset_yaml)
    if not yaml_p.exists():
        logger.critical(f"Dataset YAML descriptor not found: {yaml_p}")
        sys.exit(1)

    with open(yaml_p, "r", encoding="utf-8") as f:
        ds_info = yaml.safe_load(f)

    # Hardware Verification
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    else:
        device = device if torch.cuda.is_available() else "cpu"

    logger.info(f"Target Compute Device: {device.upper()}")
    gpu_info = "CPU"
    if device.startswith("cuda"):
        gpu_info = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        logger.info(f"AI GPU Hardware: {gpu_info} ({vram_mb:.1f} MB VRAM)")
        logger.info("FP16 Tensor Cores Activated.")

    logger.info(f"Loading Base Architecture: {base_model}")
    model = YOLO(base_model)

    runs_dir = Path("data/training/runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_dir = runs_dir / run_name

    logger.info("=" * 70)
    logger.info(f"STARTING RUN: {run_name}")
    logger.info(f"Epochs: {epochs} | Batch: {batch_size} | Image Size: {imgsz} | Seed: {seed}")
    logger.info(f"Dataset: {yaml_p.resolve()} | Target Classes: {ds_info.get('names')}")
    logger.info("=" * 70)

    start_time = time.time()
    results = model.train(
        data=str(yaml_p.resolve()),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project=str(runs_dir.resolve()),
        name=run_name,
        seed=seed,
        optimizer="AdamW",
        lr0=lr0,
        lrf=lrf,
        cos_lr=cos_lr,
        weight_decay=0.0005,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
        translate=translate,
        scale=scale,
        fliplr=0.5,
        mosaic=mosaic,
        close_mosaic=close_mosaic,
        save=True,
        verbose=True
    )
    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed:.2f} seconds ({elapsed/60:.1f} minutes).")

    # Locate best weights
    best_weights = out_dir / "weights" / "best.pt"
    if not best_weights.exists():
        # Check if Ultralytics suffixed run directory
        candidate_dirs = sorted(runs_dir.glob(f"{run_name}*"), key=os.path.getmtime, reverse=True)
        for cd in candidate_dirs:
            cw = cd / "weights" / "best.pt"
            if cw.exists():
                best_weights = cw
                out_dir = cd
                break

    sha256_hash = calculate_sha256(best_weights) if best_weights.exists() else "UNKNOWN"
    logger.info(f"Best Checkpoint: {best_weights}")
    logger.info(f"SHA-256 Checksum: {sha256_hash}")

    # Experiment Manifest
    manifest = {
        "experiment_id": "ibvap-ground-v2-exp001",
        "run_name": run_name,
        "base_model": base_model,
        "dataset_yaml": str(yaml_p),
        "dataset_version": "ground_v2",
        "epochs": epochs,
        "batch_size": batch_size,
        "image_size": imgsz,
        "optimizer": "AdamW",
        "learning_rate": 0.002,
        "seed": seed,
        "device": device,
        "gpu_hardware": gpu_info,
        "elapsed_seconds": round(elapsed, 2),
        "weights_path": str(best_weights),
        "sha256": sha256_hash,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    manifest_path = out_dir / "experiment_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Experiment manifest saved: {manifest_path}")
    return manifest

def main():
    parser = argparse.ArgumentParser(description="Train IBVAP Ground Model v2")
    parser.add_argument("--dataset", type=str, default="data/normalized/ground_v2_exp002/dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument("--base-model", type=str, default="yolo11n.pt", help="Pretrained base model weights")
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs to train")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", type=str, default="cuda:0", help="Compute device (cuda:0 or cpu)")
    parser.add_argument("--run-name", type=str, default="ibvap_ground_v2_exp002a", help="Run identifier")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--scale", type=float, default=0.1, help="Scale augmentation factor")
    parser.add_argument("--mosaic", type=float, default=0.2, help="Mosaic augmentation probability")
    parser.add_argument("--close-mosaic", type=int, default=10, help="Epochs to disable mosaic before end")
    parser.add_argument("--translate", type=float, default=0.05, help="Translation augmentation factor")
    parser.add_argument("--cos-lr", action="store_true", default=True, help="Use cosine learning rate scheduler")
    args = parser.parse_args()

    train_ground_v2(
        dataset_yaml=args.dataset,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        run_name=args.run_name,
        seed=args.seed,
        cos_lr=args.cos_lr,
        scale=args.scale,
        mosaic=args.mosaic,
        close_mosaic=args.close_mosaic,
        translate=args.translate
    )

if __name__ == "__main__":
    main()
