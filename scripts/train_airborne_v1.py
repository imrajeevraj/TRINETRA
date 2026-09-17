#!/usr/bin/env python3
"""
IBVAP — Airborne Model v1 Training Engine (Experiment 001)
Executes controlled baseline training for aerial threat detection (Drone & Aircraft).
Logs metrics to data/training/runs/ibvap_airborne_v1/ and records experiment_manifest.json.
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
logger = logging.getLogger("AirborneTrainer")

def calculate_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().upper()

def train_airborne_v1(
    dataset_yaml: str = "data/normalized/airborne_v1/dataset.yaml",
    base_model: str = "yolo11n.pt",
    epochs: int = 30,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = "cuda:0",
    run_name: str = "ibvap_airborne_v1",
    seed: int = 42,
    close_mosaic: int = 8,
    workers: int = 2,
    freeze: int = 0,
    lr0: float = 0.002
):
    logger.info("=" * 70)
    logger.info("IBVAP AIRBORNE MODEL V1 — CONTROLLED BASELINE TRAINING")
    logger.info("=" * 70)

    yaml_p = Path(dataset_yaml)
    if not yaml_p.exists():
        logger.critical(f"Airborne dataset YAML descriptor not found: {yaml_p}")
        sys.exit(1)

    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable. Falling back to CPU.")
        device = "cpu"
    else:
        device = device if torch.cuda.is_available() else "cpu"

    gpu_info = "CPU"
    if device.startswith("cuda"):
        gpu_info = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        logger.info(f"AI GPU Hardware: {gpu_info} ({vram_mb:.1f} MB VRAM) - FP16 Tensor Cores Activated")

    model = YOLO(base_model)
    runs_dir = Path("data/training/runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_dir = runs_dir / run_name

    logger.info("=" * 70)
    logger.info(f"STARTING AIRBORNE RUN: {run_name}")
    logger.info(f"Epochs: {epochs} | Batch: {batch_size} | Image Size: {imgsz} | Seed: {seed}")
    logger.info(f"Target Classes: 0 = drone, 1 = aircraft")
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
        lrf=lr0 * 0.05,
        cos_lr=True,
        scale=0.1,         # Preserve distant aerial targets
        mosaic=0.2,        # Mild mosaic to preserve sky context
        close_mosaic=close_mosaic,
        freeze=freeze if freeze > 0 else None,
        workers=workers,
        save=True,
        verbose=True
    )
    elapsed = time.time() - start_time
    logger.info(f"Airborne training completed in {elapsed:.2f} seconds ({elapsed/60:.1f} minutes).")

    best_weights = out_dir / "weights" / "best.pt"
    if not best_weights.exists():
        candidate_dirs = sorted(runs_dir.glob(f"{run_name}*"), key=os.path.getmtime, reverse=True)
        for cd in candidate_dirs:
            cw = cd / "weights" / "best.pt"
            if cw.exists():
                best_weights = cw
                out_dir = cd
                break

    sha256_hash = calculate_sha256(best_weights) if best_weights.exists() else "UNKNOWN"
    logger.info(f"Best Airborne Checkpoint: {best_weights}")
    logger.info(f"SHA-256 Checksum: {sha256_hash}")

    manifest = {
        "model_id": run_name.replace("_", "-"),
        "run_name": run_name,
        "base_model": base_model,
        "classes": {0: "drone", 1: "aircraft"},
        "epochs": epochs,
        "batch_size": batch_size,
        "image_size": imgsz,
        "device": device,
        "gpu_hardware": gpu_info,
        "elapsed_seconds": round(elapsed, 2),
        "weights_path": str(best_weights),
        "sha256": sha256_hash,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    with open(out_dir / "experiment_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Airborne experiment manifest written.")
    return manifest

def main():
    parser = argparse.ArgumentParser(description="Train IBVAP Airborne Model v1")
    parser.add_argument("--dataset", type=str, default="data/normalized/airborne_v1/dataset.yaml", help="Dataset YAML path")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", type=str, default="cuda:0", help="Compute device")
    parser.add_argument("--run-name", type=str, default="ibvap_airborne_v1", help="Run identifier")
    parser.add_argument("--base-model", type=str, default="yolo11n.pt", help="Pretrained weights or checkpoint")
    parser.add_argument("--close-mosaic", type=int, default=8, help="Close mosaic epochs")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--freeze", type=int, default=0, help="Number of backbone layers to freeze")
    parser.add_argument("--lr0", type=float, default=0.002, help="Initial learning rate")
    args = parser.parse_args()

    train_airborne_v1(
        dataset_yaml=args.dataset,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        run_name=args.run_name,
        close_mosaic=args.close_mosaic,
        workers=args.workers,
        freeze=args.freeze,
        lr0=args.lr0
    )

if __name__ == "__main__":
    main()
