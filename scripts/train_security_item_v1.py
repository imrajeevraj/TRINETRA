#!/usr/bin/env python3
"""
IBVAP — Security Item Model v1 Training Engine (Experiment 001)
Architecture: YOLO11n
Target Class: 0: firearm
Dataset: data/normalized/security_item_v1/dataset.yaml
Saves weights, training manifests, and SHA-256 checksums to:
data/training/runs/ibvap_security_item_v1_exp001/
"""

import os
import sys
import time
import hashlib
import logging
import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SecurityItemTrain")


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()


def train_security_item_v1(
    base_model: str = "yolo11n.pt",
    dataset_yaml: str = "data/normalized/security_item_v1/dataset.yaml",
    epochs: int = 5,
    batch_size: int = 16,
    imgsz: int = 640,
    lr0: float = 0.001,
    freeze: int = 10,
    device: str = "cpu",
    run_name: str = "ibvap_security_item_v1_exp001"
):
    dataset_path = Path(dataset_yaml)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset config missing: {dataset_path}")

    project_dir = Path("data/training/runs")
    project_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = project_dir / run_name

    logger.info("=" * 70)
    logger.info("IBVAP SECURITY ITEM MODEL V1 — TRAINING EXPERIMENT 001")
    logger.info("=" * 70)
    logger.info(f"Base Model:       {base_model}")
    logger.info(f"Dataset:          {dataset_yaml}")
    logger.info(f"Epochs:           {epochs}")
    logger.info(f"Batch Size:       {batch_size}")
    logger.info(f"Image Size:       {imgsz}")
    logger.info(f"Initial LR:       {lr0}")
    logger.info(f"Frozen Layers:    {freeze}")
    logger.info(f"Compute Device:   {device}")
    logger.info(f"Output Directory: {exp_dir}")
    logger.info("=" * 70)

    model = YOLO(base_model)
    t0 = time.perf_counter()

    train_results = model.train(
        data=str(dataset_path.resolve()),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project=str(project_dir),
        name=run_name,
        freeze=freeze,
        lr0=lr0,
        lrf=0.01,
        optimizer="AdamW",
        weight_decay=0.0005,
        warmup_epochs=1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        shear=2.0,
        perspective=0.0001,
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.05,
        save=True,
        save_period=-1,
        val=True,
        verbose=True,
        seed=42,
        exist_ok=True
    )

    elapsed = time.perf_counter() - t0
    logger.info(f"Training completed in {elapsed:.2f} seconds ({elapsed/60:.1f} minutes).")

    weights_dir = exp_dir / "weights"
    best_pt = weights_dir / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {best_pt}")

    sha256 = compute_sha256(best_pt)
    logger.info(f"Best Security Item Checkpoint: {best_pt}")
    logger.info(f"SHA-256 Checksum: {sha256}")

    manifest = {
        "experiment_id": run_name,
        "model_name": "IBVAP Security Item Detector v1.0",
        "domain": "SECURITY_ITEM",
        "class_mapping": {0: "firearm"},
        "base_model": base_model,
        "dataset_version": "security_item_v1",
        "epochs": epochs,
        "batch_size": batch_size,
        "image_size": imgsz,
        "optimizer": "AdamW",
        "learning_rate_initial": lr0,
        "frozen_layers": freeze,
        "device": device,
        "training_duration_seconds": round(elapsed, 2),
        "checkpoint_path": str(best_pt),
        "sha256": sha256,
        "status": "candidate"
    }

    manifest_path = exp_dir / "training_manifest.yaml"
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Wrote training manifest to {manifest_path}")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Train IBVAP Security Item Detector v1")
    parser.add_argument("--base-model", type=str, default="yolo11n.pt")
    parser.add_argument("--dataset", type=str, default="data/normalized/security_item_v1/dataset.yaml")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--freeze", type=int, default=10)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run-name", type=str, default="ibvap_security_item_v1_exp001")
    args = parser.parse_args()

    train_security_item_v1(
        base_model=args.base_model,
        dataset_yaml=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr0,
        freeze=args.freeze,
        device=args.device,
        run_name=args.run_name
    )


if __name__ == "__main__":
    main()
