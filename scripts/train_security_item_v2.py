"""
IBVAP — Security Item Model v2 Training Engine
Trains candidate models for Experiment 001 (YOLO11n) and Experiment 002 (YOLO11s)
on data/normalized/security_item_v2/.
Outputs training manifest and computes SHA-256 integrity hash.
"""

import os
import sys
import time
import hashlib
import logging
import argparse
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainSecurityItemV2")


def compute_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().upper()


def train_security_item_v2(
    base_model: str = "yolo11n.pt",
    dataset_yaml: str = "data/normalized/security_item_v2/dataset.yaml",
    experiment_name: str = "ibvap_security_item_v2_exp001",
    epochs: int = 10,
    batch_size: int = 16,
    imgsz: int = 640,
    freeze_layers: int = 10,
    device: str = "0"
):
    start_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    logger.info("=" * 70)
    logger.info(f"STARTING IBVAP SECURITY ITEM V2 TRAINING: {experiment_name}")
    logger.info(f"Base Model: {base_model} | Dataset: {dataset_yaml} | Imgsz: {imgsz}")
    logger.info("=" * 70)

    dataset_path = Path(dataset_yaml).resolve()
    if not dataset_path.exists():
        logger.critical(f"Dataset YAML not found: {dataset_path}")
        sys.exit(1)

    selected_device = device if torch.cuda.is_available() and device != "cpu" else "cpu"
    logger.info(f"Target compute device: {selected_device} (GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'})")

    model = YOLO(base_model)

    runs_dir = Path("data/training/runs").resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)

    results = model.train(
        data=str(dataset_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        freeze=freeze_layers,
        mosaic=0.8,
        mixup=0.15,
        scale=0.3,
        fliplr=0.5,
        seed=42,
        device=selected_device,
        project=str(runs_dir),
        name=experiment_name,
        exist_ok=True,
        verbose=True
    )

    exp_dir = runs_dir / experiment_name
    best_weights = exp_dir / "weights" / "best.pt"

    if not best_weights.exists():
        logger.critical(f"Expected best weights not found at: {best_weights}")
        sys.exit(1)

    sha256_hash = compute_sha256(best_weights)
    file_size_bytes = best_weights.stat().st_size
    logger.info("=" * 70)
    logger.info(f"TRAINING COMPLETE: {experiment_name}")
    logger.info(f"Weights: {best_weights} ({file_size_bytes:,} bytes)")
    logger.info(f"SHA-256 Checksum: {sha256_hash}")
    logger.info("=" * 70)

    manifest = {
        "experiment_name": experiment_name,
        "base_model": base_model,
        "dataset_yaml": str(dataset_path),
        "target_class": {0: "firearm"},
        "epochs": epochs,
        "batch_size": batch_size,
        "imgsz": imgsz,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "freeze_layers": freeze_layers,
        "device": selected_device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "start_time": start_time,
        "completion_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint_path": str(best_weights.relative_to(Path.cwd()) if best_weights.is_relative_to(Path.cwd()) else best_weights),
        "file_size_bytes": file_size_bytes,
        "sha256": sha256_hash
    }

    manifest_file = exp_dir / "training_manifest.yaml"
    with open(manifest_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    logger.info(f"Training manifest saved to: {manifest_file}")
    return best_weights, sha256_hash


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IBVAP Security Item Model v2")
    parser.add_argument("--base-model", default="yolo11n.pt", help="Base model weights")
    parser.add_argument("--dataset", default="data/normalized/security_item_v2/dataset.yaml", help="Dataset YAML path")
    parser.add_argument("--name", default="ibvap_security_item_v2_exp001", help="Experiment run name")
    parser.add_argument("--epochs", type=int, default=10, help="Epoch count")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", default="0", help="CUDA device or cpu")
    args = parser.parse_args()

    train_security_item_v2(
        base_model=args.base_model,
        dataset_yaml=args.dataset,
        experiment_name=args.name,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device
    )
