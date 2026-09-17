"""
IBVAP — Security Item Model v2.1 Training Engine (Experiment 001)
Fine-tunes directly from the v2.0 checkpoint:
data/training/runs/ibvap_security_item_v2_exp001/weights/best.pt
Using Dataset v2.1 (Hard Negatives v3 + Small/Occluded Enriched Positives).
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
logger = logging.getLogger("TrainSecurityItemV2_1")


def compute_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().upper()


def train_security_item_v2_1(
    base_checkpoint: str = "data/training/runs/ibvap_security_item_v2_exp001/weights/best.pt",
    dataset_yaml: str = "data/normalized/security_item_v2_1/dataset.yaml",
    experiment_name: str = "ibvap_security_item_v2_1_exp001",
    epochs: int = 10,
    batch_size: int = 16,
    imgsz: int = 640,
    freeze_layers: int = 10,
    device: str = "0"
):
    start_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    logger.info("=" * 70)
    logger.info(f"STARTING IBVAP SECURITY ITEM V2.1 TRAINING: {experiment_name}")
    logger.info(f"Base Checkpoint: {base_checkpoint}")
    logger.info(f"Dataset: {dataset_yaml} | Imgsz: {imgsz}")
    logger.info("=" * 70)

    dataset_path = Path(dataset_yaml).resolve()
    if not dataset_path.exists():
        logger.critical(f"Dataset YAML not found: {dataset_path}")
        sys.exit(1)

    base_ckpt_path = Path(base_checkpoint).resolve()
    if not base_ckpt_path.exists():
        logger.critical(f"Base checkpoint not found: {base_ckpt_path}")
        sys.exit(1)

    selected_device = device if torch.cuda.is_available() and device != "cpu" else "cpu"
    logger.info(f"Compute device: {selected_device} (GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'})")

    # Load model from v2.0 checkpoint
    model = YOLO(str(base_ckpt_path))

    runs_dir = Path("data/training/runs").resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)

    results = model.train(
        data=str(dataset_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        optimizer="AdamW",
        lr0=0.0005,
        lrf=0.01,
        weight_decay=0.0005,
        freeze=freeze_layers,
        mosaic=0.5,
        mixup=0.10,
        scale=0.35,
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
        "base_checkpoint": str(base_ckpt_path),
        "dataset_yaml": str(dataset_path),
        "target_class": {0: "firearm"},
        "epochs": epochs,
        "batch_size": batch_size,
        "imgsz": imgsz,
        "optimizer": "AdamW",
        "lr0": 0.0005,
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

    logger.info(f"Training manifest written to {manifest_file}")
    return best_weights, sha256_hash


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="data/training/runs/ibvap_security_item_v2_exp001/weights/best.pt")
    parser.add_argument("--dataset", default="data/normalized/security_item_v2_1/dataset.yaml")
    parser.add_argument("--name", default="ibvap_security_item_v2_1_exp001")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    train_security_item_v2_1(
        base_checkpoint=args.base,
        dataset_yaml=args.dataset,
        experiment_name=args.name,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device
    )
