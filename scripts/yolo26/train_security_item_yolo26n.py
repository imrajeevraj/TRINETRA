#!/usr/bin/env python3
"""
IBVAP — YOLO26 Security Item Model Training (IBVAP-ITEM-Y26-001)
Mirror of scripts/train_security_item_v2_1.py — base_model changed to yolo26n.pt.

NOTE: YOLO26 cannot be fine-tuned from the YOLO11 v2.1 checkpoint (different architecture).
This trains from yolo26n.pt (COCO-pretrained), identical to how Security Item v1.0 was
originally bootstrapped before fine-tuning began.

Critical limitation MUST be preserved:
  ⚠ REAL-FIREARM-VIDEO VALIDATION NOT AVAILABLE
"""

import os
import sys
import json
import time
import hashlib
import argparse
import logging
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SecurityItemY26Trainer")

EXPERIMENT_ID = "IBVAP-ITEM-Y26-001"
DEFAULT_RUN_NAME = "ibvap_security_item_y26n_exp001"
CANDIDATE_DST = "models/candidates/yolo26/security_item"
CRITICAL_LIMITATION = "REAL-FIREARM-VIDEO VALIDATION NOT AVAILABLE"


def calculate_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()


def train_security_item_yolo26n(
    base_checkpoint: str = "yolo26n.pt",          # ← from scratch, not YOLO11 weights
    dataset_yaml: str = "data/normalized/security_item_v2_1/dataset.yaml",
    experiment_name: str = DEFAULT_RUN_NAME,
    epochs: int = 10,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = "0",
):
    logger.info("=" * 70)
    logger.info(f"IBVAP SECURITY ITEM YOLO26n — CONTROLLED TRAINING: {EXPERIMENT_ID}")
    logger.info(f"⚠ NOTE: {CRITICAL_LIMITATION}")
    logger.info(f"Base checkpoint: {base_checkpoint} (YOLO26 from COCO pretrain)")
    logger.info("=" * 70)

    dataset_path = Path(dataset_yaml).resolve()
    if not dataset_path.exists():
        logger.critical(f"Dataset YAML not found: {dataset_path}")
        sys.exit(1)

    selected_device = device if torch.cuda.is_available() and device != "cpu" else "cpu"
    gpu_info = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {selected_device} | GPU: {gpu_info}")

    model = YOLO(base_checkpoint)

    runs_dir = Path("data/training/runs").resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)

    start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t0 = time.time()
    model.train(
        data=str(dataset_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        optimizer="AdamW",
        lr0=0.0005,
        lrf=0.01,
        weight_decay=0.0005,
        freeze=0,           # No freeze — YOLO26 weights not pretrained on IBVAP domain
        mosaic=0.5,
        mixup=0.10,
        scale=0.35,
        fliplr=0.5,
        seed=42,
        deterministic=True,
        device=selected_device,
        project=str(runs_dir),
        name=experiment_name,
        exist_ok=True,
        verbose=True,
    )
    elapsed = time.time() - t0
    logger.info(f"Training completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    exp_dir = runs_dir / experiment_name
    best_weights = exp_dir / "weights" / "best.pt"

    if not best_weights.exists():
        # Fallback: search for suffixed run dir
        candidates = sorted(runs_dir.glob(f"{experiment_name}*"), key=os.path.getmtime, reverse=True)
        for cd in candidates:
            cw = cd / "weights" / "best.pt"
            if cw.exists():
                best_weights = cw
                exp_dir = cd
                break

    if not best_weights.exists():
        logger.critical(f"Expected best weights not found at: {best_weights}")
        sys.exit(1)

    sha256_hash = calculate_sha256(best_weights)
    logger.info(f"Best checkpoint: {best_weights}")
    logger.info(f"SHA-256: {sha256_hash}")

    # Copy to candidates/
    dst_dir = Path(CANDIDATE_DST)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "best.pt"
    shutil.copy2(str(best_weights), str(dst))
    logger.info(f"Candidate copy: {dst}")

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "run_name": experiment_name,
        "architecture": "YOLO26n",
        "base_model": base_checkpoint,
        "note": "Trained from COCO-pretrained YOLO26n. YOLO11 weights incompatible (different architecture).",
        "dataset_yaml": str(dataset_path),
        "dataset_version": "security_item_v2_1",
        "epochs": epochs,
        "batch_size": batch_size,
        "image_size": imgsz,
        "optimizer": "AdamW",
        "lr0": 0.0005,
        "seed": 42,
        "device": selected_device,
        "gpu": gpu_info,
        "elapsed_seconds": round(elapsed, 2),
        "weights_path": str(best_weights),
        "candidate_path": str(dst),
        "sha256": sha256_hash,
        "critical_limitation": CRITICAL_LIMITATION,
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "ultralytics": __import__("ultralytics").__version__,
            "cuda": torch.version.cuda,
        },
        "started_at": start,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    manifest_path = exp_dir / "experiment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rep_dir = Path("data/reports/yolo26")
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "security_item_y26n_training_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    logger.info(f"Experiment manifest: {manifest_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Train IBVAP Security Item YOLO26n")
    parser.add_argument("--base-checkpoint", default="yolo26n.pt")
    parser.add_argument("--dataset", default="data/normalized/security_item_v2_1/dataset.yaml")
    parser.add_argument("--experiment-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    train_security_item_yolo26n(
        base_checkpoint=args.base_checkpoint,
        dataset_yaml=args.dataset,
        experiment_name=args.experiment_name,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
    )


if __name__ == "__main__":
    main()
