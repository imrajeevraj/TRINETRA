#!/usr/bin/env python3
"""
IBVAP — YOLO26 Ground Model Training (IBVAP-GROUND-Y26-001)
Exact mirror of scripts/train_ground_v2.py — only the base_model changes.
All hyperparameters, dataset, augmentation, and seed are identical to YOLO11 v2.0.

IMPORTANT: This script NEVER modifies any existing YOLO11 checkpoint.
Output: data/training/runs/ibvap_ground_y26n_exp001/
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
logger = logging.getLogger("GroundY26Trainer")

EXPERIMENT_ID = "IBVAP-GROUND-Y26-001"
DEFAULT_RUN_NAME = "ibvap_ground_y26n_exp001"
CANDIDATE_DST = "models/candidates/yolo26/ground"


def calculate_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()


def train_ground_yolo26n(
    dataset_yaml: str = "data/normalized/ground_v2_exp002/dataset.yaml",
    base_model: str = "yolo26n.pt",          # ← ONLY CHANGE vs YOLO11 script
    epochs: int = 40,
    batch_size: int = 16,
    imgsz: int = 768,                         # ← same 768px as YOLO11 Ground v2.0
    device: str = "cuda:0",
    run_name: str = DEFAULT_RUN_NAME,
    seed: int = 42,
    lr0: float = 0.002,
    lrf: float = 0.0001,
    cos_lr: bool = True,
    scale: float = 0.1,
    mosaic: float = 0.2,
    close_mosaic: int = 10,
    translate: float = 0.05,
):
    logger.info("=" * 70)
    logger.info(f"IBVAP GROUND YOLO26n — CONTROLLED TRAINING: {EXPERIMENT_ID}")
    logger.info(f"Base model: {base_model} | Run: {run_name}")
    logger.info("=" * 70)

    yaml_p = Path(dataset_yaml)
    if not yaml_p.exists():
        logger.critical(f"Dataset YAML not found: {yaml_p}")
        sys.exit(1)

    with open(yaml_p, "r", encoding="utf-8") as f:
        ds_info = yaml.safe_load(f)

    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA unavailable — falling back to CPU.")
        device = "cpu"

    logger.info(f"Compute device: {device.upper()}")
    gpu_info = "CPU"
    if device.startswith("cuda"):
        gpu_info = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        logger.info(f"GPU: {gpu_info} ({vram_mb:.1f} MB VRAM)")

    logger.info(f"Loading YOLO26n base: {base_model}")
    model = YOLO(base_model)

    runs_dir = Path("data/training/runs")
    runs_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Epochs: {epochs} | Batch: {batch_size} | Imgsz: {imgsz}px | Seed: {seed}")
    logger.info(f"Dataset: {yaml_p.resolve()} | Classes: {ds_info.get('names')}")

    start = time.time()
    model.train(
        data=str(yaml_p.resolve()),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project=str(runs_dir.resolve()),
        name=run_name,
        seed=seed,
        deterministic=True,
        optimizer="AdamW",
        lr0=lr0,
        lrf=lrf,
        cos_lr=cos_lr,
        weight_decay=0.0005,
        # ── Augmentation (identical to YOLO11 v2.0 recipe) ─────────────────
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
        translate=translate,
        scale=scale,
        fliplr=0.5,
        flipud=0.0,
        degrees=0.0,
        mosaic=mosaic,
        close_mosaic=close_mosaic,
        # ── Save ────────────────────────────────────────────────────────────
        save=True,
        verbose=True,
    )
    elapsed = time.time() - start
    logger.info(f"Training completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Locate best weights
    out_dir = runs_dir / run_name
    best = out_dir / "weights" / "best.pt"
    if not best.exists():
        # Ultralytics may suffix the run name
        candidates = sorted(runs_dir.glob(f"{run_name}*"), key=os.path.getmtime, reverse=True)
        for cd in candidates:
            cw = cd / "weights" / "best.pt"
            if cw.exists():
                best = cw
                out_dir = cd
                break

    if not best.exists():
        logger.critical("best.pt not found after training!")
        sys.exit(1)

    sha = calculate_sha256(best)
    logger.info(f"Best checkpoint: {best}")
    logger.info(f"SHA-256: {sha}")

    # Copy to candidates/
    import shutil
    dst_dir = Path(CANDIDATE_DST)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "best.pt"
    shutil.copy2(str(best), str(dst))
    logger.info(f"Candidate copy: {dst}")

    # Experiment manifest
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "run_name": run_name,
        "architecture": "YOLO26n",
        "base_model": base_model,
        "dataset_yaml": str(yaml_p),
        "dataset_version": "ground_v2_exp002",
        "epochs": epochs,
        "batch_size": batch_size,
        "image_size": imgsz,
        "optimizer": "AdamW",
        "lr0": lr0,
        "lrf": lrf,
        "cos_lr": cos_lr,
        "seed": seed,
        "device": device,
        "gpu": gpu_info,
        "elapsed_seconds": round(elapsed, 2),
        "weights_path": str(best),
        "candidate_path": str(dst),
        "sha256": sha,
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "ultralytics": __import__("ultralytics").__version__,
            "cuda": torch.version.cuda,
        },
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    manifest_path = out_dir / "experiment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(f"Experiment manifest: {manifest_path}")

    # Also save to reports
    rep_dir = Path("data/reports/yolo26")
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "ground_y26n_training_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Train IBVAP Ground Model YOLO26n")
    parser.add_argument("--dataset", default="data/normalized/ground_v2_exp002/dataset.yaml")
    parser.add_argument("--base-model", default="yolo26n.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_ground_yolo26n(
        dataset_yaml=args.dataset,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        run_name=args.run_name,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
