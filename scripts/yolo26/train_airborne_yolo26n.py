#!/usr/bin/env python3
"""
IBVAP — YOLO26 Airborne Model Training (IBVAP-AIR-Y26-001)
Exact mirror of scripts/train_airborne_v1.py — only base_model changes.
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
import yaml
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AirborneY26Trainer")

EXPERIMENT_ID = "IBVAP-AIR-Y26-001"
DEFAULT_RUN_NAME = "ibvap_airborne_y26n_exp001"
CANDIDATE_DST = "models/candidates/yolo26/airborne"


def calculate_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()


def train_airborne_yolo26n(
    dataset_yaml: str = "data/normalized/airborne_v1_1/dataset.yaml",
    base_model: str = "yolo26n.pt",    # ← ONLY CHANGE vs YOLO11 script
    epochs: int = 30,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = "cuda:0",
    run_name: str = DEFAULT_RUN_NAME,
    seed: int = 42,
    close_mosaic: int = 8,
    workers: int = 2,
    lr0: float = 0.002,
):
    logger.info("=" * 70)
    logger.info(f"IBVAP AIRBORNE YOLO26n — CONTROLLED TRAINING: {EXPERIMENT_ID}")
    logger.info("=" * 70)

    yaml_p = Path(dataset_yaml)
    if not yaml_p.exists():
        logger.critical(f"Dataset YAML not found: {yaml_p}")
        sys.exit(1)

    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA unavailable — falling back to CPU.")
        device = "cpu"

    gpu_info = "CPU"
    if device.startswith("cuda"):
        gpu_info = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        logger.info(f"GPU: {gpu_info} ({vram_mb:.1f} MB VRAM)")

    model = YOLO(base_model)
    runs_dir = Path("data/training/runs")
    runs_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Epochs: {epochs} | Batch: {batch_size} | Imgsz: {imgsz}px | Seed: {seed}")
    logger.info("Classes: 0=drone, 1=aircraft")

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
        lrf=lr0 * 0.05,
        cos_lr=True,
        scale=0.1,          # Preserve distant aerial targets
        mosaic=0.2,         # Mild mosaic to preserve sky context
        close_mosaic=close_mosaic,
        workers=workers,
        save=True,
        verbose=True,
    )
    elapsed = time.time() - start
    logger.info(f"Training completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    out_dir = runs_dir / run_name
    best = out_dir / "weights" / "best.pt"
    if not best.exists():
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
    logger.info(f"Best checkpoint: {best} | SHA-256: {sha}")

    dst_dir = Path(CANDIDATE_DST)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "best.pt"
    shutil.copy2(str(best), str(dst))
    logger.info(f"Candidate copy: {dst}")

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "run_name": run_name,
        "architecture": "YOLO26n",
        "base_model": base_model,
        "dataset_yaml": str(yaml_p),
        "dataset_version": "airborne_v1_1",
        "epochs": epochs,
        "batch_size": batch_size,
        "image_size": imgsz,
        "optimizer": "AdamW",
        "lr0": lr0,
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

    (out_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rep_dir = Path("data/reports/yolo26")
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "airborne_y26n_training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Experiment manifest saved.")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Train IBVAP Airborne YOLO26n")
    parser.add_argument("--dataset", default="data/normalized/airborne_v1_1/dataset.yaml")
    parser.add_argument("--base-model", default="yolo26n.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_airborne_yolo26n(
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
