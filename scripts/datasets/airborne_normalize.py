#!/usr/bin/env python3
"""
IBVAP — Airborne Dataset Normalization & Small-Object Analysis Engine
Standardizes drone, aircraft, and aerial hard-negative annotations into:
  0 = drone
  1 = aircraft
Calculates object-size distributions and generates data/normalized/airborne_v1/.
"""

import os
import sys
import json
import random
import shutil
import argparse
import logging
from pathlib import Path
from collections import Counter
import yaml
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AirborneNormalize")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def normalize_airborne(
    raw_dir: Path = Path("data/raw/airborne"),
    output_dir: Path = Path("data/normalized/airborne_v1"),
    reports_dir: Path = Path("data/reports/airborne_v1"),
    split_ratio: float = 0.80,
    seed: int = 42,
    max_hard_negatives: int = 200
):
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_img_dir = output_dir / "images" / "train"
    val_img_dir = output_dir / "images" / "val"
    train_lbl_dir = output_dir / "labels" / "train"
    val_lbl_dir = output_dir / "labels" / "val"

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("NORMALIZING AIRBORNE MODEL V1 DATASET")
    logger.info(f"Target Directory: {output_dir.resolve()}")
    logger.info("=" * 70)

    pool = []
    object_sizes = {"drone": [], "aircraft": []}

    # 1. Drone Dataset (raw/airborne/drones)
    drone_dir = raw_dir / "drones"
    if drone_dir.exists():
        drone_images = [f for f in drone_dir.rglob("*") if f.suffix.lower() in IMAGE_EXTS]
        logger.info(f"Processing {len(drone_images)} drone images...")
        for img_p in drone_images:
            lbl_p = img_p.with_suffix(".txt")
            if lbl_p.exists():
                pool.append({
                    "img": img_p,
                    "lbl": lbl_p,
                    "target_class": 0, # 0 = drone
                    "name": f"drone_{img_p.stem}{img_p.suffix.lower()}",
                    "hard_neg": False
                })

    # 2. Fixed-Wing UAV / Aircraft Dataset (raw/airborne/fixed_wing)
    aircraft_dir = raw_dir / "fixed_wing"
    if aircraft_dir.exists():
        aircraft_images = [f for f in aircraft_dir.rglob("*") if f.suffix.lower() in IMAGE_EXTS]
        logger.info(f"Processing {len(aircraft_images)} fixed-wing aircraft images...")
        for img_p in aircraft_images:
            lbl_p = img_p.with_suffix(".txt")
            if lbl_p.exists():
                pool.append({
                    "img": img_p,
                    "lbl": lbl_p,
                    "target_class": 1, # 1 = aircraft
                    "name": f"aircraft_{img_p.stem}{img_p.suffix.lower()}",
                    "hard_neg": False
                })

    # 3. Birds & Sky Hard Negatives (raw/airborne/birds_negatives)
    bird_dir = raw_dir / "birds_negatives"
    if bird_dir.exists():
        bird_images = [f for f in bird_dir.rglob("*") if f.suffix.lower() in IMAGE_EXTS]
        random.seed(seed)
        sampled_birds = random.sample(bird_images, min(len(bird_images), max_hard_negatives))
        logger.info(f"Sampling {len(sampled_birds)} aerial bird hard-negative images...")
        for img_p in sampled_birds:
            pool.append({
                "img": img_p,
                "lbl": None, # Empty label for negative background
                "target_class": -1,
                "name": f"birdneg_{img_p.stem}{img_p.suffix.lower()}",
                "hard_neg": True
            })

    # Stratified Split
    random.seed(seed)
    random.shuffle(pool)

    split_idx = int(len(pool) * split_ratio)
    train_pool = pool[:split_idx]
    val_pool = pool[split_idx:]

    stats = {"train": Counter(), "val": Counter()}

    def process_item(item, target_img_dir, target_lbl_dir, split_name):
        dest_img = target_img_dir / item["name"]
        dest_lbl = target_lbl_dir / f"{Path(item['name']).stem}.txt"

        if not dest_img.exists():
            shutil.copy2(item["img"], dest_img)

        clean_lines = []
        if item["lbl"] and item["lbl"].exists():
            with open(item["lbl"], "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cx, cy, w, h = map(float, parts[1:5])
                            cx = max(0.0, min(1.0, cx))
                            cy = max(0.0, min(1.0, cy))
                            w = max(0.001, min(1.0, w))
                            h = max(0.001, min(1.0, h))
                            cid = item["target_class"]
                            clean_lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                            area = w * h
                            if cid == 0:
                                stats[split_name]["drone"] += 1
                                object_sizes["drone"].append(area)
                            elif cid == 1:
                                stats[split_name]["aircraft"] += 1
                                object_sizes["aircraft"].append(area)
                        except ValueError:
                            continue

        with open(dest_lbl, "w", encoding="utf-8") as f:
            f.writelines(clean_lines)

        if item["hard_neg"] or len(clean_lines) == 0:
            stats[split_name]["hard_negatives"] += 1

        stats[split_name]["images"] += 1

    for item in train_pool:
        process_item(item, train_img_dir, train_lbl_dir, "train")

    for item in val_pool:
        process_item(item, val_img_dir, val_lbl_dir, "val")

    # Generate dataset.yaml
    dataset_yaml = {
        "path": str(output_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "drone",
            1: "aircraft"
        }
    }
    with open(output_dir / "dataset.yaml", "w", encoding="utf-8") as f:
        yaml.dump(dataset_yaml, f, sort_keys=False)

    # Small-Object Analysis (Phase 9)
    produce_small_object_report(object_sizes, reports_dir)

    logger.info("=" * 70)
    logger.info("AIRBORNE DATASET ASSEMBLY SUMMARY:")
    logger.info(f"  TRAIN: {stats['train']['images']} images | {stats['train']['drone']} drones | {stats['train']['aircraft']} aircraft | {stats['train']['hard_negatives']} hard negatives")
    logger.info(f"  VAL:   {stats['val']['images']} images | {stats['val']['drone']} drones | {stats['val']['aircraft']} aircraft | {stats['val']['hard_negatives']} hard negatives")
    logger.info(f"Descriptor: {output_dir / 'dataset.yaml'}")
    logger.info("=" * 70)
    return stats

def produce_small_object_report(object_sizes, reports_dir):
    report_data = {}
    for cname in ["drone", "aircraft"]:
        areas = np.array(object_sizes[cname]) if object_sizes[cname] else np.array([0.01])
        v_small = int(np.sum(areas < 0.001))
        small = int(np.sum((areas >= 0.001) & (areas < 0.005)))
        med = int(np.sum((areas >= 0.005) & (areas < 0.05)))
        large = int(np.sum(areas >= 0.05))
        total = len(areas)

        report_data[cname] = {
            "total_instances": total,
            "min_area": float(np.min(areas)),
            "median_area": float(np.median(areas)),
            "max_area": float(np.max(areas)),
            "very_small_under_0_1pct": {"count": v_small, "pct": round(v_small/total*100, 1)},
            "small_0_1pct_to_0_5pct": {"count": small, "pct": round(small/total*100, 1)},
            "medium_0_5pct_to_5pct": {"count": med, "pct": round(med/total*100, 1)},
            "large_over_5pct": {"count": large, "pct": round(large/total*100, 1)}
        }

    with open(reports_dir / "airborne_size_distribution.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    md_content = f"""# IBVAP — Airborne Target Small-Object Analysis (Phase 9)

## 1. Object Size Classification Matrix

| Target Class | Total Instances | Very Small ($< 0.1\%$) | Small ($0.1\% - 0.5\%$) | Medium ($0.5\% - 5\%$) | Large ($\ge 5\%$) | Median Area |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Drone** | {report_data['drone']['total_instances']} | {report_data['drone']['very_small_under_0_1pct']['pct']}% | {report_data['drone']['small_0_1pct_to_0_5pct']['pct']}% | {report_data['drone']['medium_0_5pct_to_5pct']['pct']}% | {report_data['drone']['large_over_5pct']['pct']}% | {report_data['drone']['median_area']:.6f} |
| **Aircraft** | {report_data['aircraft']['total_instances']} | {report_data['aircraft']['very_small_under_0_1pct']['pct']}% | {report_data['aircraft']['small_0_1pct_to_0_5pct']['pct']}% | {report_data['aircraft']['medium_0_5pct_to_5pct']['pct']}% | {report_data['aircraft']['large_over_5pct']['pct']}% | {report_data['aircraft']['median_area']:.6f} |

## 2. Engineering Directives for Airborne Training
- Drones in perimeter surveillance are predominantly **Small or Very Small** targets (< 0.5% area).
- A minimum resolution of **640px** with minimal scaling degradation is mandatory.
- Bird hard negatives must be maintained at ~10-15% to suppress false triggers on avian flight paths.
"""
    with open(reports_dir / "airborne_size_distribution.md", "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info(f"Small-object reports generated in {reports_dir}")

def main():
    parser = argparse.ArgumentParser(description="Normalize Airborne Datasets")
    parser.add_argument("--raw", type=str, default="data/raw/airborne", help="Raw airborne directory")
    parser.add_argument("--output", type=str, default="data/normalized/airborne_v1", help="Normalized output directory")
    parser.add_argument("--reports", type=str, default="data/reports/airborne_v1", help="Reports directory")
    parser.add_argument("--hard-negatives", type=int, default=150, help="Max bird/sky hard negatives")
    args = parser.parse_args()

    normalize_airborne(
        raw_dir=Path(args.raw),
        output_dir=Path(args.output),
        reports_dir=Path(args.reports),
        max_hard_negatives=args.hard_negatives
    )

if __name__ == "__main__":
    main()
