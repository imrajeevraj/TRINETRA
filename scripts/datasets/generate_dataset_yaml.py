#!/usr/bin/env python3
"""
IBVAP — Ground Model v2 Dataset Assembler & Split Engine
Integrates internal IBVAP surveillance data, normalized Kaggle corpora,
and hard-negative background scenes into data/normalized/ground_v2/.
Enforces deterministic train/val splitting and strict benchmark isolation.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DatasetGenerator")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def assemble_ground_v2(
    output_dir: Path = Path("data/normalized/ground_v2"),
    kiit_norm_dir: Path = Path("data/normalized/ground_v2/raw_normalized"),
    ibvap_train_imgs: Path = Path("benchmark/images/train"),
    ibvap_train_lbls: Path = Path("benchmark/labels/train"),
    ibvap_val_imgs: Path = Path("benchmark/images/valid"),
    ibvap_val_lbls: Path = Path("benchmark/labels/valid"),
    hard_neg_dir: Path = Path("data/external/kaggle/human_detection/human detection dataset/0"),
    split_ratio: float = 0.80,
    seed: int = 42,
    max_hard_negatives: int = 150,
    filter_empty_source: bool = True
):
    logger.info("=" * 70)
    logger.info("ASSEMBLING IBVAP GROUND MODEL V2 DATASET")
    logger.info(f"Target Directory: {output_dir.resolve()}")
    logger.info(f"Split Seed: {seed} | Train Ratio: {split_ratio:.2f}")
    logger.info("=" * 70)

    # 1. Anti-Leakage Gate Check
    frozen_test_dir = Path("benchmark/images/test").resolve()
    for check_p in [ibvap_train_imgs, ibvap_val_imgs]:
        if check_p.resolve() == frozen_test_dir:
            logger.critical("SECURITY VIOLATION: Frozen benchmark test images targeted for training split!")
            sys.exit(1)
    logger.info("Anti-leakage check: PASS (Frozen benchmark test set is strictly excluded).")

    train_img_dir = output_dir / "images" / "train"
    val_img_dir = output_dir / "images" / "val"
    train_lbl_dir = output_dir / "labels" / "train"
    val_lbl_dir = output_dir / "labels" / "val"

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    pool = [] # list of dicts: {"img_src", "lbl_src", "source", "name", "hard_negative"}

    # 2. Add Existing IBVAP internal surveillance samples
    logger.info("Integrating internal IBVAP surveillance keyframes...")
    ibvap_count = 0
    for img_p in list(ibvap_train_imgs.glob("*.jpg")) + list(ibvap_val_imgs.glob("*.jpg")):
        # Determine label path
        lbl_dir = ibvap_train_lbls if "train" in str(img_p.parent) else ibvap_val_lbls
        lbl_p = lbl_dir / f"{img_p.stem}.txt"
        if lbl_p.exists():
            pool.append({
                "img_src": img_p,
                "lbl_src": lbl_p,
                "source": "ibvap_internal",
                "name": f"ibvap_{img_p.name}",
                "hard_negative": False
            })
            ibvap_count += 1
    logger.info(f"Added {ibvap_count} internal IBVAP surveillance samples.")

    # 3. Add Normalized KIIT-MiTA surveillance samples
    logger.info("Integrating normalized KIIT-MiTA CCTV surveillance samples...")
    kiit_count = 0
    kiit_empty_count = 0
    kiit_imgs = kiit_norm_dir / "images"
    kiit_lbls = kiit_norm_dir / "labels"
    if kiit_imgs.exists():
        for img_p in sorted(kiit_imgs.glob("*")):
            lbl_p = kiit_lbls / f"{img_p.stem}.txt"
            if lbl_p.exists():
                has_boxes = False
                with open(lbl_p, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if line.strip() and not line.strip().startswith("#"):
                            has_boxes = True
                            break
                if has_boxes or not filter_empty_source:
                    pool.append({
                        "img_src": img_p,
                        "lbl_src": lbl_p,
                        "source": "kaggle_kiit_mita",
                        "name": img_p.name,
                        "hard_negative": not has_boxes
                    })
                    kiit_count += 1
                else:
                    kiit_empty_count += 1
    logger.info(f"Added {kiit_count} normalized Kaggle surveillance samples ({kiit_empty_count} non-target background images excluded).")

    # 4. Add Controlled Hard-Negative Background Scenes
    logger.info("Integrating controlled hard-negative background scenes...")
    hard_neg_count = 0
    if hard_neg_dir.exists():
        hn_files = sorted(list(hard_neg_dir.glob("*.jpg")) + list(hard_neg_dir.glob("*.png")))
        random.seed(seed)
        sampled_hn = random.sample(hn_files, min(len(hn_files), max_hard_negatives))
        for img_p in sampled_hn:
            pool.append({
                "img_src": img_p,
                "lbl_src": None, # Will create empty label file
                "source": "kaggle_hard_negative",
                "name": f"hardneg_{img_p.name}",
                "hard_negative": True
            })
            hard_neg_count += 1
    logger.info(f"Added {hard_neg_count} hard-negative background scenes.")

    # 5. Deterministic Stratified Split
    random.seed(seed)
    random.shuffle(pool)

    split_idx = int(len(pool) * split_ratio)
    train_samples = pool[:split_idx]
    val_samples = pool[split_idx:]

    logger.info(f"Partitioned dataset: {len(train_samples)} TRAIN samples, {len(val_samples)} VAL samples.")

    split_manifest = {
        "dataset_version": "ground_v2",
        "seed": seed,
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "train_samples": [],
        "val_samples": []
    }

    stats = {
        "train": Counter(),
        "val": Counter()
    }

    # Helper to clean and copy
    def process_sample(sample, target_img_dir, target_lbl_dir, split_name):
        dest_img = target_img_dir / sample["name"]
        dest_lbl = target_lbl_dir / f"{Path(sample['name']).stem}.txt"

        if not dest_img.exists():
            shutil.copy2(sample["img_src"], dest_img)

        clean_lines = []
        p_count = 0
        v_count = 0

        if sample["lbl_src"] and sample["lbl_src"].exists():
            with open(sample["lbl_src"], "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line_str = line.strip()
                    if not line_str or line_str.startswith("#"):
                        continue
                    parts = line_str.split()
                    if len(parts) >= 5:
                        try:
                            cid = int(parts[0])
                            if cid in [0, 1]:
                                cx, cy, w, h = map(float, parts[1:5])
                                cx = max(0.0, min(1.0, cx))
                                cy = max(0.0, min(1.0, cy))
                                w = max(0.001, min(1.0, w))
                                h = max(0.001, min(1.0, h))
                                clean_lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                                if cid == 0:
                                    p_count += 1
                                    stats[split_name]["person"] += 1
                                elif cid == 1:
                                    v_count += 1
                                    stats[split_name]["vehicle"] += 1
                        except ValueError:
                            continue

        with open(dest_lbl, "w", encoding="utf-8") as f:
            f.writelines(clean_lines)

        if sample["hard_negative"] or len(clean_lines) == 0:
            stats[split_name]["hard_negatives"] += 1

        stats[split_name]["images"] += 1

        return {
            "image": sample["name"],
            "source": sample["source"],
            "person_count": p_count,
            "vehicle_count": v_count,
            "hard_negative": sample["hard_negative"]
        }

    for s in train_samples:
        rec = process_sample(s, train_img_dir, train_lbl_dir, "train")
        split_manifest["train_samples"].append(rec)

    for s in val_samples:
        rec = process_sample(s, val_img_dir, val_lbl_dir, "val")
        split_manifest["val_samples"].append(rec)

    # 6. Generate dataset.yaml
    dataset_yaml = {
        "path": str(output_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "person",
            1: "vehicle"
        }
    }

    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(dataset_yaml, f, sort_keys=False)

    split_manifest_path = output_dir / "split_manifest.json"
    with open(split_manifest_path, "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2)

    logger.info("=" * 70)
    logger.info("IBVAP GROUND MODEL V2 DATASET SUMMARY:")
    logger.info(f"  TRAIN: {stats['train']['images']} images | {stats['train']['person']} persons | {stats['train']['vehicle']} vehicles | {stats['train']['hard_negatives']} hard negatives")
    logger.info(f"  VAL:   {stats['val']['images']} images | {stats['val']['person']} persons | {stats['val']['vehicle']} vehicles | {stats['val']['hard_negatives']} hard negatives")
    logger.info(f"YOLO descriptor: {yaml_path}")
    logger.info(f"Split manifest:  {split_manifest_path}")
    logger.info("=" * 70)

    return stats

def main():
    parser = argparse.ArgumentParser(description="Assemble Ground Model v2 Dataset")
    parser.add_argument("--output", type=str, default="data/normalized/ground_v2", help="Output directory")
    parser.add_argument("--split", type=float, default=0.80, help="Train split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    parser.add_argument("--hard-negatives", type=int, default=150, help="Max hard negatives to sample")
    parser.add_argument("--filter-empty-source", action="store_true", default=True, help="Filter out non-target empty files from source datasets")
    args = parser.parse_args()

    assemble_ground_v2(
        output_dir=Path(args.output),
        split_ratio=args.split,
        seed=args.seed,
        max_hard_negatives=args.hard_negatives,
        filter_empty_source=args.filter_empty_source
    )

if __name__ == "__main__":
    main()
