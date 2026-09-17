#!/usr/bin/env python3
"""
IBVAP — Training Dataset Preprocessor & Split Assembler
Cleans, normalizes, and packages training data into data/training/datasets/IBVAP-TRAIN-v1.0/.
Enforces strict exclusion of the frozen benchmark (benchmark/labels/test/).
"""

import os
import sys
import shutil
import argparse
import logging
from pathlib import Path
import yaml
import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DatasetPreparer")

def clean_and_normalize_label(src_path: Path, dst_path: Path, class_map: dict | None = None) -> int:
    """Clean comments like '# STATUS: APPROVED', filter invalid lines, and write standard YOLO format."""
    valid_boxes = 0
    clean_lines = []
    if not src_path.exists():
        return 0

    with open(src_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue
            parts = line_str.split()
            if len(parts) >= 5:
                try:
                    cls_id = int(parts[0])
                    # Apply mapping if needed
                    if class_map and cls_id in class_map:
                        cls_id = class_map[cls_id]
                    
                    # IBVAP only trains on 0 (person) and 1 (vehicle)
                    if cls_id not in [0, 1]:
                        continue

                    cx, cy, w, h = map(float, parts[1:5])
                    # Clamp to [0.0, 1.0]
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    w = max(0.001, min(1.0, w))
                    h = max(0.001, min(1.0, h))

                    clean_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    valid_boxes += 1
                except ValueError:
                    continue

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "w", encoding="utf-8") as f:
        f.writelines(clean_lines)

    return valid_boxes

def assemble_ibvap_train_v1(
    output_dir: Path = Path("data/training/datasets/IBVAP-TRAIN-v1.0"),
    train_src_labels: Path = Path("benchmark/labels/train"),
    train_src_images: Path = Path("benchmark/images/train"),
    val_src_labels: Path = Path("benchmark/labels/valid"),
    val_src_images: Path = Path("benchmark/images/valid")
):
    logger.info("=" * 70)
    logger.info("ASSEMBLING DATASET: IBVAP-TRAIN-v1.0")
    logger.info(f"Target Output: {output_dir.resolve()}")
    logger.info("=" * 70)

    # 1. Anti-leakage check
    test_labels_dir = Path("benchmark/labels/test").resolve()
    logger.info("Verifying strict benchmark isolation...")
    if train_src_labels.resolve() == test_labels_dir or val_src_labels.resolve() == test_labels_dir:
        logger.critical("SECURITY VIOLATION: Attempted to assemble training set using frozen test benchmark!")
        sys.exit(1)
    logger.info("Anti-leakage check: PASS (Frozen benchmark test set is strictly excluded).")

    # 2. Setup directories
    train_img_dst = output_dir / "train" / "images"
    train_lbl_dst = output_dir / "train" / "labels"
    val_img_dst = output_dir / "val" / "images"
    val_lbl_dst = output_dir / "val" / "labels"

    for d in [train_img_dst, train_lbl_dst, val_img_dst, val_lbl_dst]:
        d.mkdir(parents=True, exist_ok=True)

    # 3. Process Train Split
    train_count = 0
    train_boxes = 0
    if train_src_images.exists():
        for img_p in train_src_images.glob("*.jpg"):
            lbl_p = train_src_labels / f"{img_p.stem}.txt"
            dst_img = train_img_dst / img_p.name
            dst_lbl = train_lbl_dst / f"{img_p.stem}.txt"

            if not dst_img.exists():
                shutil.copy2(img_p, dst_img)
            b = clean_and_normalize_label(lbl_p, dst_lbl)
            train_count += 1
            train_boxes += b

    logger.info(f"Train Split: {train_count} images, {train_boxes} bounding boxes.")

    # 4. Process Validation Split
    val_count = 0
    val_boxes = 0
    if val_src_images.exists():
        for img_p in val_src_images.glob("*.jpg"):
            lbl_p = val_src_labels / f"{img_p.stem}.txt"
            dst_img = val_img_dst / img_p.name
            dst_lbl = val_lbl_dst / f"{img_p.stem}.txt"

            if not dst_img.exists():
                shutil.copy2(img_p, dst_img)
            b = clean_and_normalize_label(lbl_p, dst_lbl)
            val_count += 1
            val_boxes += b

    logger.info(f"Validation Split: {val_count} images, {val_boxes} bounding boxes.")

    # 5. Generate YOLO data.yaml
    data_yaml_content = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": {
            0: "person",
            1: "vehicle"
        }
    }

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml_content, f, sort_keys=False)

    logger.info(f"Generated YOLO descriptor: {yaml_path}")
    logger.info("Dataset assembly complete.")
    return output_dir

def main():
    parser = argparse.ArgumentParser(description="Assemble IBVAP-TRAIN-v1.0 dataset")
    parser.add_argument("--output", type=str, default="data/training/datasets/IBVAP-TRAIN-v1.0", help="Destination directory")
    args = parser.parse_args()

    assemble_ibvap_train_v1(Path(args.output))

if __name__ == "__main__":
    main()
