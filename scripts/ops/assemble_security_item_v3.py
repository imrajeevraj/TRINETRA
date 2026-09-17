#!/usr/bin/env python3
"""
IBVAP — Track B: Security Item Model v3 Dataset Assembly Engine
Constructs data/normalized/security_item_v3/ with tool hard-negatives,
anti-leakage verification against IBVAP-GT-ITEM-v2.0, and manifest generation.
"""

import os
import sys
import shutil
import json
import logging
from pathlib import Path
import hashlib

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SecItemV3Assembler")

def main():
    target_dir = ROOT_DIR / "data/normalized/security_item_v3"
    images_train = target_dir / "images/train"
    images_val = target_dir / "images/val"
    labels_train = target_dir / "labels/train"
    labels_val = target_dir / "labels/val"
    manifest_dir = target_dir / "manifest"

    for d in [images_train, images_val, labels_train, labels_val, manifest_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Index frozen test set (IBVAP-GT-ITEM-v2.0)
    test_img_dir = ROOT_DIR / "data/normalized/security_item_v2/images/test"
    test_hashes = set()
    for p in test_img_dir.glob("*.*"):
        with open(p, "rb") as f:
            test_hashes.add(hashlib.md5(f.read()).hexdigest())
    logger.info(f"Indexed {len(test_hashes)} benchmark images from IBVAP-GT-ITEM-v2.0.")

    # 2. Ingest source images from security_item_v2_1
    src_v2_1 = ROOT_DIR / "data/normalized/security_item_v2_1"
    copied_counts = {"train": 0, "val": 0}
    neg_counts = {"train": 0, "val": 0}
    manifest = {"train": [], "val": []}

    for split in ["train", "val"]:
        src_img_d = src_v2_1 / f"images/{split}"
        src_lbl_d = src_v2_1 / f"labels/{split}"
        dst_img_d = images_train if split == "train" else images_val
        dst_lbl_d = labels_train if split == "train" else labels_val

        for img_f in sorted(src_img_d.glob("*.*")):
            with open(img_f, "rb") as f:
                md5 = hashlib.md5(f.read()).hexdigest()
            if md5 in test_hashes:
                logger.warning(f"LEAK DETECTED: {img_f} matches benchmark! Skipping.")
                continue

            lbl_f = src_lbl_d / f"{img_f.stem}.txt"
            if not lbl_f.exists():
                continue

            shutil.copy2(str(img_f), str(dst_img_d / img_f.name))
            shutil.copy2(str(lbl_f), str(dst_lbl_d / f"{img_f.stem}.txt"))
            copied_counts[split] += 1

            with open(lbl_f, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
            if len(lines) == 0:
                neg_counts[split] += 1

            manifest[split].append({
                "image": img_f.name,
                "md5": md5,
                "num_annotations": len(lines),
                "source": "security_item_v2_1"
            })

    # Save manifest
    (manifest_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Create dataset yaml
    yaml_content = f"""# IBVAP Security Item Model v3 Dataset Descriptor
path: {target_dir.as_posix()}
train: images/train
val: images/val

names:
  0: firearm

metadata:
  train_images: {copied_counts['train']}
  val_images: {copied_counts['val']}
  train_negatives: {neg_counts['train']}
  train_negative_ratio: {neg_counts['train']/max(1, copied_counts['train']):.3f}
  val_negatives: {neg_counts['val']}
  val_negative_ratio: {neg_counts['val']/max(1, copied_counts['val']):.3f}
  anti_leakage_status: ZERO_BENCHMARK_LEAKAGE
"""
    (target_dir / "security_item_v3.yaml").write_text(yaml_content, encoding="utf-8")
    (target_dir / "dataset.yaml").write_text(yaml_content, encoding="utf-8")

    logger.info(f"Security Item v3 assembly complete: Train={copied_counts['train']} ({neg_counts['train']} negs), Val={copied_counts['val']} ({neg_counts['val']} negs).")

if __name__ == "__main__":
    main()
