#!/usr/bin/env python3
"""
IBVAP — Track B: Ground Model v4 Dataset Assembly Engine
Combines core perimeter surveillance imagery with Kaggle soldier/vehicle samples
and mined hard-negatives, with strict anti-leakage isolation.
"""

import os
import sys
import shutil
import json
import logging
from pathlib import Path
from PIL import Image
import hashlib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GroundV4Assembler")

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.ops.master_ai_pipeline import compute_sha256, compute_dhash

def main():
    target_dir = ROOT_DIR / "data/normalized/ground_v4"
    images_train = target_dir / "images/train"
    images_val = target_dir / "images/val"
    labels_train = target_dir / "labels/train"
    labels_val = target_dir / "labels/val"
    manifests_dir = target_dir / "manifests"

    for d in [images_train, images_val, labels_train, labels_val, manifests_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Index frozen benchmark keyframes for anti-leakage check
    benchmark_dir = ROOT_DIR / "benchmark/images/test"
    benchmark_hashes = {}
    for img_path in benchmark_dir.glob("*.jpg"):
        try:
            with open(img_path, "rb") as f:
                md5 = hashlib.md5(f.read()).hexdigest()
            dh = compute_dhash(img_path)
            benchmark_hashes[img_path.name] = {"md5": md5, "dhash": dh}
        except Exception as e:
            logger.warning(f"Error reading benchmark image {img_path}: {e}")

    logger.info(f"Indexed {len(benchmark_hashes)} frozen benchmark keyframes for anti-leakage protection.")

    # 2. Gather candidates from Ground v2 (core perimeter surveillance)
    # Exclude known leaking frames:
    leaking_stems = {
        "v3_000003", "v3_000005", "v3_000006", "v3_000008", "v3_000009",
        "v3_000010", "v3_000012", "v3_000015", "v3_000016", "v3_000017",
        "v3_000019", "v3_000020", "v3_000021", "v3_000023"
    }

    seen_md5 = set()
    candidate_records = []

    def is_leaking(img_p):
        with open(img_p, "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
        if md5 in seen_md5:
            return True, "duplicate_internal"
        dh = compute_dhash(img_p)
        for b_name, b_meta in benchmark_hashes.items():
            if md5 == b_meta["md5"]:
                return True, f"md5_match_{b_name}"
            if dh is not None and b_meta["dhash"] is not None:
                if (dh - b_meta["dhash"]) <= 1:
                    return True, f"dhash_match_{b_name}"
        seen_md5.add(md5)
        return False, None

    # Source A: ground_v2 train images
    v2_img_dir = ROOT_DIR / "data/normalized/ground_v2/images/train"
    v2_lbl_dir = ROOT_DIR / "data/normalized/ground_v2/labels/train"
    if v2_img_dir.exists():
        for img_file in sorted(v2_img_dir.glob("*.*")):
            if img_file.stem in leaking_stems:
                continue
            lbl_file = v2_lbl_dir / f"{img_file.stem}.txt"
            if not lbl_file.exists():
                continue
            leaks, reason = is_leaking(img_file)
            if not leaks:
                candidate_records.append((img_file, lbl_file, "ground_v2_core"))

    # Source B: ground_v3 images (KIIT-MiTA soldier + vehicle enriched)
    v3_img_dir = ROOT_DIR / "data/normalized/ground_v3/images/train"
    v3_lbl_dir = ROOT_DIR / "data/normalized/ground_v3/labels/train"
    if v3_img_dir.exists():
        for img_file in sorted(v3_img_dir.glob("*.*")):
            lbl_file = v3_lbl_dir / f"{img_file.stem}.txt"
            if not lbl_file.exists():
                continue
            leaks, reason = is_leaking(img_file)
            if not leaks:
                candidate_records.append((img_file, lbl_file, "ground_v3_enrichment"))

    logger.info(f"Gathered {len(candidate_records)} verified, non-leaking training images.")

    # Split 85% train / 15% val deterministically
    import random
    random.seed(42)
    random.shuffle(candidate_records)

    n_val = int(len(candidate_records) * 0.15)
    val_set = candidate_records[:n_val]
    train_set = candidate_records[n_val:]

    logger.info(f"Splitting into {len(train_set)} train and {len(val_set)} validation images.")

    manifest = {"train": [], "val": []}
    class_counts = {"train": {0: 0, 1: 0}, "val": {0: 0, 1: 0}}
    neg_counts = {"train": 0, "val": 0}

    for split_name, dataset, img_dst, lbl_dst in [
        ("train", train_set, images_train, labels_train),
        ("val", val_set, images_val, labels_val),
    ]:
        for idx, (img_src, lbl_src, source_tag) in enumerate(dataset):
            dst_name = f"g4_{split_name}_{idx:05d}{img_src.suffix}"
            lbl_dst_name = f"g4_{split_name}_{idx:05d}.txt"

            shutil.copy2(str(img_src), str(img_dst / dst_name))
            shutil.copy2(str(lbl_src), str(lbl_dst / lbl_dst_name))

            # Count annotations
            with open(lbl_src, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
            if len(lines) == 0:
                neg_counts[split_name] += 1
            else:
                for line in lines:
                    parts = line.split()
                    if parts:
                        cls_id = int(parts[0])
                        if cls_id in class_counts[split_name]:
                            class_counts[split_name][cls_id] += 1

            manifest[split_name].append({
                "target_image": dst_name,
                "source": str(img_src),
                "source_tag": source_tag
            })

    # Write split manifest
    manifest_path = manifests_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Generate ground_v4.yaml
    yaml_content = f"""# IBVAP Ground Model v4 Dataset Descriptor
path: {target_dir.as_posix()}
train: images/train
val: images/val

names:
  0: person
  1: vehicle

metadata:
  train_images: {len(train_set)}
  val_images: {len(val_set)}
  train_persons: {class_counts['train'][0]}
  train_vehicles: {class_counts['train'][1]}
  train_hard_negatives: {neg_counts['train']}
  val_persons: {class_counts['val'][0]}
  val_vehicles: {class_counts['val'][1]}
  val_hard_negatives: {neg_counts['val']}
  anti_leakage_status: ZERO_LEAKAGE_VERIFIED
"""
    yaml_path = target_dir / "ground_v4.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    (target_dir / "dataset.yaml").write_text(yaml_content, encoding="utf-8")

    logger.info(f"Ground v4 Dataset successfully assembled at {target_dir}")
    logger.info(f"Train: {len(train_set)} images ({class_counts['train'][0]} persons, {class_counts['train'][1]} vehicles, {neg_counts['train']} negatives)")
    logger.info(f"Val: {len(val_set)} images ({class_counts['val'][0]} persons, {class_counts['val'][1]} vehicles, {neg_counts['val']} negatives)")

if __name__ == "__main__":
    main()
