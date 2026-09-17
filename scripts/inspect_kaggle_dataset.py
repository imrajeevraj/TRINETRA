#!/usr/bin/env python3
"""
IBVAP — Dataset Inspection & Quality Audit Tool
Audits image resolutions, label counts, invalid bounding boxes, duplicates,
and class distributions for training datasets in data/training/.
"""

import os
import sys
import argparse
import hashlib
from pathlib import Path
from collections import Counter
import cv2
import yaml

def hash_image(image_path: Path) -> str:
    """Compute perceptual or binary hash for duplicate detection."""
    hasher = hashlib.md5()
    with open(image_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def inspect_dataset(dataset_dir: Path, class_mapping_file: str | None = None):
    print("=" * 70)
    print(f"IBVAP DATASET INSPECTION REPORT: {dataset_dir.name}")
    print(f"Path: {dataset_dir.resolve()}")
    print("=" * 70)

    if not dataset_dir.exists():
        print(f"ERROR: Dataset path does not exist: {dataset_dir}")
        return None

    # Load class mapping if provided
    class_map = {}
    if class_mapping_file and os.path.exists(class_mapping_file):
        with open(class_mapping_file, "r") as f:
            cmap_data = yaml.safe_load(f) or {}
            class_map = cmap_data.get("target_classes", {})

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_images = [f for f in dataset_dir.rglob("*") if f.suffix.lower() in image_extensions]
    all_labels = [f for f in dataset_dir.rglob("*") if f.suffix.lower() == ".txt" and f.name != "classes.txt"]

    print(f"Total Image Files Discovered: {len(all_images)}")
    print(f"Total Label Files Discovered: {len(all_labels)}")

    corrupted_images = []
    resolutions = Counter()
    seen_hashes = {}
    duplicates = []
    
    # Check images
    for img_path in all_images:
        # Check duplicate
        img_hash = hash_image(img_path)
        if img_hash in seen_hashes:
            duplicates.append((img_path.name, seen_hashes[img_hash].name))
        else:
            seen_hashes[img_hash] = img_path

        # Check readable
        img = cv2.imread(str(img_path))
        if img is None:
            corrupted_images.append(str(img_path))
            continue
        h, w = img.shape[:2]
        resolutions[f"{w}x{h}"] += 1

    # Check labels and bounding boxes
    class_counter = Counter()
    invalid_boxes = 0
    empty_labels = 0

    for lbl_path in all_labels:
        try:
            with open(lbl_path, "r") as f:
                lines = f.readlines()
                if not lines:
                    empty_labels += 1
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        
                        # Validate normalized coords
                        if cx < 0 or cx > 1 or cy < 0 or cy > 1 or w <= 0 or w > 1 or h <= 0 or h > 1:
                            invalid_boxes += 1
                        
                        cls_name = class_map.get(cls_id, str(cls_id))
                        class_counter[cls_name] += 1
        except Exception:
            invalid_boxes += 1

    # Summary
    print("\n--- Image Resolution Breakdown ---")
    for res, count in resolutions.most_common(5):
        print(f"  {res}: {count} image(s)")

    print("\n--- Class Distribution ---")
    for cls_name, count in sorted(class_counter.items(), key=lambda x: x[0]):
        print(f"  Class [{cls_name}]: {count} annotations")

    print("\n--- Data Integrity & Quality Metrics ---")
    print(f"  Corrupted Images:     {len(corrupted_images)}")
    print(f"  Duplicate Images:     {len(duplicates)}")
    print(f"  Invalid Bounding Boxes: {invalid_boxes}")
    print(f"  Empty Label Files:    {empty_labels}")

    if duplicates:
        print(f"\n  Sample Duplicates: {duplicates[:3]}")

    status = "READY" if len(corrupted_images) == 0 and invalid_boxes == 0 else "WARNINGS_DETECTED"
    print(f"\nOverall Compatibility Status: {status}")
    print("=" * 70)

    return {
        "total_images": len(all_images),
        "total_labels": len(all_labels),
        "corrupted": len(corrupted_images),
        "duplicates": len(duplicates),
        "invalid_boxes": invalid_boxes,
        "class_distribution": dict(class_counter),
        "resolutions": dict(resolutions)
    }

def main():
    parser = argparse.ArgumentParser(description="IBVAP Dataset Quality Inspection")
    parser.add_argument("--dataset", type=str, required=True, help="Directory of dataset to inspect")
    parser.add_argument("--class-mapping", type=str, default="configs/class_mapping.yaml", help="Class mapping YAML file")
    args = parser.parse_args()

    inspect_dataset(Path(args.dataset), args.class_mapping)

if __name__ == "__main__":
    main()
