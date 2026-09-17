#!/usr/bin/env python3
"""
IBVAP — Multi-Format Dataset Normalizer & Class Standardization Engine
Converts YOLO, YOLOv5/v8 directory structures, and COCO JSON annotations into
the canonical IBVAP taxonomy: 0 = person, 1 = vehicle.
Tracks complete data provenance in manifest.json.
"""

import os
import sys
import json
import argparse
import logging
import shutil
from pathlib import Path
from collections import Counter
import yaml
import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DatasetNormalizer")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Canonical IBVAP Targets
IBVAP_CLASSES = {
    0: "person",
    1: "vehicle"
}

# Configurable source class to IBVAP mappings
DEFAULT_TAXONOMY_MAP = {
    # Person variants
    "person": 0,
    "soldier": 0,
    "pedestrian": 0,
    "human": 0,
    "people": 0,
    "man": 0,
    "woman": 0,

    # Vehicle variants
    "vehicle": 1,
    "tank": 1,
    "car": 1,
    "automobile": 1,
    "truck": 1,
    "bus": 1,
    "van": 1,
    "suv": 1,
    "motorcycle": 1,
    "motorbike": 1,
    "jeep": 1,
    "tractor": 1,
    "pickup": 1
}

def parse_yolo_dataset(
    source_dir: Path,
    output_dir: Path,
    source_name_map: dict[int, str],
    taxonomy_map: dict[str, int],
    dataset_name: str
):
    """Normalize a YOLO-format dataset."""
    norm_images_dir = output_dir / "images"
    norm_labels_dir = output_dir / "labels"
    norm_images_dir.mkdir(parents=True, exist_ok=True)
    norm_labels_dir.mkdir(parents=True, exist_ok=True)

    image_files = [f for f in source_dir.rglob("*") if f.suffix.lower() in IMAGE_EXTS]
    label_files = {f.stem: f for f in source_dir.rglob("*") if f.suffix.lower() == ".txt" and f.name != "classes.txt"}

    manifest_records = []
    stats = Counter()

    logger.info(f"Normalizing YOLO dataset '{dataset_name}': {len(image_files)} images found.")

    for img_p in image_files:
        stem = img_p.stem
        lbl_p = label_files.get(stem)

        # Copy image
        target_img_name = f"{dataset_name}_{stem}{img_p.suffix.lower()}"
        target_img_p = norm_images_dir / target_img_name
        if not target_img_p.exists():
            shutil.copy2(img_p, target_img_p)

        target_lbl_name = f"{dataset_name}_{stem}.txt"
        target_lbl_p = norm_labels_dir / target_lbl_name

        normalized_lines = []
        person_count = 0
        vehicle_count = 0

        if lbl_p and lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line_str = line.strip()
                    if not line_str or line_str.startswith("#"):
                        continue
                    parts = line_str.split()
                    if len(parts) >= 5:
                        try:
                            raw_id = int(parts[0])
                            cx, cy, w, h = map(float, parts[1:5])
                            
                            # Clamp geometry
                            cx = max(0.0, min(1.0, cx))
                            cy = max(0.0, min(1.0, cy))
                            w = max(0.001, min(1.0, w))
                            h = max(0.001, min(1.0, h))

                            # Map class name to canonical IBVAP class
                            raw_name = source_name_map.get(raw_id, str(raw_id)).lower()
                            if raw_name in taxonomy_map:
                                target_id = taxonomy_map[raw_name]
                                normalized_lines.append(f"{target_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                                if target_id == 0:
                                    person_count += 1
                                    stats["person"] += 1
                                elif target_id == 1:
                                    vehicle_count += 1
                                    stats["vehicle"] += 1
                            else:
                                stats["excluded_classes"] += 1
                        except ValueError:
                            continue

        # Write clean label file
        with open(target_lbl_p, "w", encoding="utf-8") as f:
            f.writelines(normalized_lines)

        is_hard_negative = (len(normalized_lines) == 0)
        if is_hard_negative:
            stats["hard_negatives"] += 1
        else:
            stats["annotated_images"] += 1

        manifest_records.append({
            "image_id": target_img_name,
            "source_dataset": dataset_name,
            "source_path": str(img_p),
            "normalized_image_path": str(target_img_p),
            "normalized_label_path": str(target_lbl_p),
            "person_count": person_count,
            "vehicle_count": vehicle_count,
            "hard_negative": is_hard_negative
        })

    logger.info(f"Normalization complete for '{dataset_name}':")
    logger.info(f"  Person Annotations:  {stats['person']}")
    logger.info(f"  Vehicle Annotations: {stats['vehicle']}")
    logger.info(f"  Hard Negatives:      {stats['hard_negatives']}")
    logger.info(f"  Excluded Objects:    {stats['excluded_classes']}")

    return manifest_records, dict(stats)

def main():
    parser = argparse.ArgumentParser(description="IBVAP Dataset Normalizer")
    parser.add_argument("--input", type=str, required=True, help="Input raw dataset directory")
    parser.add_argument("--output", type=str, default="data/normalized/ground_v2", help="Output normalized directory")
    parser.add_argument("--name", type=str, default="kiit_mita", help="Dataset name identifier")
    parser.add_argument("--names-map", type=str, default=None, help="JSON or YAML mapping of raw class indices to names")
    args = parser.parse_args()

    out_p = Path(args.output)
    out_p.mkdir(parents=True, exist_ok=True)

    # Class name map for KIIT-MiTA
    source_name_map = {
        0: "Artilary",
        1: "Missile",
        2: "Radar",
        3: "M. Rocket Launcher",
        4: "Soldier",
        5: "Tank",
        6: "Vehicle"
    }
    if args.names_map and os.path.exists(args.names_map):
        with open(args.names_map, "r") as f:
            source_name_map = yaml.safe_load(f)

    records, stats = parse_yolo_dataset(
        Path(args.input),
        out_p,
        source_name_map,
        DEFAULT_TAXONOMY_MAP,
        args.name
    )

    manifest_path = out_p / f"manifest_{args.name}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"dataset": args.name, "stats": stats, "records": records}, f, indent=2)

    logger.info(f"Manifest written to: {manifest_path}")

if __name__ == "__main__":
    main()
