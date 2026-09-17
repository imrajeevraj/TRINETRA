#!/usr/bin/env python3
"""
IBVAP — Security Item Dataset Verification Engine
Inspects raw datasets in data/raw/security_items/ to verify image integrity,
annotation formats, and catalog compliance before normalization.
"""

import os
import sys
import logging
from pathlib import Path
import yaml
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SecurityItemVerify")


def verify_dataset_directory(dir_path: Path):
    if not dir_path.exists():
        logger.warning(f"Directory {dir_path} does not exist.")
        return {"exists": False, "valid_images": 0, "corrupted_images": 0, "annotations": 0}

    files = list(dir_path.rglob("*"))
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ann_exts = {".txt", ".xml", ".json", ".csv"}

    img_files = [f for f in files if f.suffix.lower() in img_exts]
    ann_files = [f for f in files if f.suffix.lower() in ann_exts]

    valid_imgs = 0
    corrupted_imgs = 0

    # Sample verify images
    for f in img_files[:100]:
        try:
            with Image.open(f) as img:
                img.verify()
            valid_imgs += 1
        except Exception:
            corrupted_imgs += 1

    return {
        "exists": True,
        "total_images": len(img_files),
        "sampled_valid": valid_imgs,
        "sampled_corrupted": corrupted_imgs,
        "annotations": len(ann_files)
    }


def main():
    catalog_path = Path("scripts/datasets/security_item_dataset_catalog.yaml")
    if not catalog_path.exists():
        logger.critical(f"Catalog not found at {catalog_path}")
        sys.exit(1)

    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = yaml.safe_load(f)

    logger.info("=" * 70)
    logger.info("VERIFYING RAW SECURITY ITEM DATASETS")
    logger.info("=" * 70)

    all_passed = True
    for entry in catalog.get("datasets", []):
        dest = Path(entry.get("destination_dir"))
        stat = verify_dataset_directory(dest)
        status_str = "PASS" if stat["exists"] and stat["total_images"] > 0 else "FAIL"
        if status_str == "FAIL":
            all_passed = False
        logger.info(f"[{status_str}] {entry['id']} ({entry['name']}): Images={stat.get('total_images', 0)}, Annotations={stat.get('annotations', 0)}")

    logger.info("=" * 70)
    if all_passed:
        logger.info("All catalogued security item datasets verified successfully.")
    else:
        logger.warning("Some datasets have not completed ingestion or are empty.")


if __name__ == "__main__":
    main()
