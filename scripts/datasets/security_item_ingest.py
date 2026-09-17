#!/usr/bin/env python3
"""
IBVAP — Security Item Dataset Ingestion Engine
Downloads and extracts firearm datasets from Kaggle and organizes local curated
handheld negative datasets in data/raw/security_items/.
"""

import os
import sys
import shutil
import argparse
import logging
from pathlib import Path
import yaml
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SecurityItemIngest")


def authenticate_kaggle():
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        logger.info(f"Kaggle API successfully authenticated for user: {os.getenv('KAGGLE_USERNAME', 'default')}")
        return api
    except Exception as e:
        logger.critical(f"Kaggle API authentication error: {e}")
        sys.exit(1)


def ingest_kaggle_dataset(api, dataset_slug: str, destination_dir: Path, force: bool = False):
    destination_dir.mkdir(parents=True, exist_ok=True)
    complete_marker = destination_dir / ".ingest_complete"

    if complete_marker.exists() and not force:
        logger.info(f"Dataset '{dataset_slug}' already present at {destination_dir}. Skipping download.")
        return True

    logger.info("=" * 70)
    logger.info(f"DOWNLOADING SECURITY ITEM DATASET: {dataset_slug}")
    logger.info(f"Destination: {destination_dir.resolve()}")
    logger.info("=" * 70)

    try:
        api.dataset_download_files(dataset_slug, path=str(destination_dir), unzip=True, force=force)
        files = list(destination_dir.rglob("*"))
        images = [f for f in files if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]]
        annotations = [f for f in files if f.suffix.lower() in [".txt", ".json", ".xml", ".csv"]]

        complete_marker.write_text(f"dataset: {dataset_slug}\nimages: {len(images)}\nannotations: {len(annotations)}\n", encoding="utf-8")
        logger.info(f"Extraction complete for '{dataset_slug}': {len(images)} images, {len(annotations)} annotations.")
        return True
    except Exception as e:
        logger.critical(f"Failed to ingest dataset '{dataset_slug}': {e}")
        return False


def ingest_local_negatives(destination_dir: Path, force: bool = False):
    destination_dir.mkdir(parents=True, exist_ok=True)
    complete_marker = destination_dir / ".ingest_complete"

    if complete_marker.exists() and not force:
        logger.info(f"Local negatives already present at {destination_dir}. Skipping copy.")
        return True

    logger.info("Curating local handheld negatives from surveillance imagery...")
    source_dir = Path("data/normalized/ground_v2/images")
    if not source_dir.exists():
        source_dir = Path("data/raw/ground")

    copied = 0
    candidate_images = list(source_dir.rglob("*.jpg")) + list(source_dir.rglob("*.jpeg")) + list(source_dir.rglob("*.png"))
    for img_path in candidate_images[:350]:
        dest_path = destination_dir / f"neg_{copied:04d}_{img_path.name}"
        shutil.copy2(img_path, dest_path)
        # Create empty label file indicating pure negative
        label_dest = destination_dir / f"neg_{copied:04d}_{img_path.stem}.txt"
        label_dest.write_text("", encoding="utf-8")
        copied += 1

    complete_marker.write_text(f"dataset: local_surveillance_negatives\nimages: {copied}\nannotations: {copied}\n", encoding="utf-8")
    logger.info(f"Extracted {copied} local handheld surveillance negative images to {destination_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Ingest Security Item Datasets")
    parser.add_argument("--config", type=str, default="scripts/datasets/security_item_dataset_catalog.yaml", help="Catalog config path")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        catalog = yaml.safe_load(f)

    api = None
    entries = catalog.get("datasets", [])

    for entry in entries:
        if not entry.get("enabled", True):
            continue

        src = entry.get("source")
        dest_dir = Path(entry.get("destination_dir"))

        if src == "kaggle":
            if api is None:
                api = authenticate_kaggle()
            slug = f"{entry.get('owner')}/{entry.get('dataset')}"
            ingest_kaggle_dataset(api, slug, dest_dir, force=args.force)
        elif src == "local_ingest":
            ingest_local_negatives(dest_dir, force=args.force)


if __name__ == "__main__":
    main()
