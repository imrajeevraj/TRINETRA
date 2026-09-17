#!/usr/bin/env python3
"""
IBVAP — Airborne Dataset Ingestion Engine
Authenticates with Kaggle, downloads catalogued airborne datasets (drones, aircraft,
and bird hard negatives), and extracts them to data/raw/airborne/.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import yaml
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AirborneIngest")

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

def ingest_airborne(api, dataset_slug: str, destination_dir: Path, force: bool = False):
    destination_dir.mkdir(parents=True, exist_ok=True)
    complete_marker = destination_dir / ".ingest_complete"

    if complete_marker.exists() and not force:
        logger.info(f"Dataset '{dataset_slug}' already present at {destination_dir}. Use --force to re-download.")
        return True

    logger.info("=" * 70)
    logger.info(f"DOWNLOADING AIRBORNE DATASET: {dataset_slug}")
    logger.info(f"Destination: {destination_dir.resolve()}")
    logger.info("=" * 70)

    try:
        api.dataset_download_files(dataset_slug, path=str(destination_dir), unzip=True, force=force)
        files = list(destination_dir.rglob("*"))
        images = [f for f in files if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]]
        annotations = [f for f in files if f.suffix.lower() in [".txt", ".json", ".xml"]]

        complete_marker.write_text(f"dataset: {dataset_slug}\nimages: {len(images)}\nannotations: {len(annotations)}\n", encoding="utf-8")
        logger.info(f"Extraction complete for '{dataset_slug}': {len(images)} images, {len(annotations)} annotations.")
        return True
    except Exception as e:
        logger.critical(f"Failed to ingest airborne dataset '{dataset_slug}': {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Ingest Airborne Datasets")
    parser.add_argument("--config", type=str, default="scripts/datasets/airborne_dataset_catalog.yaml", help="Catalog config path")
    parser.add_argument("--dataset", type=str, default=None, help="Download a specific dataset slug")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()

    api = authenticate_kaggle()

    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        catalog = yaml.safe_load(f)

    entries = catalog.get("datasets", [])
    if args.dataset:
        entries = [e for e in entries if f"{e.get('owner')}/{e.get('dataset')}" == args.dataset or e.get("id") == args.dataset]

    success = 0
    for entry in entries:
        if not entry.get("enabled", True) and not args.dataset:
            continue
        slug = f"{entry.get('owner')}/{entry.get('dataset')}"
        dest = Path(entry.get("destination_dir"))
        if ingest_airborne(api, slug, dest, force=args.force):
            success += 1

    logger.info("=" * 70)
    logger.info(f"AIRBORNE INGESTION COMPLETE: {success}/{len(entries)} datasets ingested.")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
