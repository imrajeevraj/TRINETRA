#!/usr/bin/env python3
"""
IBVAP — Production Kaggle Dataset Ingestion Engine
Authenticates with Kaggle, downloads catalogued datasets, extracts archives,
and registers raw artifacts in data/external/kaggle/.
"""

import os
import sys
import argparse
import logging
import zipfile
from pathlib import Path
import yaml
from dotenv import load_dotenv

# Configure structured logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("KaggleIngest")

def authenticate_kaggle():
    """
    Authenticate Kaggle API using environment variables (.env) or ~/.kaggle/kaggle.json.
    Never exposes raw secrets or writes credentials to disk.
    """
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)

    # Inject into environment if present in .env
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    token = os.environ.get("KAGGLE_API_TOKEN")

    if not username and not key and not token:
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if not kaggle_json.exists():
            logger.critical("KAGGLE AUTHENTICATION FAILED: No credentials detected in .env or ~/.kaggle/kaggle.json")
            logger.critical("Set KAGGLE_USERNAME and KAGGLE_KEY in .env before running ingestion.")
            sys.exit(1)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        logger.info(f"Kaggle API successfully authenticated for user: {os.getenv('KAGGLE_USERNAME', 'default')}")
        return api
    except Exception as e:
        logger.critical(f"Kaggle API authentication error: {e}")
        sys.exit(1)

def ingest_dataset(api, dataset_slug: str, destination_dir: Path, force: bool = False):
    """
    Download and extract a Kaggle dataset into a dedicated target directory.
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    complete_marker = destination_dir / ".ingest_complete"

    if complete_marker.exists() and not force:
        logger.info(f"Dataset '{dataset_slug}' already ingested at {destination_dir}. Use --force to re-download.")
        return True

    logger.info("=" * 70)
    logger.info(f"STARTING KAGGLE DOWNLOAD: {dataset_slug}")
    logger.info(f"Target Directory: {destination_dir.resolve()}")
    logger.info("=" * 70)

    try:
        api.dataset_download_files(dataset_slug, path=str(destination_dir), unzip=True, force=force)
        
        # Verify extraction
        files = list(destination_dir.rglob("*"))
        images = [f for f in files if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]]
        annotations = [f for f in files if f.suffix.lower() in [".txt", ".json", ".xml", ".csv"]]

        if not images:
            logger.warning(f"Downloaded archive for '{dataset_slug}' contains 0 recognizable image files!")
        else:
            logger.info(f"Extracted {len(images)} images and {len(annotations)} annotation files.")

        complete_marker.write_text(f"dataset: {dataset_slug}\nfiles_extracted: {len(files)}\n", encoding="utf-8")
        logger.info(f"Ingestion successful for '{dataset_slug}'.")
        return True

    except Exception as e:
        logger.critical(f"Failed to download or extract Kaggle dataset '{dataset_slug}': {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="IBVAP Kaggle Dataset Ingestion Engine")
    parser.add_argument("--config", type=str, default="scripts/datasets/dataset_catalog.yaml", help="Path to catalog YAML")
    parser.add_argument("--dataset", type=str, default=None, help="Download a specific dataset slug (owner/dataset)")
    parser.add_argument("--download-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--force", action="store_true", help="Force re-download even if already present")
    args = parser.parse_args()

    api = authenticate_kaggle()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.critical(f"Dataset catalog not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        catalog = yaml.safe_load(f)

    target_entries = catalog.get("datasets", [])
    if args.dataset:
        target_entries = [d for d in target_entries if f"{d.get('owner')}/{d.get('dataset')}" == args.dataset or d.get("id") == args.dataset]
        if not target_entries:
            owner, slug = args.dataset.split("/") if "/" in args.dataset else ("unknown", args.dataset)
            target_entries = [{
                "id": args.dataset.replace("/", "_"),
                "owner": owner,
                "dataset": slug,
                "enabled": True,
                "destination_dir": args.download_dir or f"data/external/kaggle/{slug}"
            }]

    success = 0
    total = 0

    for entry in target_entries:
        if not entry.get("enabled", True) and not args.dataset:
            logger.info(f"Skipping disabled catalog entry: {entry.get('id')}")
            continue

        total += 1
        slug = f"{entry.get('owner')}/{entry.get('dataset')}"
        dest = Path(args.download_dir or entry.get("destination_dir", f"data/external/kaggle/{entry.get('dataset')}"))

        if ingest_dataset(api, slug, dest, force=args.force):
            success += 1

    logger.info("=" * 70)
    logger.info(f"INGESTION COMPLETE: {success}/{total} datasets ingested successfully.")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
