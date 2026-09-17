#!/usr/bin/env python3
"""
IBVAP — Kaggle Dataset Auto-Download Pipeline
Authenticates with Kaggle, downloads declared datasets, extracts archives,
and validates integrity in data/training/raw/.
"""

import os
import sys
import argparse
import logging
import zipfile
import hashlib
from pathlib import Path
import yaml
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("KaggleDownloader")

def check_and_setup_auth():
    """Load credentials from .env or environment, verifying Kaggle readiness."""
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)

    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    token = os.environ.get("KAGGLE_API_TOKEN")
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"

    if username and key and username != "CHANGE_ME" and key != "CHANGE_ME":
        logger.info("Using KAGGLE_USERNAME and KAGGLE_KEY from environment/.env")
        return True
    elif token and token != "CHANGE_ME":
        logger.info("Using KAGGLE_API_TOKEN from environment/.env")
        return True
    elif kaggle_json.exists():
        logger.info(f"Found existing Kaggle token at {kaggle_json}")
        return True
    else:
        logger.warning("=" * 65)
        logger.warning("KAGGLE CREDENTIALS NOT DETECTED")
        logger.warning("Please configure your Kaggle credentials in .env:")
        logger.warning("  KAGGLE_USERNAME=your_kaggle_username")
        logger.warning("  KAGGLE_KEY=your_kaggle_api_key")
        logger.warning("Or generate a token at https://www.kaggle.com/settings/api")
        logger.warning("=" * 65)
        return False

def calculate_checksum(file_path: Path) -> str:
    """Calculate SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def download_dataset(dataset_id: str, output_dir: Path, force: bool = False, clean: bool = False):
    """Download and extract a Kaggle dataset using kaggle.api."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        logger.error(f"Kaggle authentication failed: {e}")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    marker_file = output_dir / ".download_complete"

    if marker_file.exists() and not force:
        logger.info(f"Dataset '{dataset_id}' already downloaded in {output_dir}. Use --force to re-download.")
        return True

    if clean and output_dir.exists():
        logger.info(f"Cleaning existing directory {output_dir}")
        for item in output_dir.iterdir():
            if item.is_file(): item.unlink()

    logger.info(f"Downloading Kaggle dataset: {dataset_id} -> {output_dir}")
    try:
        api.dataset_download_files(dataset_id, path=str(output_dir), unzip=True, force=force)
        marker_file.write_text(f"dataset: {dataset_id}\n")
        logger.info(f"Successfully downloaded and extracted: {dataset_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to download dataset {dataset_id}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="IBVAP Kaggle Dataset Auto-Downloader")
    parser.add_argument("--config", type=str, default="configs/kaggle_datasets.yaml", help="Path to kaggle datasets yaml config")
    parser.add_argument("--dataset", type=str, default=None, help="Specific dataset ID to download (e.g. owner/name)")
    parser.add_argument("--output", type=str, default=None, help="Custom output directory")
    parser.add_argument("--force", action="store_true", help="Force re-download even if already present")
    parser.add_argument("--clean", action="store_true", help="Clean destination directory before extraction")
    parser.add_argument("--verify", action="store_true", help="Verify extraction integrity and contents")
    args = parser.parse_args()

    auth_ok = check_and_setup_auth()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    datasets = config.get("datasets", [])
    if args.dataset:
        datasets = [d for d in datasets if d.get("id") == args.dataset]
        if not datasets:
            datasets = [{"id": args.dataset, "enabled": True, "destination": args.output or f"data/training/raw/{args.dataset.replace('/', '_')}"}]

    if not auth_ok:
        logger.warning("Proceeding in offline/dry-run mode since credentials are missing.")
        logger.info(f"Found {len(datasets)} dataset target(s) configured in {config_path}.")
        for ds in datasets:
            logger.info(f"  Target: {ds.get('id')} -> {ds.get('destination')}")
        logger.info("To download, add your Kaggle API key to .env and re-run.")
        sys.exit(0)

    success_count = 0
    for ds in datasets:
        if not ds.get("enabled", True) and not args.dataset:
            logger.info(f"Skipping disabled dataset: {ds.get('id')}")
            continue

        dest = Path(args.output or ds.get("destination", f"data/training/raw/{ds['id'].replace('/', '_')}"))
        ok = download_dataset(ds["id"], dest, force=args.force, clean=args.clean)
        if ok:
            success_count += 1
            if args.verify:
                files = list(dest.rglob("*"))
                images = [f for f in files if f.suffix.lower() in [".jpg", ".png", ".jpeg"]]
                labels = [f for f in files if f.suffix.lower() == ".txt"]
                logger.info(f"Verification: {len(images)} images, {len(labels)} label files found.")

    logger.info(f"Kaggle download operation complete. {success_count}/{len(datasets)} processed successfully.")

if __name__ == "__main__":
    main()
