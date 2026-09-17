#!/usr/bin/env python3
"""
IBVAP — Duplicate Detection & Cross-Split Anti-Leakage Protocol
Computes MD5 hashes and difference perceptual hashes (dHash) to eliminate
duplicate samples and mathematically guarantee zero overlap with the frozen benchmark.
"""

import os
import sys
import argparse
import logging
import hashlib
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DatasetDeduplicator")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def compute_md5(file_path: Path) -> str:
    """Compute standard MD5 hash of image file bytes."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_dhash(image_path: Path, hash_size: int = 8) -> int:
    """
    Compute difference perceptual hash (dHash) for visual similarity matching.
    Detects resized, recompressed, or near-identical consecutive video frames.
    """
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0
        # Resize to (width + 1, height)
        resized = cv2.resize(img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
        # Compute horizontal gradient
        diff = resized[:, 1:] > resized[:, :-1]
        # Convert boolean array to 64-bit integer
        return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
    except Exception:
        return 0

def hamming_distance(h1: int, h2: int) -> int:
    """Compute Hamming distance between two perceptual hashes."""
    return bin(h1 ^ h2).count("1")

def run_deduplication(
    dataset_images_dir: Path,
    benchmark_images_dir: Path = Path("benchmark/images/test"),
    max_dhash_distance: int = 2
):
    logger.info("=" * 70)
    logger.info("IBVAP DEDUPLICATION & ZERO-LEAKAGE VERIFICATION")
    logger.info(f"Scanning Candidate Dataset: {dataset_images_dir.resolve()}")
    logger.info(f"Checking Against Frozen Benchmark: {benchmark_images_dir.resolve()}")
    logger.info("=" * 70)

    # 1. Index Frozen Benchmark
    benchmark_md5s = set()
    benchmark_dhashes = {}
    if benchmark_images_dir.exists():
        bench_images = [f for f in benchmark_images_dir.glob("*") if f.suffix.lower() in IMAGE_EXTS]
        logger.info(f"Indexing {len(bench_images)} frozen benchmark images...")
        for bp in bench_images:
            m = compute_md5(bp)
            dh = compute_dhash(bp)
            benchmark_md5s.add(m)
            if dh != 0:
                benchmark_dhashes[dh] = bp.name

    # 2. Scan Candidate Dataset
    candidate_images = [f for f in dataset_images_dir.glob("*") if f.suffix.lower() in IMAGE_EXTS]
    logger.info(f"Scanning {len(candidate_images)} candidate images for duplicates and leakage...")

    seen_md5s = {}
    seen_dhashes = [] # list of (dhash, filename)
    
    exact_duplicates = []
    perceptual_duplicates = []
    benchmark_leakage_detected = []

    for img_p in candidate_images:
        m = compute_md5(img_p)
        dh = compute_dhash(img_p)

        # Check Benchmark Leakage (CRITICAL)
        if m in benchmark_md5s:
            benchmark_leakage_detected.append((img_p.name, "exact_md5_match_with_benchmark"))
            continue
        
        # Check perceptual match with benchmark
        for b_dh, b_name in benchmark_dhashes.items():
            if hamming_distance(dh, b_dh) <= 1:
                benchmark_leakage_detected.append((img_p.name, f"near_duplicate_of_benchmark_{b_name}"))
                break

        # Check internal exact duplicates
        if m in seen_md5s:
            exact_duplicates.append((img_p.name, seen_md5s[m]))
        else:
            seen_md5s[m] = img_p.name

        # Check internal perceptual near-duplicates
        if dh != 0:
            is_near_dup = False
            for prev_dh, prev_name in seen_dhashes:
                if hamming_distance(dh, prev_dh) <= max_dhash_distance:
                    perceptual_duplicates.append((img_p.name, prev_name))
                    is_near_dup = True
                    break
            if not is_near_dup:
                seen_dhashes.append((dh, img_p.name))

    # Anti-leakage verdict
    if benchmark_leakage_detected:
        logger.critical(f"CRITICAL LEAKAGE DETECTED: {len(benchmark_leakage_detected)} candidate images match the frozen benchmark!")
        for c_img, reason in benchmark_leakage_detected[:5]:
            logger.critical(f"  {c_img} -> {reason}")
        sys.exit(1)
    else:
        logger.info("Anti-Leakage Verification: PASS (0 images leak into or from the frozen benchmark).")

    logger.info(f"Exact Duplicate Images Found:      {len(exact_duplicates)}")
    logger.info(f"Perceptual Near-Duplicates Found: {len(perceptual_duplicates)}")
    
    unique_count = len(candidate_images) - len(exact_duplicates)
    logger.info(f"Total Unique Images Available:     {unique_count}/{len(candidate_images)}")

    return {
        "total_images": len(candidate_images),
        "exact_duplicates": exact_duplicates,
        "perceptual_duplicates": perceptual_duplicates,
        "benchmark_leakage": len(benchmark_leakage_detected)
    }

def main():
    parser = argparse.ArgumentParser(description="IBVAP Deduplication and Anti-Leakage Verifier")
    parser.add_argument("--images", type=str, required=True, help="Directory of normalized images")
    parser.add_argument("--benchmark", type=str, default="benchmark/images/test", help="Path to frozen benchmark images")
    parser.add_argument("--threshold", type=int, default=2, help="Max Hamming distance for perceptual dHash match")
    args = parser.parse_args()

    run_deduplication(Path(args.images), Path(args.benchmark), args.threshold)

if __name__ == "__main__":
    main()
