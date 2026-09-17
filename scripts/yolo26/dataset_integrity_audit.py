#!/usr/bin/env python3
"""
IBVAP — Phase 2: Dataset Integrity Audit
Checks all three IBVAP training datasets for:
  - Duplicate images (SHA-256 dedup)
  - Missing labels
  - Invalid bounding boxes
  - Class ID mismatches
  - Corrupt / zero-byte images
  - Cross-contamination with frozen benchmarks

Produces:
  data/reports/yolo26/dataset_integrity_report.md
  data/reports/yolo26/dataset_integrity.json
"""

import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DatasetIntegrityAudit")

# ── Dataset definitions ──────────────────────────────────────────────────────

DATASETS = {
    "ground": {
        "root": "data/normalized/ground_v2_exp002",
        "splits": ["train", "val"],
        "expected_classes": {0: "person", 1: "vehicle"},
        "max_class_id": 1,
    },
    "airborne": {
        "root": "data/normalized/airborne_v1_1",
        "splits": ["train", "val"],
        "expected_classes": {0: "drone", 1: "aircraft"},
        "max_class_id": 1,
    },
    "security_item": {
        "root": "data/normalized/security_item_v2_1",
        "splits": ["train", "val"],
        "expected_classes": {0: "firearm"},
        "max_class_id": 0,
    },
}

FROZEN_BENCHMARKS = {
    "IBVAP-GT-v1.0": [
        "benchmark/images/test",
    ],
    "IBVAP-GT-ITEM-v2.0": [
        "data/normalized/security_item_v2/images/test",
    ],
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def collect_images(split_dir: Path) -> list:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in split_dir.iterdir() if p.suffix.lower() in exts])


def audit_dataset(name: str, cfg: dict, benchmark_hashes: set) -> dict:
    logger.info("=" * 60)
    logger.info(f"Auditing dataset: {name}")
    root = Path(cfg["root"])
    max_cls = cfg["max_class_id"]

    result = {
        "dataset": name,
        "root": str(root),
        "splits": {},
        "duplicate_images": [],
        "benchmark_contamination": [],
        "total_images": 0,
        "total_labels": 0,
        "errors": [],
    }

    global_hashes: dict[str, str] = {}  # hash → path (cross-split dedup)

    for split in cfg["splits"]:
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split

        if not img_dir.exists():
            result["errors"].append(f"Image dir missing: {img_dir}")
            logger.warning(f"  [{split}] Image directory missing: {img_dir}")
            continue

        images = collect_images(img_dir)
        split_stats = {
            "images": len(images),
            "labels_found": 0,
            "labels_missing": 0,
            "corrupt_images": 0,
            "zero_byte_images": 0,
            "invalid_boxes": 0,
            "class_mismatches": 0,
            "empty_label_files": 0,
        }

        for img_p in images:
            result["total_images"] += 1

            # Zero-byte check
            if img_p.stat().st_size == 0:
                split_stats["zero_byte_images"] += 1
                result["errors"].append(f"Zero-byte image: {img_p}")
                continue

            # Corrupt image check
            try:
                img = cv2.imread(str(img_p))
                if img is None:
                    split_stats["corrupt_images"] += 1
                    result["errors"].append(f"Corrupt image (cv2 read failed): {img_p}")
                    continue
                h, w = img.shape[:2]
            except Exception as e:
                split_stats["corrupt_images"] += 1
                result["errors"].append(f"Corrupt image ({e}): {img_p}")
                continue

            # SHA-256 dedup (within dataset)
            img_hash = sha256_file(img_p)

            if img_hash in global_hashes:
                result["duplicate_images"].append({
                    "hash": img_hash,
                    "file_a": global_hashes[img_hash],
                    "file_b": str(img_p),
                })
            else:
                global_hashes[img_hash] = str(img_p)

            # Benchmark contamination check
            if img_hash in benchmark_hashes:
                result["benchmark_contamination"].append({
                    "file": str(img_p),
                    "hash": img_hash,
                    "split": split,
                })

            # Label check
            lbl_p = lbl_dir / f"{img_p.stem}.txt"
            if not lbl_p.exists():
                split_stats["labels_missing"] += 1
                result["errors"].append(f"Missing label: {lbl_p}")
                continue

            split_stats["labels_found"] += 1
            result["total_labels"] += 1

            # Parse labels
            lines = [l.strip() for l in lbl_p.read_text(encoding="utf-8").splitlines()
                     if l.strip() and not l.startswith("#")]
            if not lines:
                split_stats["empty_label_files"] += 1
                continue

            for ln in lines:
                parts = ln.split()
                if len(parts) < 5:
                    split_stats["invalid_boxes"] += 1
                    result["errors"].append(f"Short label line in {lbl_p}: '{ln}'")
                    continue

                try:
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                except ValueError:
                    split_stats["invalid_boxes"] += 1
                    result["errors"].append(f"Non-numeric label in {lbl_p}: '{ln}'")
                    continue

                # Class ID range
                if cls_id < 0 or cls_id > max_cls:
                    split_stats["class_mismatches"] += 1
                    result["errors"].append(
                        f"Invalid class_id={cls_id} (max={max_cls}) in {lbl_p}: '{ln}'"
                    )

                # Box sanity (YOLO normalized)
                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                    split_stats["invalid_boxes"] += 1
                    result["errors"].append(
                        f"Out-of-range bbox in {lbl_p}: cx={cx:.4f} cy={cy:.4f} w={bw:.4f} h={bh:.4f}"
                    )

        result["splits"][split] = split_stats
        logger.info(
            f"  [{split}] {split_stats['images']} images, "
            f"{split_stats['labels_found']} labels, "
            f"{split_stats['labels_missing']} missing, "
            f"{split_stats['invalid_boxes']} bad boxes, "
            f"{split_stats['corrupt_images']} corrupt"
        )

    result["duplicate_count"] = len(result["duplicate_images"])
    result["contamination_count"] = len(result["benchmark_contamination"])
    result["error_count"] = len(result["errors"])
    result["passed"] = (
        result["contamination_count"] == 0
        and result["duplicate_count"] == 0
        and result["error_count"] == 0
    )
    return result


def build_benchmark_hashes() -> set:
    logger.info("Hashing frozen benchmark images...")
    hashes = set()
    for bname, dirs in FROZEN_BENCHMARKS.items():
        for d in dirs:
            p = Path(d)
            if not p.exists():
                logger.warning(f"  Benchmark dir not found: {p}")
                continue
            imgs = [f for f in p.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            for img in imgs:
                if img.stat().st_size > 0:
                    hashes.add(sha256_file(img))
            logger.info(f"  {bname} ({d}): {len(imgs)} images hashed")
    return hashes


def main():
    out_dir = Path("data/reports/yolo26")
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark_hashes = build_benchmark_hashes()
    logger.info(f"Benchmark fingerprints loaded: {len(benchmark_hashes)} unique image hashes")

    all_results = {}
    overall_pass = True
    for name, cfg in DATASETS.items():
        r = audit_dataset(name, cfg, benchmark_hashes)
        all_results[name] = r
        if not r["passed"]:
            overall_pass = False

    # ── JSON output ──────────────────────────────────────────────────────────
    report = {
        "audit_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "overall_pass": overall_pass,
        "datasets": all_results,
    }
    json_path = out_dir / "dataset_integrity.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(f"JSON report: {json_path}")

    # ── Markdown report ───────────────────────────────────────────────────────
    lines = [
        "# IBVAP YOLO26 Migration — Dataset Integrity Report",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
        "",
        f"## Overall: {'✅ PASS' if overall_pass else '❌ FAIL — See errors below'}",
        "",
    ]
    for name, r in all_results.items():
        badge = "✅ PASS" if r["passed"] else "❌ FAIL"
        lines += [
            f"---",
            f"## {name.upper()} Dataset — {badge}",
            f"- Root: `{r['root']}`",
            f"- Total images: {r['total_images']}",
            f"- Total labels: {r['total_labels']}",
            f"- Duplicates: {r['duplicate_count']}",
            f"- Benchmark contamination: {r['contamination_count']}",
            f"- Errors: {r['error_count']}",
            "",
        ]
        for split, s in r.get("splits", {}).items():
            lines += [
                f"### Split: {split}",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Images | {s['images']} |",
                f"| Labels found | {s['labels_found']} |",
                f"| Labels missing | {s['labels_missing']} |",
                f"| Corrupt images | {s['corrupt_images']} |",
                f"| Zero-byte images | {s['zero_byte_images']} |",
                f"| Invalid boxes | {s['invalid_boxes']} |",
                f"| Class mismatches | {s['class_mismatches']} |",
                "",
            ]
        if r["benchmark_contamination"]:
            lines.append("### ⚠️ BENCHMARK CONTAMINATION DETECTED")
            for c in r["benchmark_contamination"]:
                lines.append(f"- `{c['file']}` (hash: `{c['hash'][:16]}...`) in split `{c['split']}`")
            lines.append("")
        if r["errors"]:
            lines.append(f"### Errors ({min(len(r['errors']), 20)} shown of {len(r['errors'])})")
            for e in r["errors"][:20]:
                lines.append(f"- {e}")
            lines.append("")

    md_path = out_dir / "dataset_integrity_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Markdown report: {md_path}")

    if not overall_pass:
        logger.critical("DATASET INTEGRITY AUDIT FAILED. DO NOT PROCEED WITH TRAINING.")
        sys.exit(1)
    else:
        logger.info("Dataset integrity audit PASSED. Safe to proceed with YOLO26 training.")


if __name__ == "__main__":
    main()
