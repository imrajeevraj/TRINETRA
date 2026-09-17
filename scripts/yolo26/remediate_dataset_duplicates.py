#!/usr/bin/env python3
"""
IBVAP — Dataset Deduplication & Leakage Remediation
Removes duplicate images from training splits to eliminate train/val leakage.

Strategy per IBVAP data policy:
  - Train/val leakage: REMOVE from training split (keep validation copy clean)
  - Within-train duplicates: REMOVE the later-named copy (keep earlier alphabetically)

Operates non-destructively:
  - Moved files go to data/quarantine/dataset_name/split/ (not deleted)
  - Labels are moved alongside images
  - A manifest is written listing every move

Run dry-run first:
  python scripts/yolo26/remediate_dataset_duplicates.py --dry-run

Then apply:
  python scripts/yolo26/remediate_dataset_duplicates.py
"""

import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DatasetRemediation")

DATASETS = {
    "ground": "data/normalized/ground_v2_exp002",
    "airborne": "data/normalized/airborne_v1_1",
    "security_item": "data/normalized/security_item_v2_1",
}
QUARANTINE_BASE = "data/quarantine"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def move_pair(img_p: Path, lbl_dir: Path, quarantine_img_dir: Path,
              quarantine_lbl_dir: Path, dry_run: bool, reason: str) -> dict:
    """Move image + label to quarantine."""
    lbl_p = lbl_dir / f"{img_p.stem}.txt"
    action = {"image": str(img_p), "label": str(lbl_p) if lbl_p.exists() else None, "reason": reason}

    if not dry_run:
        quarantine_img_dir.mkdir(parents=True, exist_ok=True)
        quarantine_lbl_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(img_p), str(quarantine_img_dir / img_p.name))
        if lbl_p.exists():
            shutil.move(str(lbl_p), str(quarantine_lbl_dir / lbl_p.name))

    prefix = "[DRY-RUN] " if dry_run else ""
    logger.info(f"{prefix}Quarantine ({reason}): {img_p.name}")
    return action


def remediate_dataset(name: str, root: str, audit_data: dict, dry_run: bool) -> dict:
    logger.info("=" * 60)
    logger.info(f"Remediating dataset: {name.upper()}")

    root_p = Path(root)
    quarantine_base = Path(QUARANTINE_BASE) / name
    moves = []

    dups = audit_data.get("duplicate_images", [])

    for dup in dups:
        a_path = Path(dup["file_a"])
        b_path = Path(dup["file_b"])

        a_in_train = "train" in str(a_path)
        b_in_train = "train" in str(b_path)
        a_in_val = "val" in str(a_path)
        b_in_val = "val" in str(b_path)

        if a_in_train and b_in_val:
            # Train→Val leakage: remove from training
            train_img = a_path
            lbl_dir = train_img.parent.parent.parent / "labels" / "train"
            q_img = quarantine_base / "train" / "images"
            q_lbl = quarantine_base / "train" / "labels"
            if train_img.exists():
                m = move_pair(train_img, lbl_dir, q_img, q_lbl, dry_run,
                              reason=f"train_val_leakage (matches val/{b_path.name})")
                moves.append(m)

        elif b_in_train and a_in_val:
            # Train→Val leakage: remove from training
            train_img = b_path
            lbl_dir = train_img.parent.parent.parent / "labels" / "train"
            q_img = quarantine_base / "train" / "images"
            q_lbl = quarantine_base / "train" / "labels"
            if train_img.exists():
                m = move_pair(train_img, lbl_dir, q_img, q_lbl, dry_run,
                              reason=f"train_val_leakage (matches val/{a_path.name})")
                moves.append(m)

        elif a_in_train and b_in_train:
            # Within-train duplicate: remove alphabetically later
            keep, remove = sorted([a_path, b_path], key=lambda x: x.name)
            lbl_dir = remove.parent.parent.parent / "labels" / "train"
            q_img = quarantine_base / "train_dups" / "images"
            q_lbl = quarantine_base / "train_dups" / "labels"
            if remove.exists():
                m = move_pair(remove, lbl_dir, q_img, q_lbl, dry_run,
                              reason=f"within_train_duplicate (kept {keep.name})")
                moves.append(m)

    logger.info(f"  {name}: {len(moves)} files quarantined {'(DRY RUN)' if dry_run else ''}")
    return {"dataset": name, "moves": moves, "count": len(moves)}


def main():
    parser = argparse.ArgumentParser(description="IBVAP Dataset Deduplication")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, no moves")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()) + ["all"], default="all")
    args = parser.parse_args()

    audit_json = Path("data/reports/yolo26/dataset_integrity.json")
    if not audit_json.exists():
        logger.critical("Run dataset_integrity_audit.py first.")
        sys.exit(1)

    with open(audit_json, encoding="utf-8") as f:
        audit = json.load(f)

    datasets = {args.dataset: DATASETS[args.dataset]} if args.dataset != "all" else DATASETS
    all_moves = {}
    for name, root in datasets.items():
        audit_data = audit.get("datasets", {}).get(name, {})
        r = remediate_dataset(name, root, audit_data, dry_run=args.dry_run)
        all_moves[name] = r

    manifest = {
        "remediation_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dry_run": args.dry_run,
        "summary": {n: r["count"] for n, r in all_moves.items()},
        "details": all_moves,
    }

    manifest_path = Path("data/reports/yolo26/remediation_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(f"Manifest: {manifest_path}")

    total = sum(r["count"] for r in all_moves.values())
    if args.dry_run:
        logger.info(f"DRY RUN complete. {total} files would be quarantined.")
        logger.info("Run without --dry-run to apply remediation.")
    else:
        logger.info(f"Remediation complete. {total} files quarantined.")
        logger.info("Re-run dataset_integrity_audit.py to confirm PASS.")


if __name__ == "__main__":
    main()
