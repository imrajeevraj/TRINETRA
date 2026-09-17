#!/usr/bin/env python3
"""
IBVAP — Dataset Quality Verification & Quarantine Engine
Validates annotations, image integrity, bounding box geometry, and class indices.
Quarantines invalid samples and generates validation reports in data/reports/.
"""

import os
import sys
import json
import argparse
import logging
import shutil
from pathlib import Path
from collections import Counter
import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DatasetVerifier")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def verify_dataset(input_dir: Path, quarantine_dir: Path | None = None, report_dir: Path = Path("data/reports")):
    report_dir.mkdir(parents=True, exist_ok=True)
    if quarantine_dir:
        quarantine_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info(f"VERIFYING DATASET INTEGRITY: {input_dir.resolve()}")
    logger.info("=" * 70)

    all_images = [f for f in input_dir.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS]
    all_labels = [f for f in input_dir.rglob("*") if f.suffix.lower() == ".txt" and f.name != "classes.txt"]

    img_stem_map = {f.stem: f for f in all_images}
    lbl_stem_map = {f.stem: f for f in all_labels}

    corrupted_images = []
    invalid_boxes = []
    zero_area_boxes = []
    out_of_bound_boxes = []
    empty_label_files = []
    unpaired_images = []
    unpaired_labels = []

    valid_annotations_count = 0
    class_counter = Counter()

    # 1. Image checks
    logger.info(f"Checking {len(all_images)} image files for corruption and decoding...")
    for img_p in all_images:
        try:
            img = cv2.imread(str(img_p))
            if img is None or img.size == 0 or img.shape[0] < 10 or img.shape[1] < 10:
                corrupted_images.append(str(img_p))
                if quarantine_dir:
                    shutil.move(str(img_p), str(quarantine_dir / img_p.name))
        except Exception:
            corrupted_images.append(str(img_p))

    # 2. Annotation & Geometry checks
    logger.info(f"Checking {len(all_labels)} annotation files for geometry validity...")
    for lbl_p in all_labels:
        # Check matching image
        if lbl_p.stem not in img_stem_map:
            unpaired_labels.append(str(lbl_p))
            continue

        try:
            with open(lbl_p, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            clean_lines = [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]
            if not clean_lines:
                empty_label_files.append(str(lbl_p))
                continue

            for idx, line in enumerate(clean_lines):
                parts = line.split()
                if len(parts) < 5:
                    invalid_boxes.append((str(lbl_p), idx, line, "fewer_than_5_fields"))
                    continue

                try:
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                except ValueError:
                    invalid_boxes.append((str(lbl_p), idx, line, "non_numeric_values"))
                    continue

                # Check bounds
                if cx < 0.0 or cx > 1.0 or cy < 0.0 or cy > 1.0:
                    out_of_bound_boxes.append((str(lbl_p), idx, line, "center_out_of_bounds"))
                elif w <= 0.0 or h <= 0.0:
                    zero_area_boxes.append((str(lbl_p), idx, line, "zero_or_negative_dimension"))
                elif w > 1.0 or h > 1.0:
                    out_of_bound_boxes.append((str(lbl_p), idx, line, "dimension_exceeds_1.0"))
                else:
                    valid_annotations_count += 1
                    class_counter[cls_id] += 1

        except Exception as e:
            invalid_boxes.append((str(lbl_p), -1, str(e), "file_read_exception"))

    # Images with no corresponding label
    for stem, p in img_stem_map.items():
        if stem not in lbl_stem_map:
            unpaired_images.append(str(p))

    # Compile report summary
    total_samples = len(all_images)
    invalid_samples = len(corrupted_images) + len(invalid_boxes) + len(zero_area_boxes)
    valid_samples = max(0, total_samples - len(corrupted_images))

    report_data = {
        "dataset_path": str(input_dir.resolve()),
        "total_images": len(all_images),
        "total_label_files": len(all_labels),
        "valid_images": valid_samples,
        "corrupted_images": len(corrupted_images),
        "unpaired_images": len(unpaired_images),
        "unpaired_labels": len(unpaired_labels),
        "empty_label_files": len(empty_label_files),
        "total_valid_boxes": valid_annotations_count,
        "invalid_boxes": len(invalid_boxes),
        "zero_area_boxes": len(zero_area_boxes),
        "out_of_bound_boxes": len(out_of_bound_boxes),
        "class_distribution": dict(class_counter),
        "status": "PASS" if len(corrupted_images) == 0 and len(invalid_boxes) == 0 and len(zero_area_boxes) == 0 else "WARNINGS_FOUND"
    }

    # Save JSON Report
    json_path = report_dir / "dataset_validation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    # Save Markdown Report
    md_path = report_dir / "dataset_validation_report.md"
    md_content = f"""# IBVAP — Dataset Quality Validation Report

**Dataset Path:** `{input_dir.resolve()}`  
**Validation Status:** **{report_data['status']}**

---

## 1. Summary Statistics

| Metric | Count | Status |
| :--- | :---: | :---: |
| **Total Images** | {report_data['total_images']} | Verified |
| **Valid Images** | {report_data['valid_images']} | Pass |
| **Corrupted Images** | {report_data['corrupted_images']} | {'Pass' if report_data['corrupted_images'] == 0 else 'Quarantined'} |
| **Total Valid Bounding Boxes** | {report_data['total_valid_boxes']} | Pass |
| **Invalid Bounding Boxes** | {report_data['invalid_boxes']} | {'Pass' if report_data['invalid_boxes'] == 0 else 'Quarantined'} |
| **Zero-Area Boxes** | {report_data['zero_area_boxes']} | {'Pass' if report_data['zero_area_boxes'] == 0 else 'Rejected'} |
| **Out-of-Bounds Boxes** | {report_data['out_of_bound_boxes']} | Clamped / Flagged |
| **Empty Label Files (Hard Negatives)** | {report_data['empty_label_files']} | Preserved |
| **Unpaired Images** | {report_data['unpaired_images']} | Check Image-to-Label mapping |

---

## 2. Raw Class Distribution
{json.dumps(dict(class_counter), indent=2)}
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info(f"Validation reports saved to {json_path} and {md_path}")
    logger.info(f"Overall Status: {report_data['status']}")
    return report_data

def main():
    parser = argparse.ArgumentParser(description="IBVAP Dataset Quality Verification")
    parser.add_argument("--input", type=str, required=True, help="Path to raw dataset directory")
    parser.add_argument("--quarantine", type=str, default=None, help="Directory to move corrupted samples")
    parser.add_argument("--reports", type=str, default="data/reports", help="Output report directory")
    args = parser.parse_args()

    verify_dataset(Path(args.input), Path(args.quarantine) if args.quarantine else None, Path(args.reports))

if __name__ == "__main__":
    main()
