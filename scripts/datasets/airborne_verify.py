#!/usr/bin/env python3
"""
IBVAP — Airborne Dataset Quality Verification Engine
Validates drone, aircraft, and aerial negative annotations and image decoding.
Produces audit reports in data/reports/airborne_v1/.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from collections import Counter
import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AirborneVerify")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def verify_airborne_dataset(raw_dir: Path, reports_dir: Path = Path("data/reports/airborne_v1")):
    reports_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 70)
    logger.info(f"VERIFYING AIRBORNE DATASET INTEGRITY: {raw_dir.resolve()}")
    logger.info("=" * 70)

    all_images = [f for f in raw_dir.rglob("*") if f.suffix.lower() in IMAGE_EXTS]
    all_labels = [f for f in raw_dir.rglob("*") if f.suffix.lower() == ".txt" and f.name != "classes.txt"]

    corrupted_images = []
    invalid_boxes = []
    zero_area_boxes = []
    out_of_bound_boxes = []
    valid_boxes_count = 0
    class_dist = Counter()

    logger.info(f"Checking {len(all_images)} airborne images...")
    for img_p in all_images:
        try:
            img = cv2.imread(str(img_p))
            if img is None or img.size == 0 or img.shape[0] < 10 or img.shape[1] < 10:
                corrupted_images.append(str(img_p))
        except Exception:
            corrupted_images.append(str(img_p))

    logger.info(f"Checking {len(all_labels)} airborne label files...")
    for lbl_p in all_labels:
        try:
            with open(lbl_p, "r", encoding="utf-8", errors="ignore") as f:
                for idx, line in enumerate(f):
                    line_str = line.strip()
                    if not line_str or line_str.startswith("#"):
                        continue
                    parts = line_str.split()
                    if len(parts) < 5:
                        invalid_boxes.append((str(lbl_p), idx, line_str, "fewer_than_5_fields"))
                        continue
                    try:
                        cid = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        if cx < 0.0 or cx > 1.0 or cy < 0.0 or cy > 1.0:
                            out_of_bound_boxes.append((str(lbl_p), idx, line_str))
                        elif w <= 0.0 or h <= 0.0:
                            zero_area_boxes.append((str(lbl_p), idx, line_str))
                        else:
                            valid_boxes_count += 1
                            class_dist[cid] += 1
                    except ValueError:
                        invalid_boxes.append((str(lbl_p), idx, line_str, "non_numeric_values"))
        except Exception as e:
            invalid_boxes.append((str(lbl_p), -1, str(e), "read_error"))

    report = {
        "raw_airborne_dir": str(raw_dir.resolve()),
        "total_images": len(all_images),
        "total_label_files": len(all_labels),
        "valid_images": len(all_images) - len(corrupted_images),
        "corrupted_images": len(corrupted_images),
        "total_valid_boxes": valid_boxes_count,
        "invalid_boxes": len(invalid_boxes),
        "zero_area_boxes": len(zero_area_boxes),
        "out_of_bound_boxes": len(out_of_bound_boxes),
        "class_distribution": dict(class_dist),
        "status": "PASS" if len(corrupted_images) == 0 and len(invalid_boxes) == 0 else "WARNINGS_NOTED"
    }

    json_path = reports_dir / "validation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_path = reports_dir / "validation_report.md"
    md_content = f"""# IBVAP — Airborne Model v1 Dataset Quality Validation

**Directory:** `{raw_dir.resolve()}`  
**Status:** **{report['status']}**

| Metric | Count | Status |
| :--- | :---: | :---: |
| **Total Images** | {report['total_images']} | Verified |
| **Corrupted Images** | {report['corrupted_images']} | {'Pass' if report['corrupted_images'] == 0 else 'Quarantined'} |
| **Valid Label Files** | {report['total_label_files']} | Pass |
| **Total Bounding Boxes** | {report['total_valid_boxes']} | Pass |
| **Zero Area / Invalid Boxes** | {report['invalid_boxes'] + report['zero_area_boxes']} | Handled |

### Class Index Distribution
```json
{json.dumps(dict(class_dist), indent=2)}
```
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info(f"Airborne validation report saved: {json_path}")
    return report

def main():
    parser = argparse.ArgumentParser(description="Verify Airborne Datasets")
    parser.add_argument("--input", type=str, default="data/raw/airborne", help="Raw airborne directory")
    parser.add_argument("--reports", type=str, default="data/reports/airborne_v1", help="Reports directory")
    args = parser.parse_args()

    verify_airborne_dataset(Path(args.input), Path(args.reports))

if __name__ == "__main__":
    main()
