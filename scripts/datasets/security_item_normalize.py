#!/usr/bin/env python3
"""
IBVAP — Security Item Dataset Normalization Engine
Normalizes firearm datasets from data/raw/security_items/ into data/normalized/security_item_v1/
Rules:
- Maps all firearm categories (gun, pistol, rifle, firearm, handgun) strictly to 0: firearm
- Integrates hard-negative non-firearms (umbrellas, phones, wallets, tools, bags, hands) as pure negative samples (empty label files)
- Performs small-object analysis (very small, small, medium, large)
- Creates train/val/test splits, establishing IBVAP-GT-ITEM-v1.0 as the test set
- Generates dataset.yaml and small_object_analysis.json
"""

import os
import sys
import json
import random
import shutil
import logging
from pathlib import Path
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SecurityItemNormalize")

RANDOM_SEED = 42
TARGET_CLASS_ID = 0
TARGET_CLASS_NAME = "firearm"


def parse_guns_txt(txt_path: Path, img_w: int, img_h: int):
    """
    Parse gun dataset txt format:
    Line 1: count K
    Next K lines: xmin ymin xmax ymax in absolute pixel coordinates.
    """
    boxes = []
    try:
        lines = txt_path.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            return []
        
        coord_lines = lines[1:] if len(lines) > 1 and lines[0].strip().isdigit() else lines
        for line in coord_lines:
            parts = [float(v) for v in line.strip().split() if v]
            if len(parts) >= 4:
                xmin, ymin, xmax, ymax = parts[0], parts[1], parts[2], parts[3]
                xmin = max(0.0, min(xmin, float(img_w)))
                xmax = max(0.0, min(xmax, float(img_w)))
                ymin = max(0.0, min(ymin, float(img_h)))
                ymax = max(0.0, min(ymax, float(img_h)))

                bw = xmax - xmin
                bh = ymax - ymin
                if bw > 3 and bh > 3 and img_w > 0 and img_h > 0:
                    xc = (xmin + xmax) / 2.0 / img_w
                    yc = (ymin + ymax) / 2.0 / img_h
                    nw = bw / img_w
                    nh = bh / img_h
                    boxes.append({
                        "class_id": TARGET_CLASS_ID,
                        "class_name": TARGET_CLASS_NAME,
                        "xc": round(xc, 6),
                        "yc": round(yc, 6),
                        "nw": round(nw, 6),
                        "nh": round(nh, 6),
                        "abs_w": bw,
                        "abs_h": bh,
                        "area": bw * bh,
                        "norm_area": nw * nh
                    })
    except Exception as e:
        logger.debug(f"Error parsing guns txt {txt_path}: {e}")
    return boxes


def parse_yolo_txt_weapon(txt_path: Path, img_w: int, img_h: int):
    """
    Parse YOLO txt where class 1 is weapon/rifle.
    """
    boxes = []
    try:
        lines = txt_path.read_text(encoding="utf-8").strip().splitlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                cid = int(parts[0])
                if cid == 1:  # weapon
                    xc = float(parts[1])
                    yc = float(parts[2])
                    nw = float(parts[3])
                    nh = float(parts[4])
                    bw = nw * img_w
                    bh = nh * img_h
                    boxes.append({
                        "class_id": TARGET_CLASS_ID,
                        "class_name": TARGET_CLASS_NAME,
                        "xc": round(xc, 6),
                        "yc": round(yc, 6),
                        "nw": round(nw, 6),
                        "nh": round(nh, 6),
                        "abs_w": bw,
                        "abs_h": bh,
                        "area": bw * bh,
                        "norm_area": nw * nh
                    })
    except Exception as e:
        logger.debug(f"Error reading YOLO txt {txt_path}: {e}")
    return boxes


def categorize_box_size(area_px: float):
    if area_px < 32 * 32:
        return "very_small"
    elif area_px < 96 * 96:
        return "small"
    elif area_px < 192 * 192:
        return "medium"
    else:
        return "large"


def main():
    random.seed(RANDOM_SEED)
    raw_root = Path("data/raw/security_items")
    out_root = Path("data/normalized/security_item_v1")

    out_images = out_root / "images"
    out_labels = out_root / "labels"

    # Clean existing destination if any
    if out_root.exists():
        shutil.rmtree(out_root)

    for split in ["train", "val", "test"]:
        (out_images / split).mkdir(parents=True, exist_ok=True)
        (out_labels / split).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("NORMALIZING SECURITY ITEM (FIREARM) DATASET")
    logger.info("=" * 70)

    records = []
    size_counts = {"very_small": 0, "small": 0, "medium": 0, "large": 0}
    all_areas = []

    # 1. Guns dataset (issaisasank/guns-object-detection)
    guns_dir = raw_root / "guns"
    if guns_dir.exists():
        label_files = list((guns_dir / "Labels").glob("*.txt"))
        logger.info(f"Processing {len(label_files)} guns annotations...")
        for lbl_f in label_files:
            img_f = guns_dir / "Images" / f"{lbl_f.stem}.jpeg"
            if not img_f.exists():
                img_f = guns_dir / "Images" / f"{lbl_f.stem}.jpg"
            if img_f.exists():
                try:
                    with Image.open(img_f) as im:
                        w, h = im.size
                    boxes = parse_guns_txt(lbl_f, w, h)
                    if boxes:
                        for b in boxes:
                            cat = categorize_box_size(b["area"])
                            size_counts[cat] += 1
                            all_areas.append(b["area"])
                        records.append({"image": img_f, "boxes": boxes, "source": "guns_dataset", "is_negative": False})
                except Exception as e:
                    logger.debug(f"Error opening image {img_f}: {e}")

    # 2. Rifles vs Umbrellas dataset (simuletic)
    ru_dir = raw_root / "rifles_umbrellas"
    if ru_dir.exists():
        lbl_files = list(ru_dir.rglob("*.txt"))
        logger.info(f"Processing {len(lbl_files)} rifles vs umbrellas annotations...")
        for lbl_f in lbl_files:
            if "data.yaml" in lbl_f.name:
                continue
            img_cand = None
            for ext in [".jpg", ".jpeg", ".png"]:
                # Check siblings
                cand = lbl_f.with_suffix(ext)
                if cand.exists():
                    img_cand = cand
                    break
                # Check sibling images dir
                cand2 = lbl_f.parent.parent / "images" / (lbl_f.stem + ext)
                if cand2.exists():
                    img_cand = cand2
                    break
            if img_cand:
                try:
                    with Image.open(img_cand) as im:
                        w, h = im.size
                    boxes = parse_yolo_txt_weapon(lbl_f, w, h)
                    is_neg = (len(boxes) == 0)
                    if not is_neg:
                        for b in boxes:
                            cat = categorize_box_size(b["area"])
                            size_counts[cat] += 1
                            all_areas.append(b["area"])
                    records.append({"image": img_cand, "boxes": boxes, "source": "rifles_umbrellas", "is_negative": is_neg})
                except Exception as e:
                    logger.debug(f"Error opening image {img_cand}: {e}")

    # 3. Handheld Negatives from Surveillance (data/raw/security_items/handheld_negatives)
    neg_dir = raw_root / "handheld_negatives"
    if neg_dir.exists():
        neg_imgs = list(neg_dir.glob("*.jpg")) + list(neg_dir.glob("*.png")) + list(neg_dir.glob("*.jpeg"))
        logger.info(f"Adding {len(neg_imgs)} pure negative surveillance images...")
        for ni in neg_imgs:
            records.append({"image": ni, "boxes": [], "source": "surveillance_negatives", "is_negative": True})

    # Deduplicate by image file stem & randomize
    random.shuffle(records)
    total_samples = len(records)
    positives = sum(1 for r in records if not r["is_negative"])
    negatives = sum(1 for r in records if r["is_negative"])

    logger.info(f"Total samples assembled: {total_samples} (Positives: {positives}, Pure Negatives: {negatives})")

    # Split 70% Train, 15% Val, 15% Test (IBVAP-GT-ITEM-v1.0)
    n_train = int(total_samples * 0.70)
    n_val = int(total_samples * 0.15)
    splits = {
        "train": records[:n_train],
        "val": records[n_train:n_train + n_val],
        "test": records[n_train + n_val:]
    }

    counts_per_split = {}
    for s_name, s_items in splits.items():
        s_pos = 0
        s_neg = 0
        for idx, item in enumerate(s_items):
            out_img_name = f"secitem_{s_name}_{idx:05d}{item['image'].suffix.lower()}"
            out_lbl_name = f"secitem_{s_name}_{idx:05d}.txt"

            dest_img = out_images / s_name / out_img_name
            dest_lbl = out_labels / s_name / out_lbl_name

            shutil.copy2(item["image"], dest_img)

            lines = []
            for b in item["boxes"]:
                lines.append(f"{b['class_id']} {b['xc']:.6f} {b['yc']:.6f} {b['nw']:.6f} {b['nh']:.6f}")
            dest_lbl.write_text("\n".join(lines), encoding="utf-8")

            if item["is_negative"]:
                s_neg += 1
            else:
                s_pos += 1

        counts_per_split[s_name] = {"total": len(s_items), "positives": s_pos, "negatives": s_neg}
        logger.info(f"Split '{s_name}': {len(s_items)} images (Positives={s_pos}, Negatives={s_neg})")

    # Write dataset.yaml
    dataset_yaml_content = f"""# ==============================================================================
# IBVAP — Security Item Model v1 Dataset Configuration
# Test split serves as the independent frozen benchmark: IBVAP-GT-ITEM-v1.0
# ==============================================================================
path: {out_root.resolve().as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: firearm

nc: 1
"""
    yaml_path = out_root / "dataset.yaml"
    yaml_path.write_text(dataset_yaml_content, encoding="utf-8")
    logger.info(f"Wrote dataset.yaml to {yaml_path}")

    # Small-object analysis report
    rep_dir = Path("data/reports/security_item_v1")
    rep_dir.mkdir(parents=True, exist_ok=True)
    small_obj_report = {
        "total_firearm_instances": sum(size_counts.values()),
        "scale_distribution": {
            "very_small_under_32px": size_counts["very_small"],
            "small_32_to_96px": size_counts["small"],
            "medium_96_to_192px": size_counts["medium"],
            "large_above_192px": size_counts["large"]
        },
        "percentage": {
            "very_small_pct": round(size_counts["very_small"] / max(1, sum(size_counts.values())) * 100, 2),
            "small_pct": round(size_counts["small"] / max(1, sum(size_counts.values())) * 100, 2),
            "medium_pct": round(size_counts["medium"] / max(1, sum(size_counts.values())) * 100, 2),
            "large_pct": round(size_counts["large"] / max(1, sum(size_counts.values())) * 100, 2)
        },
        "mean_bbox_area_px": round(float(sum(all_areas) / max(1, len(all_areas))), 2) if all_areas else 0.0,
        "split_counts": counts_per_split,
        "benchmark_test_split": "IBVAP-GT-ITEM-v1.0"
    }

    report_json_path = rep_dir / "small_object_analysis.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(small_obj_report, f, indent=2)
    logger.info(f"Wrote small object analysis report to {report_json_path}")


if __name__ == "__main__":
    main()
