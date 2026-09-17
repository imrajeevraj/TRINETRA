"""
IBVAP — Security Item Dataset Normalization & Benchmark v2 Engine
Constructs data/normalized/security_item_v2/ with:
- Surveillance-adapted firearm positives
- Security Item Hard Negative Dataset v2 (phones, radios, tools, flashlights, hands)
- Frozen independent benchmark: IBVAP-GT-ITEM-v2.0
- Person-item association manifest without subjective inferences
"""

import os
import shutil
import hashlib
import json
import logging
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("NormalizeV2")

RAW_ROOT = Path("data/raw/security_items")
OUTPUT_DIR = Path("data/normalized/security_item_v2")
BENCHMARK_NAME = "IBVAP-GT-ITEM-v2.0"


def apply_surveillance_augmentations(img: np.ndarray, scenario: str) -> np.ndarray:
    """
    Applies realistic physical border/CCTV degradation:
    - Low-light outdoor scenes
    - High-angle perspective tilt
    - Video compression & sensor noise
    """
    h, w = img.shape[:2]
    out = img.copy()

    if scenario == "low_light":
        # Attenuate brightness and add sensor noise
        gamma = np.random.uniform(1.8, 2.4)
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        out = cv2.LUT(out, table)
        noise = np.random.normal(0, 8, out.shape).astype(np.int16)
        out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    elif scenario == "compression_blur":
        # JPEG compression artifact emulation
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), np.random.randint(35, 60)]
        _, enc = cv2.imencode('.jpg', out, encode_param)
        out = cv2.imdecode(enc, 1)
        # Subtle motion blur
        k_sz = np.random.choice([3, 5])
        kernel = np.zeros((k_sz, k_sz))
        kernel[int((k_sz - 1) / 2), :] = np.ones(k_sz)
        kernel /= k_sz
        out = cv2.filter2D(out, -1, kernel)

    elif scenario == "outdoor_clutter":
        # Color temperature shift (cold outdoor dusk)
        b, g, r = cv2.split(out)
        b = np.clip(b.astype(np.int16) + 15, 0, 255).astype(np.uint8)
        r = np.clip(r.astype(np.int16) - 10, 0, 255).astype(np.uint8)
        out = cv2.merge([b, g, r])

    return out


def parse_guns_txt(txt_path: Path, img_w: int, img_h: int):
    """Parse guns txt format: xmin ymin xmax ymax in absolute pixel coordinates."""
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
                    boxes.append((xc, yc, nw, nh))
    except Exception as e:
        logger.debug(f"Error parsing guns txt {txt_path}: {e}")
    return boxes


def run_normalization():
    logger.info("Initializing Security Item Dataset Normalization v2...")
    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    positives = []
    negatives = []
    person_associations = []

    # 1. Ingest Guns dataset
    guns_dir = RAW_ROOT / "guns"
    if guns_dir.exists():
        lbl_files = list((guns_dir / "Labels").glob("*.txt"))
        logger.info(f"Processing {len(lbl_files)} guns label files...")
        for lbl_p in lbl_files:
            img_p = guns_dir / "Images" / f"{lbl_p.stem}.jpeg"
            if not img_p.exists():
                img_p = guns_dir / "Images" / f"{lbl_p.stem}.jpg"
            if img_p.exists():
                im = cv2.imread(str(img_p))
                if im is not None:
                    h, w = im.shape[:2]
                    boxes = parse_guns_txt(lbl_p, w, h)
                    if boxes:
                        positives.append({"img_path": img_p, "boxes": boxes, "source": "voc_guns"})

    # 2. Ingest Rifles / CCTV Weapons
    rifles_dir = RAW_ROOT / "rifles_umbrellas"
    if rifles_dir.exists():
        lbl_files = list(rifles_dir.rglob("*.txt"))
        logger.info(f"Processing {len(lbl_files)} rifles vs umbrellas label files...")
        for lbl_p in lbl_files:
            if "data.yaml" in lbl_p.name: continue
            img_p = None
            for ext in [".jpg", ".jpeg", ".png"]:
                cand = lbl_p.with_suffix(ext)
                if cand.exists():
                    img_p = cand
                    break
            if img_p and img_p.exists():
                boxes = []
                lines = lbl_p.read_text(encoding="utf-8").strip().splitlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5 and int(parts[0]) == 1:  # weapon
                        cx, cy, bw, bh = [float(v) for v in parts[1:5]]
                        boxes.append((cx, cy, bw, bh))
                if boxes:
                    positives.append({"img_path": img_p, "boxes": boxes, "source": "rifles_surveillance"})

    # 3. Ingest Hard Negatives v2
    neg_v2_dir = RAW_ROOT / "hard_negatives_v2" / "images"
    if neg_v2_dir.exists():
        for p in neg_v2_dir.glob("*.jpg"):
            negatives.append({"img_path": p, "source": "hard_negatives_v2"})

    # 4. Ingest Handheld Negatives
    handheld_dir = RAW_ROOT / "handheld_negatives" / "images"
    if handheld_dir.exists():
        for p in handheld_dir.glob("*.*"):
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                negatives.append({"img_path": p, "source": "handheld_negatives"})

    logger.info(f"Assembled {len(positives)} positive images and {len(negatives)} hard-negative images.")

    # Controlled Split: 70% train, 15% val, 15% test (IBVAP-GT-ITEM-v2.0)
    np.random.seed(42)
    np.random.shuffle(positives)
    np.random.shuffle(negatives)

    n_pos_test = int(len(positives) * 0.15)
    n_pos_val = int(len(positives) * 0.15)
    pos_test = positives[:n_pos_test]
    pos_val = positives[n_pos_test:n_pos_test + n_pos_val]
    pos_train = positives[n_pos_test + n_pos_val:]

    n_neg_test = int(len(negatives) * 0.15)
    n_neg_val = int(len(negatives) * 0.15)
    neg_test = negatives[:n_neg_test]
    neg_val = negatives[n_neg_test:n_neg_test + n_neg_val]
    neg_train = negatives[n_neg_test + n_neg_val:]

    splits_data = {
        "train": (pos_train, neg_train),
        "val": (pos_val, neg_val),
        "test": (pos_test, neg_test)
    }

    manifest = {"benchmark": BENCHMARK_NAME, "files": {}}

    for split_name, (pos_list, neg_list) in splits_data.items():
        idx = 0
        dest_img_dir = OUTPUT_DIR / "images" / split_name
        dest_lbl_dir = OUTPUT_DIR / "labels" / split_name

        # Process positives
        for item in pos_list:
            src_img = cv2.imread(str(item["img_path"]))
            if src_img is None: continue

            # Apply realistic surveillance condition for training & testing
            scenario = "standard"
            if idx % 4 == 1: scenario = "low_light"
            elif idx % 4 == 2: scenario = "compression_blur"
            elif idx % 4 == 3: scenario = "outdoor_clutter"

            proc_img = apply_surveillance_augmentations(src_img, scenario)
            out_img_name = f"secitem_v2_{split_name}_{idx:05d}.jpg"
            out_img_path = dest_img_dir / out_img_name
            cv2.imwrite(str(out_img_path), proc_img)

            # Write YOLO labels (class 0: firearm)
            out_lbl_path = dest_lbl_dir / f"secitem_v2_{split_name}_{idx:05d}.txt"
            with open(out_lbl_path, "w", encoding="utf-8") as lf:
                for cx, cy, bw, bh in item["boxes"]:
                    lf.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            # Track person association metadata
            if item["source"] in ["rifles_surveillance", "voc_guns"]:
                # Synthesize contextual person bounding box around firearm
                h, w = proc_img.shape[:2]
                for b_idx, (cx, cy, bw, bh) in enumerate(item["boxes"]):
                    px1 = max(0.0, (cx - bw) * w)
                    py1 = max(0.0, (cy - bh * 1.5) * h)
                    px2 = min(float(w), (cx + bw * 1.2) * w)
                    py2 = min(float(h), (cy + bh * 2.5) * h)
                    person_associations.append({
                        "image": out_img_name,
                        "split": split_name,
                        "firearm_bbox": [round((cx - bw / 2.0) * w, 1), round((cy - bh / 2.0) * h, 1),
                                         round((cx + bw / 2.0) * w, 1), round((cy + bh / 2.0) * h, 1)],
                        "person_bbox": [round(px1, 1), round(py1, 1), round(px2, 1), round(py2, 1)],
                        "association_type": "SPATIAL_PROXIMITY_HANDHELD"
                    })

            if split_name == "test":
                hasher = hashlib.sha256()
                with open(out_img_path, "rb") as f:
                    while chunk := f.read(65536): hasher.update(chunk)
                manifest["files"][out_img_name] = hasher.hexdigest().upper()

            idx += 1

        # Process negatives (empty labels)
        for item in neg_list:
            src_img = cv2.imread(str(item["img_path"]))
            if src_img is None: continue

            out_img_name = f"secitem_v2_{split_name}_{idx:05d}_neg.jpg"
            out_img_path = dest_img_dir / out_img_name
            cv2.imwrite(str(out_img_path), src_img)

            # Empty label
            out_lbl_path = dest_lbl_dir / f"secitem_v2_{split_name}_{idx:05d}_neg.txt"
            out_lbl_path.write_text("", encoding="utf-8")

            if split_name == "test":
                hasher = hashlib.sha256()
                with open(out_img_path, "rb") as f:
                    while chunk := f.read(65536): hasher.update(chunk)
                manifest["files"][out_img_name] = hasher.hexdigest().upper()

            idx += 1

        logger.info(f"Split '{split_name}' finalized: {idx} images ({len(pos_list)} pos, {len(neg_list)} neg).")

    # Save dataset.yaml
    dataset_yaml_content = f"""# ==============================================================================
# IBVAP — Security Item Model v2 Dataset Configuration
# Test split serves as the independent frozen benchmark: IBVAP-GT-ITEM-v2.0
# ==============================================================================
path: {OUTPUT_DIR.resolve().as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: firearm

nc: 1
"""
    with open(OUTPUT_DIR / "dataset.yaml", "w", encoding="utf-8") as f:
        f.write(dataset_yaml_content)

    # Save benchmark manifest
    manifest_path = Path("data/reports/security_item_v2/ibvap_gt_item_v2_manifest.yaml")
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f)

    # Save person-item associations
    assoc_path = Path("data/reports/security_item_v2/person_item_associations.json")
    with open(assoc_path, "w", encoding="utf-8") as f:
        json.dump(person_associations, f, indent=2)

    logger.info(f"Normalization v2 completed. Benchmark manifest written to {manifest_path}")


if __name__ == "__main__":
    run_normalization()
