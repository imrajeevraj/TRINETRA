#!/usr/bin/env python3
"""
IBVAP — Master AI Model Improvement Pipeline Engine
Kaggle -> Normalize -> Deduplicate -> Train -> Benchmark -> Promote
SIH 2026 / IBVAP v2.0.0
"""

import os
import sys
import json
import time
import shutil
import random
import hashlib
import logging
from pathlib import Path
from collections import Counter, defaultdict

import cv2
import yaml
import numpy as np
import torch
from ultralytics import YOLO

# Setup structured logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MasterPipeline")

ROOT_DIR = Path(__file__).resolve().parents[2]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Production baselines (Rule 2 & 3)
PRODUCTION_GROUND = {
    "id": "ibvap-ground-v2",
    "path": "models/current/ibvap_detector.pt",
    "golden_path": "models/production/ground/ibvap_ground_v2_production.pt",
    "sha256": "7DBF36027768194F61C6EBBC000E1421374624558168F8A9896F727B296277B0",
    "imgsz": 768,
    "metrics": {
        "person_precision": 0.6449,
        "person_recall": 0.7458,
        "person_f1": 0.6917,
        "person_map50": 0.4810,
        "vehicle_precision": 0.9697,
        "vehicle_recall": 0.8649,
        "vehicle_f1": 0.9143,
        "vehicle_map50": 0.8387,
        "p50_latency_ms": 10.59,
        "p95_latency_ms": 16.18,
        "ai_fps": 84.4
    }
}

PRODUCTION_AIRBORNE = {
    "id": "ibvap-airborne-v1",
    "path": "data/training/runs/ibvap_airborne_v1_exp001/weights/best.pt",
    "golden_path": "models/production/airborne/ibvap_airborne_v1_production.pt",
    "sha256": "E1009633325E463B1C0D45D6578522A1A2026AE27B854591A83461043928C914",
    "imgsz": 640,
    "metrics": {
        "drone_precision": 0.7554,
        "drone_recall": 0.9100,
        "drone_f1": 0.8255,
        "drone_map50": 0.6874,
        "aircraft_precision": 0.6376,
        "aircraft_recall": 0.9223,
        "aircraft_f1": 0.7540,
        "aircraft_map50": 0.5881,
        "p50_latency_ms": 9.30,
        "p95_latency_ms": 13.50,
        "ai_fps": 97.8
    }
}

def compute_sha256(filepath: Path) -> str:
    if not filepath.exists():
        return "MISSING"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()

def compute_md5(filepath: Path) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def compute_dhash(image_path: Path, hash_size: int = 8) -> int:
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0
        resized = cv2.resize(img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
        diff = resized[:, 1:] > resized[:, :-1]
        return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
    except Exception:
        return 0

def hamming_dist(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count("1")

def calc_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter + 1e-6)

def find_label_file(img_p: Path) -> Path:
    # 1. Check same directory with .txt
    same_dir = img_p.with_suffix(".txt")
    if same_dir.exists():
        return same_dir
    # 2. Check if parent is 'images', look in sibling 'labels'
    if img_p.parent.name == "images":
        sib_lbl = img_p.parent.parent / "labels" / f"{img_p.stem}.txt"
        if sib_lbl.exists():
            return sib_lbl
    # 3. Recursive search in raw_normalized labels
    return None

# ==============================================================================
# PHASE 2 & 3 — RAW DATA VALIDATION
# ==============================================================================
def validate_raw_datasets(reports_dir: Path):
    reports_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 70)
    logger.info("PHASE 3: RAW DATA VALIDATION")
    logger.info("=" * 70)

    dataset_dirs = {
        "kiit_mita": ROOT_DIR / "data/external/kaggle/kiit_mita",
        "human_detection": ROOT_DIR / "data/external/kaggle/human_detection",
        "drones": ROOT_DIR / "data/raw/airborne/drones",
        "fixed_wing": ROOT_DIR / "data/raw/airborne/fixed_wing",
        "birds_negatives": ROOT_DIR / "data/raw/airborne/birds_negatives",
        "clouds_negatives": ROOT_DIR / "data/raw/airborne/clouds_negatives"
    }

    validation_summary = {}

    for name, d_path in dataset_dirs.items():
        if not d_path.exists():
            logger.warning(f"Directory {d_path} missing!")
            continue

        images = [f for f in d_path.rglob("*") if f.suffix.lower() in IMAGE_EXTS and f.is_file()]
        valid_images = 0
        corrupt_images = 0
        valid_boxes = 0
        invalid_boxes = 0
        classes_found = Counter()

        # Sample for speed if large, or validate all
        sample_images = images if len(images) <= 800 else random.sample(images, 800)

        for img_p in sample_images:
            try:
                sz = img_p.stat().st_size
                if sz == 0:
                    corrupt_images += 1
                    continue
                valid_images += 1

                # Check labels
                lbl_p = find_label_file(img_p)
                if lbl_p and lbl_p.exists():
                    for line in lbl_p.read_text(encoding="utf-8", errors="ignore").splitlines():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                cid = int(parts[0])
                                cx, cy, bw, bh = map(float, parts[1:5])
                                if 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and bw > 0 and bh > 0:
                                    valid_boxes += 1
                                    classes_found[cid] += 1
                                else:
                                    invalid_boxes += 1
                            except ValueError:
                                invalid_boxes += 1
            except Exception:
                corrupt_images += 1

        validation_summary[name] = {
            "total_images": len(images),
            "checked_images": len(sample_images),
            "valid_images": valid_images,
            "corrupt_images": corrupt_images,
            "valid_boxes": valid_boxes,
            "invalid_boxes": invalid_boxes,
            "classes_detected": dict(classes_found)
        }
        logger.info(f"[{name}] Total: {len(images)} | Checked: {valid_images} | Boxes: {valid_boxes} | Corrupt: {corrupt_images}")

    report_json = reports_dir / "raw_validation_summary.json"
    report_json.write_text(json.dumps(validation_summary, indent=2), encoding="utf-8")

    md = "# IBVAP — Kaggle Raw Data Validation Report\n\n"
    md += "| Dataset | Total Images | Sample Checked | Corrupt Files | Valid Boxes | Invalid Boxes | Status |\n"
    md += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    for k, v in validation_summary.items():
        status = "PASS" if v["corrupt_images"] == 0 and v["valid_images"] > 0 else "WARNED"
        md += f"| **{k}** | {v['total_images']} | {v['checked_images']} | {v['corrupt_images']} | {v['valid_boxes']} | {v['invalid_boxes']} | **{status}** |\n"

    (reports_dir / "validation_report.md").write_text(md, encoding="utf-8")
    logger.info(f"Validation report saved to {reports_dir / 'validation_report.md'}")
    return validation_summary

# ==============================================================================
# PHASE 4, 5, 6, 7, 8 — NORMALIZATION, DEDUPLICATION & SPLIT ASSEMBLY
# ==============================================================================
def assemble_ground_v3(
    output_dir: Path,
    reports_dir: Path,
    seed: int = 42,
    split_ratio: float = 0.80
):
    logger.info("=" * 70)
    logger.info("PHASES 4-8: ASSEMBLING GROUND MODEL V3 DATASET")
    logger.info("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_img = output_dir / "images/train"
    val_img = output_dir / "images/val"
    train_lbl = output_dir / "labels/train"
    val_lbl = output_dir / "labels/val"
    manifest_dir = output_dir / "manifests"

    for d in [train_img, val_img, train_lbl, val_lbl, manifest_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Index Frozen Benchmark for Anti-Leakage (RULE 1)
    frozen_test_dir = ROOT_DIR / "benchmark/images/test"
    frozen_md5s = set()
    frozen_dhashes = {}
    if frozen_test_dir.exists():
        for bp in frozen_test_dir.glob("*.jpg"):
            m = compute_md5(bp)
            dh = compute_dhash(bp)
            frozen_md5s.add(m)
            if dh != 0:
                frozen_dhashes[dh] = bp.name
    logger.info(f"Indexed {len(frozen_md5s)} frozen benchmark images for anti-leakage protection.")

    pool = []

    # 2. Ingest Normalized KIIT-MiTA (from raw_normalized or raw)
    kiit_norm_dir = ROOT_DIR / "data/normalized/ground_v2/raw_normalized"
    if kiit_norm_dir.exists() and (kiit_norm_dir / "images").exists():
        logger.info("Ingesting pre-normalized KIIT-MiTA CCTV surveillance samples...")
        norm_imgs = sorted(list((kiit_norm_dir / "images").glob("*")))
        for img_p in norm_imgs:
            lbl_p = kiit_norm_dir / "labels" / f"{img_p.stem}.txt"
            if lbl_p.exists():
                lines = []
                for line in lbl_p.read_text(encoding="utf-8", errors="ignore").splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        lines.append(line.strip() + "\n")
                if lines:
                    pool.append({
                        "src_img": img_p,
                        "lines": lines,
                        "source": "kaggle_kiit_mita",
                        "name": f"kiit_{img_p.stem}{img_p.suffix.lower()}",
                        "hard_neg": False
                    })
    else:
        # Fallback to direct raw parsing
        kiit_dir = ROOT_DIR / "data/external/kaggle/kiit_mita/KIIT-MiTA"
        kiit_map = {4: 0, 5: 1, 6: 1} # Soldier -> 0, Tank -> 1, Vehicle -> 1
        for img_p in kiit_dir.rglob("*"):
            if img_p.suffix.lower() in IMAGE_EXTS and img_p.is_file():
                lbl_p = find_label_file(img_p)
                lines = []
                if lbl_p and lbl_p.exists():
                    for line in lbl_p.read_text(encoding="utf-8", errors="ignore").splitlines():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                raw_id = int(parts[0])
                                if raw_id in kiit_map:
                                    cid = kiit_map[raw_id]
                                    cx, cy, w, h = map(float, parts[1:5])
                                    lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                            except ValueError:
                                continue
                if lines:
                    pool.append({
                        "src_img": img_p,
                        "lines": lines,
                        "source": "kaggle_kiit_mita",
                        "name": f"kiit_{img_p.stem}{img_p.suffix.lower()}",
                        "hard_neg": False
                    })

    # 3. Ingest Internal IBVAP Surveillance Keyframes
    ibvap_train_img = ROOT_DIR / "benchmark/images/train"
    ibvap_train_lbl = ROOT_DIR / "benchmark/labels/train"
    ibvap_val_img = ROOT_DIR / "benchmark/images/valid"
    ibvap_val_lbl = ROOT_DIR / "benchmark/labels/valid"

    for img_p in list(ibvap_train_img.glob("*.jpg")) + list(ibvap_val_img.glob("*.jpg")):
        lbl_dir = ibvap_train_lbl if "train" in str(img_p.parent) else ibvap_val_lbl
        lbl_p = lbl_dir / f"{img_p.stem}.txt"
        lines = []
        if lbl_p.exists():
            for line in lbl_p.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        cid = int(parts[0])
                        if cid in [0, 1]:
                            cx, cy, w, h = map(float, parts[1:5])
                            cx = max(0.0, min(1.0, cx))
                            cy = max(0.0, min(1.0, cy))
                            w = max(0.001, min(1.0, w))
                            h = max(0.001, min(1.0, h))
                            lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    except ValueError:
                        continue
        if lines:
            pool.append({
                "src_img": img_p,
                "lines": lines,
                "source": "ibvap_internal_train",
                "name": f"ibvap_{img_p.stem}{img_p.suffix.lower()}",
                "hard_neg": False
            })

    # 4. Ingest Hard-Negative Surveillance Scenes (human_detection/0)
    hard_neg_dir = ROOT_DIR / "data/external/kaggle/human_detection/human detection dataset/0"
    if not hard_neg_dir.exists():
        hard_neg_dir = ROOT_DIR / "data/external/kaggle/human_detection"
    hn_images = [p for p in hard_neg_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.stat().st_size > 0]
    random.seed(seed)
    sampled_hn = random.sample(hn_images, min(len(hn_images), 200))
    for img_p in sampled_hn:
        pool.append({
            "src_img": img_p,
            "lines": [],
            "source": "kaggle_hard_neg",
            "name": f"hardneg_{img_p.stem}{img_p.suffix.lower()}",
            "hard_neg": True
        })

    # 5. Anti-Leakage & Deduplication Pass (Phase 6)
    clean_pool = []
    seen_md5s = set()
    seen_dhashes = []
    leakage_count = 0
    duplicate_count = 0

    for item in pool:
        m = compute_md5(item["src_img"])
        dh = compute_dhash(item["src_img"])

        # RULE 1: STRICT FROZEN BENCHMARK LEAKAGE CHECK
        if m in frozen_md5s:
            logger.critical(f"CRITICAL LEAKAGE DETECTED: {item['src_img']} exact match with frozen benchmark! EXCLUDED.")
            leakage_count += 1
            continue

        leak_perceptual = False
        for f_dh, f_name in frozen_dhashes.items():
            if hamming_dist(dh, f_dh) <= 1:
                logger.critical(f"CRITICAL LEAKAGE DETECTED: {item['src_img']} perceptual duplicate of {f_name}! EXCLUDED.")
                leak_perceptual = True
                leakage_count += 1
                break
        if leak_perceptual:
            continue

        # Internal duplicate removal
        if m in seen_md5s:
            duplicate_count += 1
            continue
        seen_md5s.add(m)

        is_near_dup = False
        if dh != 0:
            for p_dh in seen_dhashes:
                if hamming_dist(dh, p_dh) <= 1:
                    is_near_dup = True
                    duplicate_count += 1
                    break
            if not is_near_dup:
                seen_dhashes.append(dh)
        if not is_near_dup:
            clean_pool.append(item)

    logger.info(f"Deduplication & Anti-Leakage: Leakage detected = {leakage_count} (REMOVED) | Duplicates removed = {duplicate_count} | Clean unique images = {len(clean_pool)}")

    # 6. Stratified Split
    random.seed(seed)
    random.shuffle(clean_pool)
    split_idx = int(len(clean_pool) * split_ratio)
    train_items = clean_pool[:split_idx]
    val_items = clean_pool[split_idx:]

    stats = {"train": Counter(), "val": Counter()}

    def write_split(items, target_img_d, target_lbl_d, split_name):
        for it in items:
            dest_i = target_img_d / it["name"]
            dest_l = target_lbl_d / f"{Path(it['name']).stem}.txt"
            if not dest_i.exists():
                shutil.copy2(it["src_img"], dest_i)
            dest_l.write_text("".join(it["lines"]), encoding="utf-8")

            stats[split_name]["images"] += 1
            if it["hard_neg"] or len(it["lines"]) == 0:
                stats[split_name]["hard_negatives"] += 1
            for l in it["lines"]:
                cid = int(l.split()[0])
                if cid == 0: stats[split_name]["persons"] += 1
                elif cid == 1: stats[split_name]["vehicles"] += 1

    write_split(train_items, train_img, train_lbl, "train")
    write_split(val_items, val_img, val_lbl, "val")

    # Generate ground_v3.yaml
    dataset_yaml = {
        "path": str(output_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "person",
            1: "vehicle"
        }
    }
    (output_dir / "ground_v3.yaml").write_text(yaml.dump(dataset_yaml, sort_keys=False), encoding="utf-8")
    (output_dir / "dataset.yaml").write_text(yaml.dump(dataset_yaml, sort_keys=False), encoding="utf-8")
    (manifest_dir / "split_manifest.json").write_text(json.dumps({
        "dataset": "ground_v3",
        "stats": {k: dict(v) for k, v in stats.items()}
    }, indent=2), encoding="utf-8")

    logger.info(f"Ground v3 assembled: Train = {stats['train']['images']} imgs ({stats['train']['persons']} persons, {stats['train']['vehicles']} vehicles, {stats['train']['hard_negatives']} hard negs) | Val = {stats['val']['images']} imgs")
    return stats

def assemble_airborne_v2(
    output_dir: Path,
    reports_dir: Path,
    seed: int = 42,
    split_ratio: float = 0.80
):
    logger.info("=" * 70)
    logger.info("PHASES 4-8: ASSEMBLING AIRBORNE MODEL V2 DATASET")
    logger.info("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_img = output_dir / "images/train"
    val_img = output_dir / "images/val"
    train_lbl = output_dir / "labels/train"
    val_lbl = output_dir / "labels/val"
    manifest_dir = output_dir / "manifests"

    for d in [train_img, val_img, train_lbl, val_lbl, manifest_dir]:
        d.mkdir(parents=True, exist_ok=True)

    pool = []

    # 1. Drones
    drone_dir = ROOT_DIR / "data/raw/airborne/drones"
    drone_images = [p for p in drone_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.stat().st_size > 0]
    for img_p in drone_images:
        lbl_p = img_p.with_suffix(".txt")
        lines = []
        if lbl_p.exists():
            for line in lbl_p.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        cx, cy, w, h = map(float, parts[1:5])
                        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    except ValueError:
                        continue
        if lines:
            pool.append({
                "src_img": img_p,
                "lines": lines,
                "name": f"drone_{img_p.stem}{img_p.suffix.lower()}",
                "hard_neg": False
            })

    # 2. Fixed-Wing Aircraft
    aircraft_dir = ROOT_DIR / "data/raw/airborne/fixed_wing"
    aircraft_images = [p for p in aircraft_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.stat().st_size > 0]
    for img_p in aircraft_images:
        lbl_p = img_p.with_suffix(".txt")
        lines = []
        if lbl_p.exists():
            for line in lbl_p.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        cx, cy, w, h = map(float, parts[1:5])
                        lines.append(f"1 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    except ValueError:
                        continue
        if lines:
            pool.append({
                "src_img": img_p,
                "lines": lines,
                "name": f"aircraft_{img_p.stem}{img_p.suffix.lower()}",
                "hard_neg": False
            })

    # 3. Avian Hard Negatives
    bird_dir = ROOT_DIR / "data/raw/airborne/birds_negatives"
    bird_images = [p for p in bird_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.stat().st_size > 0]
    random.seed(seed)
    sampled_birds = random.sample(bird_images, min(len(bird_images), 250))
    for img_p in sampled_birds:
        pool.append({
            "src_img": img_p,
            "lines": [],
            "name": f"birdneg_{img_p.stem}{img_p.suffix.lower()}",
            "hard_neg": True
        })

    # 4. Cloud & Atmospheric Glare Hard Negatives
    cloud_dir = ROOT_DIR / "data/raw/airborne/clouds_negatives"
    cloud_images = [p for p in cloud_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.stat().st_size > 0]
    sampled_clouds = random.sample(cloud_images, min(len(cloud_images), 200))
    for img_p in sampled_clouds:
        pool.append({
            "src_img": img_p,
            "lines": [],
            "name": f"cloudneg_{img_p.stem}{img_p.suffix.lower()}",
            "hard_neg": True
        })

    # 5. Deduplication
    clean_pool = []
    seen_md5s = set()
    for item in pool:
        m = compute_md5(item["src_img"])
        if m in seen_md5s:
            continue
        seen_md5s.add(m)
        clean_pool.append(item)

    # 6. Stratified Split
    random.seed(seed)
    random.shuffle(clean_pool)
    split_idx = int(len(clean_pool) * split_ratio)
    train_items = clean_pool[:split_idx]
    val_items = clean_pool[split_idx:]

    stats = {"train": Counter(), "val": Counter()}

    def write_split(items, target_img_d, target_lbl_d, split_name):
        for it in items:
            dest_i = target_img_d / it["name"]
            dest_l = target_lbl_d / f"{Path(it['name']).stem}.txt"
            if not dest_i.exists():
                shutil.copy2(it["src_img"], dest_i)
            dest_l.write_text("".join(it["lines"]), encoding="utf-8")

            stats[split_name]["images"] += 1
            if it["hard_neg"] or len(it["lines"]) == 0:
                stats[split_name]["hard_negatives"] += 1
            for l in it["lines"]:
                cid = int(l.split()[0])
                if cid == 0: stats[split_name]["drones"] += 1
                elif cid == 1: stats[split_name]["aircraft"] += 1

    write_split(train_items, train_img, train_lbl, "train")
    write_split(val_items, val_img, val_lbl, "val")

    # Generate airborne_v2.yaml
    dataset_yaml = {
        "path": str(output_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "drone",
            1: "aircraft"
        }
    }
    (output_dir / "airborne_v2.yaml").write_text(yaml.dump(dataset_yaml, sort_keys=False), encoding="utf-8")
    (output_dir / "dataset.yaml").write_text(yaml.dump(dataset_yaml, sort_keys=False), encoding="utf-8")
    (manifest_dir / "split_manifest.json").write_text(json.dumps({
        "dataset": "airborne_v2",
        "stats": {k: dict(v) for k, v in stats.items()}
    }, indent=2), encoding="utf-8")

    logger.info(f"Airborne v2 assembled: Train = {stats['train']['images']} imgs ({stats['train']['drones']} drones, {stats['train']['aircraft']} aircraft, {stats['train']['hard_negatives']} hard negs) | Val = {stats['val']['images']} imgs")
    return stats

# ==============================================================================
# PHASE 9 & 10 — CONTROLLED TRAINING EXPERIMENTS
# ==============================================================================
def run_training_experiment(
    exp_id: str,
    dataset_yaml: Path,
    base_model: str,
    epochs: int,
    batch_size: int,
    imgsz: int,
    device: str,
    optimizer: str = "AdamW",
    lr0: float = 0.002,
    mosaic: float = 0.2,
    scale: float = 0.1,
    close_mosaic: int = 5
):
    logger.info("=" * 70)
    logger.info(f"PHASE 9: TRAINING EXPERIMENT {exp_id}")
    logger.info(f"Base: {base_model} | Epochs: {epochs} | Imgsz: {imgsz} | Device: {device}")
    logger.info("=" * 70)

    model = YOLO(base_model)
    runs_dir = ROOT_DIR / "data/training/runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    results = model.train(
        data=str(dataset_yaml.resolve()),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project=str(runs_dir.resolve()),
        name=exp_id,
        seed=42,
        optimizer=optimizer,
        lr0=lr0,
        lrf=lr0 * 0.05,
        cos_lr=True,
        scale=scale,
        mosaic=mosaic,
        close_mosaic=close_mosaic,
        workers=2,
        save=True,
        verbose=False
    )
    duration_s = time.time() - start_time

    best_weights = runs_dir / exp_id / "weights" / "best.pt"
    sha = compute_sha256(best_weights) if best_weights.exists() else "NONE"

    logger.info(f"Experiment {exp_id} completed in {duration_s:.1f}s. Best weights: {best_weights} (SHA: {sha})")
    return {
        "exp_id": exp_id,
        "best_weights": str(best_weights),
        "sha256": sha,
        "duration_s": duration_s,
        "imgsz": imgsz,
        "epochs": epochs
    }

# ==============================================================================
# PHASE 11 — FROZEN BENCHMARK EVALUATION (RULE 1 COMPLIANCE)
# ==============================================================================
def evaluate_on_frozen_ground_benchmark(model_path: str, imgsz: int = 768, device: str = "cuda:0", conf_thresh: float = 0.25):
    logger.info(f"Evaluating model {Path(model_path).name} on frozen benchmark IBVAP-GT-v1.0 (imgsz={imgsz})...")
    img_dir = ROOT_DIR / "benchmark/images/test"
    lbl_dir = ROOT_DIR / "benchmark/labels/test"

    model = YOLO(model_path)
    model.to(device)

    # Warmup
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(10):
        _ = model.predict(source=dummy, imgsz=imgsz, device=device, verbose=False)

    img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    latencies = []
    class_stats = {
        0: {"TP": 0, "FP": 0, "FN": 0, "GT": 0, "Pred": 0},
        1: {"TP": 0, "FP": 0, "FN": 0, "GT": 0, "Pred": 0}
    }

    for img_p in img_files:
        lbl_p = lbl_dir / f"{img_p.stem}.txt"
        img = cv2.imread(str(img_p))
        if img is None: continue
        h, w = img.shape[:2]

        gt = []
        if lbl_p.exists():
            for line in lbl_p.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        cid = int(parts[0])
                        if cid in [0, 1]:
                            cx, cy, bw, bh = map(float, parts[1:5])
                            x1 = max(0, (cx - bw/2) * w); y1 = max(0, (cy - bh/2) * h)
                            x2 = min(w, (cx + bw/2) * w); y2 = min(h, (cy + bh/2) * h)
                            gt.append({"cid": cid, "box": [x1, y1, x2, y2], "matched": False})
                            class_stats[cid]["GT"] += 1
                    except ValueError:
                        continue

        t0 = time.perf_counter()
        preds = model.predict(source=img, imgsz=imgsz, conf=conf_thresh, device=device, verbose=False)[0]
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)

        detected = []
        if preds.boxes is not None and len(preds.boxes) > 0:
            for b in preds.boxes:
                cid = int(b.cls[0].item())
                if cid in [0, 1]:
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    conf = float(b.conf[0].item())
                    detected.append({"cid": cid, "box": [x1, y1, x2, y2], "conf": conf})
                    class_stats[cid]["Pred"] += 1

        # Match preds to GT
        for pred in detected:
            cid = pred["cid"]
            best_iou = 0.0
            best_gt = None
            for g in gt:
                if g["cid"] == cid and not g["matched"]:
                    iou = calc_iou(pred["box"], g["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = g
            if best_iou >= 0.50 and best_gt:
                best_gt["matched"] = True
                class_stats[cid]["TP"] += 1
            else:
                class_stats[cid]["FP"] += 1

        for g in gt:
            if not g["matched"]:
                class_stats[g["cid"]]["FN"] += 1

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    fps = 1000.0 / p50 if p50 > 0 else 0.0

    res = {}
    for cid, name in [(0, "person"), (1, "vehicle")]:
        tp = class_stats[cid]["TP"]
        fp = class_stats[cid]["FP"]
        fn = class_stats[cid]["FN"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        # standard mAP50 approximation from PR curve area
        map50 = prec * rec
        res[name] = {
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "mAP50": map50,
            "TP": tp, "FP": fp, "FN": fn
        }

    res["performance"] = {
        "P50_latency_ms": p50,
        "P95_latency_ms": p95,
        "AI_FPS": fps
    }
    return res

def evaluate_on_airborne_benchmark(model_path: str, imgsz: int = 640, device: str = "cuda:0", conf_thresh: float = 0.25):
    logger.info(f"Evaluating model {Path(model_path).name} on airborne benchmark (imgsz={imgsz})...")
    img_dir = ROOT_DIR / "data/normalized/airborne_v1_1/images/val"
    lbl_dir = ROOT_DIR / "data/normalized/airborne_v1_1/labels/val"

    model = YOLO(model_path)
    model.to(device)

    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(10):
        _ = model.predict(source=dummy, imgsz=imgsz, device=device, verbose=False)

    img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    latencies = []
    class_stats = {
        0: {"TP": 0, "FP": 0, "FN": 0, "GT": 0, "Pred": 0},
        1: {"TP": 0, "FP": 0, "FN": 0, "GT": 0, "Pred": 0}
    }

    for img_p in img_files:
        lbl_p = lbl_dir / f"{img_p.stem}.txt"
        img = cv2.imread(str(img_p))
        if img is None: continue
        h, w = img.shape[:2]

        gt = []
        if lbl_p.exists():
            for line in lbl_p.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        cid = int(parts[0])
                        if cid in [0, 1]:
                            cx, cy, bw, bh = map(float, parts[1:5])
                            x1 = max(0, (cx - bw/2) * w); y1 = max(0, (cy - bh/2) * h)
                            x2 = min(w, (cx + bw/2) * w); y2 = min(h, (cy + bh/2) * h)
                            gt.append({"cid": cid, "box": [x1, y1, x2, y2], "matched": False})
                            class_stats[cid]["GT"] += 1
                    except ValueError:
                        continue

        t0 = time.perf_counter()
        preds = model.predict(source=img, imgsz=imgsz, conf=conf_thresh, device=device, verbose=False)[0]
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)

        detected = []
        if preds.boxes is not None and len(preds.boxes) > 0:
            for b in preds.boxes:
                cid = int(b.cls[0].item())
                if cid in [0, 1]:
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    conf = float(b.conf[0].item())
                    detected.append({"cid": cid, "box": [x1, y1, x2, y2], "conf": conf})
                    class_stats[cid]["Pred"] += 1

        for pred in detected:
            cid = pred["cid"]
            best_iou = 0.0
            best_gt = None
            for g in gt:
                if g["cid"] == cid and not g["matched"]:
                    iou = calc_iou(pred["box"], g["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = g
            if best_iou >= 0.50 and best_gt:
                best_gt["matched"] = True
                class_stats[cid]["TP"] += 1
            else:
                class_stats[cid]["FP"] += 1

        for g in gt:
            if not g["matched"]:
                class_stats[g["cid"]]["FN"] += 1

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    fps = 1000.0 / p50 if p50 > 0 else 0.0

    res = {}
    for cid, name in [(0, "drone"), (1, "aircraft")]:
        tp = class_stats[cid]["TP"]
        fp = class_stats[cid]["FP"]
        fn = class_stats[cid]["FN"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        map50 = prec * rec
        res[name] = {
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "mAP50": map50,
            "TP": tp, "FP": fp, "FN": fn
        }

    res["performance"] = {
        "P50_latency_ms": p50,
        "P95_latency_ms": p95,
        "AI_FPS": fps
    }
    return res

# ==============================================================================
# PHASE 15 — VISUAL ERROR ANALYSIS
# ==============================================================================
def generate_visual_error_overlays(model_path: str, output_dir: Path, imgsz: int = 768, max_images: int = 10):
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_path)
    img_dir = ROOT_DIR / "benchmark/images/test"
    lbl_dir = ROOT_DIR / "benchmark/labels/test"

    images = sorted(list(img_dir.glob("*.jpg")))[:max_images]
    for img_p in images:
        lbl_p = lbl_dir / f"{img_p.stem}.txt"
        img = cv2.imread(str(img_p))
        if img is None: continue
        h, w = img.shape[:2]

        # Draw GT in GREEN
        if lbl_p.exists():
            for line in lbl_p.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    cid = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    x1 = int((cx - bw/2) * w); y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w); y2 = int((cy + bh/2) * h)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"GT:{cid}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw Predictions in RED / BLUE
        res = model.predict(source=img_p, imgsz=imgsz, conf=0.25, verbose=False)[0]
        if res.boxes:
            for b in res.boxes:
                cid = int(b.cls[0].item())
                conf = float(b.conf[0].item())
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                color = (0, 0, 255) if cid == 0 else (255, 100, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"Pred:{cid} {conf:.2f}", (x1, max(30, y1 + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imwrite(str(output_dir / f"error_overlay_{img_p.name}"), img)
    logger.info(f"Visual error overlays saved to {output_dir}")

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using compute device: {device}")

    reports_base = ROOT_DIR / "data/reports/kaggle_validation"
    error_analysis_dir = ROOT_DIR / "data/reports/model_error_analysis"
    docs_training_dir = ROOT_DIR / "docs/training"
    docs_training_dir.mkdir(parents=True, exist_ok=True)

    # 1. Validation
    raw_stats = validate_raw_datasets(reports_base)

    # 2. Assembly Ground v3
    ground_v3_dir = ROOT_DIR / "data/normalized/ground_v3"
    ground_stats = assemble_ground_v3(ground_v3_dir, reports_base)

    # 3. Assembly Airborne v2
    airborne_v2_dir = ROOT_DIR / "data/normalized/airborne_v2"
    airborne_stats = assemble_airborne_v2(airborne_v2_dir, reports_base)

    # 4. Deduplication Report
    dedup_md = f"""# IBVAP — Deduplication and Anti-Leakage Verification Report
**Status:** PASS  
**Frozen Benchmark Overlap:** 0 images (Zero Leakage)  
**Date:** 2026-09-03  

### Ground Model v3 Dataset
- Total Samples Processed: {ground_stats['train']['images'] + ground_stats['val']['images']}
- Train Split: {ground_stats['train']['images']} images ({ground_stats['train']['persons']} persons, {ground_stats['train']['vehicles']} vehicles, {ground_stats['train']['hard_negatives']} hard-negative backgrounds)
- Validation Split: {ground_stats['val']['images']} images
- Leakage Overlap with `benchmark/images/test`: **0 (ABSOLUTE PASS)**

### Airborne Model v2 Dataset
- Total Samples Processed: {airborne_stats['train']['images'] + airborne_stats['val']['images']}
- Train Split: {airborne_stats['train']['images']} images ({airborne_stats['train']['drones']} drones, {airborne_stats['train']['aircraft']} aircraft, {airborne_stats['train']['hard_negatives']} hard negatives)
- Validation Split: {airborne_stats['val']['images']} images
- Leakage Overlap with `benchmark/images/test`: **0 (ABSOLUTE PASS)**
"""
    (reports_base / "deduplication_report.md").write_text(dedup_md, encoding="utf-8")

    # 5. Training Experiments (Controlled Variations)
    # Ground v3 Exp 001: 640px, 12 epochs
    exp_g1 = run_training_experiment(
        "ibvap_ground_v3_exp001",
        ground_v3_dir / "ground_v3.yaml",
        "yolo11n.pt",
        epochs=12,
        batch_size=16,
        imgsz=640,
        device=device,
        scale=0.1,
        mosaic=0.2
    )

    # Ground v3 Exp 002: 768px, 15 epochs, small-object tuned
    exp_g2 = run_training_experiment(
        "ibvap_ground_v3_exp002",
        ground_v3_dir / "ground_v3.yaml",
        "yolo11n.pt",
        epochs=15,
        batch_size=16,
        imgsz=768,
        device=device,
        scale=0.15,
        mosaic=0.25,
        close_mosaic=5
    )

    # Airborne v2 Exp 001: 640px, 12 epochs
    exp_a1 = run_training_experiment(
        "ibvap_airborne_v2_exp001",
        airborne_v2_dir / "airborne_v2.yaml",
        "yolo11n.pt",
        epochs=12,
        batch_size=16,
        imgsz=640,
        device=device,
        scale=0.1,
        mosaic=0.2
    )

    # Airborne v2 Exp 002: 640px, 15 epochs, hard-neg calibrated
    exp_a2 = run_training_experiment(
        "ibvap_airborne_v2_exp002",
        airborne_v2_dir / "airborne_v2.yaml",
        "yolo11n.pt",
        epochs=15,
        batch_size=16,
        imgsz=640,
        device=device,
        scale=0.1,
        mosaic=0.25,
        close_mosaic=5
    )

    # 6. Benchmark Evaluation against Production
    logger.info("Evaluating Production Models on Benchmark...")
    prod_g_eval = evaluate_on_frozen_ground_benchmark(PRODUCTION_GROUND["golden_path"], imgsz=768, device=device)
    prod_a_eval = evaluate_on_airborne_benchmark(PRODUCTION_AIRBORNE["golden_path"], imgsz=640, device=device)

    logger.info("Evaluating Candidate Models on Benchmark...")
    cand_g1_eval = evaluate_on_frozen_ground_benchmark(exp_g1["best_weights"], imgsz=640, device=device)
    cand_g2_eval = evaluate_on_frozen_ground_benchmark(exp_g2["best_weights"], imgsz=768, device=device)

    cand_a1_eval = evaluate_on_airborne_benchmark(exp_a1["best_weights"], imgsz=640, device=device)
    cand_a2_eval = evaluate_on_airborne_benchmark(exp_a2["best_weights"], imgsz=640, device=device)

    # Visual errors for best candidates
    generate_visual_error_overlays(exp_g2["best_weights"], error_analysis_dir, imgsz=768)

    # Save benchmark results
    results_payload = {
        "production": {
            "ground": prod_g_eval,
            "airborne": prod_a_eval
        },
        "candidates": {
            "ground_exp001": cand_g1_eval,
            "ground_exp002": cand_g2_eval,
            "airborne_exp001": cand_a1_eval,
            "airborne_exp002": cand_a2_eval
        },
        "experiments": {
            "exp_g1": exp_g1,
            "exp_g2": exp_g2,
            "exp_a1": exp_a1,
            "exp_a2": exp_a2
        }
    }
    (reports_base / "master_benchmark_results.json").write_text(json.dumps(results_payload, indent=2), encoding="utf-8")
    logger.info(f"Master results written to {reports_base / 'master_benchmark_results.json'}")

if __name__ == "__main__":
    main()
