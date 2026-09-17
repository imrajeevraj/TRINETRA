#!/usr/bin/env python3
"""
IBVAP — Track A: Safe Airborne Model Promotion Script
Executes atomic promotion of ibvap_airborne_v2_exp002 with rollback preservation.
"""

import os
import sys
import shutil
import hashlib
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.ops.master_ai_pipeline import compute_sha256

def main():
    candidate_src = ROOT_DIR / "data/training/runs/ibvap_airborne_v2_exp002/weights/best.pt"
    expected_sha = "5229632C3D7A12A279A45032D558FB4712BF9DB43667828D30367D2354106384"
    assert candidate_src.exists(), f"Source candidate missing: {candidate_src}"
    actual_sha = compute_sha256(candidate_src)
    assert actual_sha.upper() == expected_sha.upper(), f"SHA mismatch on source candidate: {actual_sha}"

    # 1. Preserve old production & create rollback checkpoint
    v1_golden = ROOT_DIR / "models/production/airborne/ibvap_airborne_v1_production.pt"
    rollback_dst = ROOT_DIR / "models/production/airborne/rollback_v1_production.pt"
    if v1_golden.exists():
        shutil.copy2(str(v1_golden), str(rollback_dst))
        print(f"[PRESERVED] Created rollback checkpoint at {rollback_dst}")

    # 2. Copy candidate to production destination
    v2_dst = ROOT_DIR / "models/production/airborne/ibvap_airborne_v2_production.pt"
    shutil.copy2(str(candidate_src), str(v2_dst))
    copied_sha = compute_sha256(v2_dst)
    assert copied_sha.upper() == expected_sha.upper(), f"SHA mismatch on copied destination: {copied_sha}"
    print(f"[PROMOTED] Copied candidate to {v2_dst} (SHA-256 verified: {copied_sha})")

    # Also place at models/current/ibvap_airborne_detector.pt for standardized active deployment
    current_dst = ROOT_DIR / "models/current/ibvap_airborne_detector.pt"
    shutil.copy2(str(candidate_src), str(current_dst))
    print(f"[DEPLOYED] Copied candidate to active path {current_dst}")

if __name__ == "__main__":
    main()
