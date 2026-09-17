#!/usr/bin/env python3
"""
IBVAP — YOLO26 Rollback Utility
One-command rollback: restores YOLO11 production checkpoint.

Usage:
  python scripts/yolo26/rollback.py --domain ground
  python scripts/yolo26/rollback.py --domain airborne
  python scripts/yolo26/rollback.py --domain security_item
  python scripts/yolo26/rollback.py --domain all

Rollback mechanism:
  1. Reads registry to find rollback_target for the domain
  2. Locates rollback checkpoint in models/production/
  3. SHA-256 verifies the rollback checkpoint
  4. Updates models/current/ symlink / active config pointer
  5. Logs rollback event
  6. Does NOT delete the YOLO26 candidate — it remains in models/candidates/
"""

import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("IBVAP_Rollback")

ROLLBACK_TARGETS = {
    "ground": {
        "rollback_checkpoint": "models/production/ground/ibvap_ground_v2_production.pt",
        "active_path": "models/current/ibvap_detector.pt",
        "expected_sha256": "7DBF36027768194F61C6EBBC000E1421374624558168F8A9896F727B296277B0",
        "rollback_to_id": "ibvap-ground-v2",
        "rollback_to_version": "v2.0",
    },
    "airborne": {
        "rollback_checkpoint": "models/production/airborne/ibvap_airborne_v1_production.pt",
        "active_path": "data/training/runs/ibvap_airborne_v1_exp001/weights/best.pt",
        "expected_sha256": "E1009633325E463B1C0D45D6578522A1A2026AE27B854591A83461043928C914",
        "rollback_to_id": "ibvap-airborne-v1",
        "rollback_to_version": "v1.1",
    },
    "security_item": {
        "rollback_checkpoint": "models/production/security_item/ibvap_security_item_v2_1_production.pt",
        "active_path": "data/training/runs/ibvap_security_item_v2_1_exp001/weights/best.pt",
        "expected_sha256": "72464C778DE57270800146AB5ADF2AB683FFEAC55338C89231EE3A83E7F6E1DA",
        "rollback_to_id": "ibvap-security-item-v2.1-exp001",
        "rollback_to_version": "v2.1",
    },
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()


def rollback_domain(domain: str, dry_run: bool = False) -> dict:
    logger.info("=" * 60)
    logger.info(f"ROLLBACK: {domain.upper()}")

    cfg = ROLLBACK_TARGETS.get(domain)
    if not cfg:
        logger.error(f"Unknown domain: {domain}")
        return {"domain": domain, "status": "ERROR", "error": "unknown domain"}

    rollback_ckpt = Path(cfg["rollback_checkpoint"])
    if not rollback_ckpt.exists():
        logger.critical(f"Rollback checkpoint not found: {rollback_ckpt}")
        return {"domain": domain, "status": "FAILED", "error": "rollback checkpoint missing"}

    # SHA-256 verification
    actual_sha = sha256_file(rollback_ckpt)
    expected_sha = cfg["expected_sha256"]
    if actual_sha != expected_sha:
        logger.critical(f"SHA-256 MISMATCH! Rollback aborted.")
        logger.critical(f"Expected: {expected_sha}")
        logger.critical(f"Actual:   {actual_sha}")
        return {"domain": domain, "status": "FAILED", "error": "sha256_mismatch",
                "expected": expected_sha, "actual": actual_sha}

    logger.info(f"SHA-256 verified: {actual_sha[:32]}...")
    active_path = Path(cfg["active_path"])

    if dry_run:
        logger.info(f"[DRY RUN] Would restore: {rollback_ckpt} → {active_path}")
        return {"domain": domain, "status": "DRY_RUN_OK", "sha256": actual_sha}

    # Backup current (possibly YOLO26) weights before overwrite
    if active_path.exists():
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup_dir = Path("models/archive/pre_rollback_backups")
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / f"{domain}_pre_rollback_{ts}.pt"
        shutil.copy2(str(active_path), str(backup_path))
        logger.info(f"Backed up current weights to: {backup_path}")

    # Restore
    active_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(rollback_ckpt), str(active_path))
    logger.info(f"Restored: {rollback_ckpt} → {active_path}")

    # Verify restore
    restored_sha = sha256_file(active_path)
    if restored_sha != expected_sha:
        logger.critical(f"Post-restore SHA-256 mismatch! Active checkpoint may be corrupt.")
        return {"domain": domain, "status": "FAILED", "error": "post_restore_mismatch"}

    result = {
        "domain": domain,
        "status": "SUCCESS",
        "rollback_to": cfg["rollback_to_id"],
        "rollback_to_version": cfg["rollback_to_version"],
        "restored_from": str(rollback_ckpt),
        "restored_to": str(active_path),
        "sha256": restored_sha,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    logger.info(f"Rollback {domain} → {cfg['rollback_to_id']} ({cfg['rollback_to_version']}): SUCCESS")
    return result


def main():
    parser = argparse.ArgumentParser(description="IBVAP One-Command Rollback Utility")
    parser.add_argument("--domain", choices=["ground", "airborne", "security_item", "all"], required=True,
                        help="Which detector to roll back")
    parser.add_argument("--dry-run", action="store_true", help="Verify only, no writes")
    args = parser.parse_args()

    domains = list(ROLLBACK_TARGETS.keys()) if args.domain == "all" else [args.domain]

    results = {}
    overall = True
    for d in domains:
        r = rollback_domain(d, dry_run=args.dry_run)
        results[d] = r
        if r["status"] not in ("SUCCESS", "DRY_RUN_OK"):
            overall = False

    # Log rollback event
    event_log = Path("data/reports/yolo26/rollback_log.json")
    existing = []
    if event_log.exists():
        try:
            existing = json.loads(event_log.read_text(encoding="utf-8"))
        except Exception:
            pass
    existing.append({
        "event_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dry_run": args.dry_run,
        "results": results,
    })
    event_log.parent.mkdir(parents=True, exist_ok=True)
    event_log.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    logger.info("=" * 60)
    for d, r in results.items():
        icon = "✅" if r["status"] in ("SUCCESS", "DRY_RUN_OK") else "❌"
        logger.info(f"  {icon} {d}: {r['status']}")
    if not overall:
        logger.critical("One or more rollbacks FAILED. Investigate immediately.")
        sys.exit(1)
    else:
        logger.info("Rollback complete." + (" (DRY RUN — no changes made)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
