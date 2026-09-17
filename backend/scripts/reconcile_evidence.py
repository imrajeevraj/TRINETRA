#!/usr/bin/env python3
"""
TRINETRA — D-06: Forensic Evidence Reconciliation Tool
Scans data/evidence directory, recalculates cryptographic SHA-256 hashes,
associates orphaned media files with PostgreSQL security event records, and
removes 0-byte corrupted clips.
"""
import sys
import os
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.services.evidence_service import evidence_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EvidenceReconciler")


def main():
    logger.info("Starting TRINETRA Forensic Evidence Volume Reconciliation...")
    result = evidence_service.reconcile_orphaned_evidence()
    logger.info(
        "Reconciliation Complete: %d orphaned files recovered, %d verified existing, %d corrupt removed.",
        result["recovered"],
        result["verified"],
        result["corrupt_purged"],
    )


if __name__ == "__main__":
    main()
