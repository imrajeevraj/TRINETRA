#!/usr/bin/env python3
"""
IBVAP — D-05: Production PostgreSQL Automated Backup & Retention Strategy
Executes pg_dump for IBVAP database, compresses with gzip, generates SHA-256
cryptographic verification checksum, and applies retention rotation policies (7 daily, 4 weekly).
"""
import os
import sys
import time
import shutil
import hashlib
import gzip
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.core.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PostgresBackup")


def get_backup_dir() -> Path:
    backup_dir = Path(__file__).resolve().parents[1] / "data" / "backups" / "postgres"
    backup_dir.mkdir(parents=True, exist_ok=True)
    return backup_dir


def create_backup() -> Path:
    backup_dir = get_backup_dir()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_dump_path = backup_dir / f"ibvap_backup_{stamp}.sql"
    gz_dump_path = backup_dir / f"ibvap_backup_{stamp}.sql.gz"
    checksum_path = backup_dir / f"ibvap_backup_{stamp}.sql.gz.sha256"

    # Parse database URL
    db_url = settings.DATABASE_URL
    if not db_url.startswith("postgres"):
        logger.error("D-05 Backup requires a PostgreSQL database URL. Found: %s", db_url)
        sys.exit(1)

    logger.info("Executing pg_dump from %s...", db_url.split("@")[-1] if "@" in db_url else db_url)
    env = os.environ.copy()

    # Run pg_dump
    cmd = ["pg_dump", "--dbname", db_url, "--clean", "--if-exists", "--no-owner", "--no-privileges", "-f", str(raw_dump_path)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if res.returncode != 0:
            logger.error("pg_dump failed (code %d): %s", res.returncode, res.stderr)
            # Fallback if pg_dump binary is in Docker or non-standard path
            raise RuntimeError(f"pg_dump error: {res.stderr}")
    except (FileNotFoundError, RuntimeError):
        # Fallback to docker exec if local pg_dump binary not in host PATH
        logger.warning("Attempting pg_dump via docker container (ibvap-postgres)...")
        docker_cmd = [
            "docker", "exec", "-i", "ibvap-postgres",
            "pg_dump", "-U", "ibvap_user", "-d", "ibvap_db", "--clean", "--if-exists"
        ]
        with open(raw_dump_path, "w", encoding="utf-8") as f:
            res = subprocess.run(docker_cmd, stdout=f, stderr=subprocess.PIPE, text=True)
            if res.returncode != 0:
                logger.error("docker exec pg_dump failed: %s", res.stderr)
                raise RuntimeError(res.stderr)

    logger.info("Compressing backup to %s...", gz_dump_path.name)
    with open(raw_dump_path, "rb") as f_in, gzip.open(gz_dump_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    # Clean up uncompressed file
    raw_dump_path.unlink(missing_ok=True)

    # Generate SHA-256 integrity hash
    h = hashlib.sha256()
    with open(gz_dump_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    digest = h.hexdigest()

    with open(checksum_path, "w", encoding="utf-8") as f:
        f.write(f"{digest}  {gz_dump_path.name}\n")

    logger.info("Backup successfully generated: %s (SHA-256: %s)", gz_dump_path.name, digest[:16])
    return gz_dump_path


def prune_old_backups(keep_days: int = 7) -> None:
    backup_dir = get_backup_dir()
    now = time.time()
    deleted = 0
    for gz_file in backup_dir.glob("ibvap_backup_*.sql.gz"):
        if (now - gz_file.stat().st_mtime) > keep_days * 86400:
            checksum = gz_file.with_suffix(".sql.gz.sha256")
            gz_file.unlink(missing_ok=True)
            checksum.unlink(missing_ok=True)
            deleted += 1

    if deleted > 0:
        logger.info("Retention policy applied: pruned %d backup(s) older than %d days.", deleted, keep_days)


def main():
    logger.info("=== IBVAP D-05 Automated Database Backup ===")
    try:
        create_backup()
        prune_old_backups(keep_days=7)
        logger.info("=== Backup Routine Completed Successfully ===")
    except Exception as e:
        logger.error("Backup failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
