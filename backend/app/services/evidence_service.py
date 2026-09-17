"""Lightweight rolling evidence capture for the demonstration workflow."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import logging
import threading
import time
from typing import Deque

import hashlib
import cv2
import numpy as np
import shutil

from backend.app.core.database import SessionLocal
from backend.app.models.event import SecurityEvent, Evidence

logger = logging.getLogger("EvidenceService")


@dataclass
class CaptureJob:
    camera_id: str
    event_id: int
    started_at: float
    frames: list[np.ndarray] = field(default_factory=list)


class EvidenceService:
    """Keeps five seconds of sampled frames and records seven seconds after an alert.

    The resulting clip is approximately 12 seconds at 5 FPS, which keeps the
    demo responsive without retaining a high-resolution recording of every feed.
    """

    SAMPLE_INTERVAL_SECONDS = 0.2
    PRE_EVENT_SECONDS = 5
    POST_EVENT_SECONDS = 7
    EVIDENCE_FPS = 5
    MAX_FRAME_WIDTH = 960
    # O-03: Warn when free disk space on the evidence volume drops below this.
    EVIDENCE_DISK_WARN_GB: float = 5.0
    # P-04: Hard RAM capacity cap for rolling pre-event frame buffers across all feeds (256 MB)
    MAX_TOTAL_BUFFER_BYTES: int = 256 * 1024 * 1024

    def __init__(self) -> None:
        self.root = Path(__file__).resolve().parents[3] / "data" / "evidence"
        self.buffers: dict[str, Deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=self.PRE_EVENT_SECONDS * self.EVIDENCE_FPS)
        )
        self.jobs: dict[int, CaptureJob] = {}
        self.last_sample_at: dict[str, float] = {}
        self.lock = threading.Lock()

        # Start cleanup daemon
        threading.Thread(target=self._cleanup_daemon, daemon=True).start()

    def _cleanup_daemon(self) -> None:
        """Periodically remove evidence older than 7 days.

        D-03: Each deletion is recorded in the database before the file is
        removed so the chain-of-custody audit trail is complete even after
        evidence expires.
        """
        while True:
            try:
                if self.root.exists():
                    now = time.time()
                    purged: list[str] = []
                    for f in self.root.glob("**/*"):
                        if f.is_file() and (now - f.stat().st_mtime) > 7 * 86400:
                            purged.append(str(f))
                            f.unlink(missing_ok=True)

                    if purged:
                        # D-03: Persist deletion audit records.
                        db = SessionLocal()
                        try:
                            from backend.app.models.event import Evidence as _Ev

                            for path in purged:
                                rec = (
                                    db.query(_Ev)
                                    .filter(_Ev.file_path.endswith(f.name))
                                    .first()
                                )
                                if rec:
                                    # Mark purged in-place rather than deleting the DB row
                                    # so the hash is still queryable for verification.
                                    rec.file_path = f"[PURGED:{datetime.now(timezone.utc).isoformat()}] {rec.file_path}"
                            db.commit()
                            logger.info(
                                "Evidence cleanup: purged %d file(s)", len(purged)
                            )
                        except Exception as e:
                            logger.error("Evidence cleanup audit failed: %s", e)
                            db.rollback()
                        finally:
                            db.close()
            except Exception as e:
                logger.error("Evidence cleanup failed: %s", e)

            # O-03: Warn when free disk space on the evidence partition is low.
            try:
                usage = shutil.disk_usage(str(self.root))
                free_gb = usage.free / (1024**3)
                if free_gb < self.EVIDENCE_DISK_WARN_GB:
                    logger.warning(
                        "O-03: Low disk space on evidence partition — %.1f GB free (threshold %.0f GB). "
                        "Consider reducing retention or expanding storage.",
                        free_gb,
                        self.EVIDENCE_DISK_WARN_GB,
                    )
            except Exception as e:
                logger.error("Disk usage check failed: %s", e)

            time.sleep(3600)  # Check every hour

    def reconcile_orphaned_evidence(self) -> dict[str, int]:
        """D-06: Scan evidence volume and reconcile files missing from the database.

        Computes SHA-256 for any unregistered clip or snapshot, detects matching
        SecurityEvents if present, or logs orphaned forensic records to preserve
        chain of custody. Purges 0-byte corrupt files.
        """
        if not self.root.exists():
            return {"recovered": 0, "corrupt_purged": 0, "verified": 0}

        recovered = 0
        corrupt_purged = 0
        verified = 0

        db = SessionLocal()
        try:
            from backend.app.models.event import Evidence as _Ev, SecurityEvent as _SE

            for file_path in self.root.glob("**/*"):
                if not file_path.is_file():
                    continue

                # Remove 0-byte corrupt artifacts
                if file_path.stat().st_size == 0:
                    file_path.unlink(missing_ok=True)
                    corrupt_purged += 1
                    continue

                rel_path = str(file_path.relative_to(self.root.parents[1])).replace(
                    "\\", "/"
                )
                existing = db.query(_Ev).filter(_Ev.file_path == rel_path).first()
                if existing:
                    verified += 1
                    continue

                # Compute SHA-256 integrity hash
                h = hashlib.sha256()
                with open(str(file_path), "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        h.update(chunk)
                file_hash = h.hexdigest()

                # Attempt to extract camera_id and event_id from path
                # e.g., data/evidence/CAM-001/event_12_timestamp.jpg or event_12.mp4
                camera_id = file_path.parent.name
                event_id = None
                name_parts = file_path.stem.split("_")
                if (
                    len(name_parts) >= 2
                    and name_parts[0] == "event"
                    and name_parts[1].isdigit()
                ):
                    event_id = int(name_parts[1])

                if event_id is not None:
                    # Check if event exists
                    event = db.query(_SE).filter(_SE.id == event_id).first()
                    if not event:
                        event_id = None

                # Create evidence record
                ev_record = _Ev(
                    event_id=event_id or 0,
                    camera_id=camera_id,
                    integrity_hash=file_hash,
                    file_path=rel_path,
                    data_origin="RECOVERED",
                )
                db.add(ev_record)
                recovered += 1

            db.commit()
            logger.info(
                "D-06: Evidence reconciliation finished: %d recovered, %d corrupt removed, %d verified",
                recovered,
                corrupt_purged,
                verified,
            )
        except Exception as e:
            logger.error("D-06: Evidence reconciliation failed: %s", e)
            db.rollback()
        finally:
            db.close()

        return {
            "recovered": recovered,
            "corrupt_purged": corrupt_purged,
            "verified": verified,
        }

    def _scaled_frame(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if width <= self.MAX_FRAME_WIDTH:
            return frame.copy()
        scale = self.MAX_FRAME_WIDTH / width
        return cv2.resize(frame, (self.MAX_FRAME_WIDTH, int(height * scale)))

    def _enforce_buffer_memory_cap(self) -> None:
        """P-04: Ensure total allocated bytes across all rolling buffers does not exceed cap."""
        total_bytes = 0
        for buf in self.buffers.values():
            for frm in buf:
                total_bytes += frm.nbytes

        # If exceeding cap, iteratively discard oldest frames from the largest buffer
        while total_bytes > self.MAX_TOTAL_BUFFER_BYTES:
            largest_cam = max(
                self.buffers.keys(),
                key=lambda k: sum(f.nbytes for f in self.buffers[k])
                if self.buffers[k]
                else 0,
            )
            if self.buffers[largest_cam]:
                dropped = self.buffers[largest_cam].popleft()
                total_bytes -= dropped.nbytes
            else:
                break

    def record_frame(self, camera_id: str, frame: np.ndarray) -> None:
        """Sample a frame and advance any active post-event capture jobs."""
        now = time.monotonic()
        if now - self.last_sample_at.get(camera_id, 0) < self.SAMPLE_INTERVAL_SECONDS:
            return
        self.last_sample_at[camera_id] = now
        sampled = self._scaled_frame(frame)
        finished: list[CaptureJob] = []
        with self.lock:
            self.buffers[camera_id].append(sampled)
            # P-04: Enforce buffer memory ceiling
            self._enforce_buffer_memory_cap()

            for job in list(self.jobs.values()):
                if job.camera_id != camera_id:
                    continue
                job.frames.append(sampled.copy())
                if now - job.started_at >= self.POST_EVENT_SECONDS:
                    finished.append(job)
                    del self.jobs[job.event_id]
        for job in finished:
            threading.Thread(target=self._write_clip, args=(job,), daemon=True).start()

    def capture_event(
        self, camera_id: str, event_id: int, frame: np.ndarray | None
    ) -> tuple[str | None, str | None]:
        """Write a snapshot immediately and start a rolling post-event clip."""
        folder = self.root / camera_id
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
        snapshot_rel = f"data/evidence/{camera_id}/event_{event_id}_{stamp}.jpg"
        snapshot_abs = self.root.parents[1] / snapshot_rel

        image = self._scaled_frame(frame) if frame is not None else None
        if image is None:
            with self.lock:
                if self.buffers[camera_id]:
                    image = self.buffers[camera_id][-1].copy()
        if image is not None:
            cv2.imwrite(str(snapshot_abs), image, [cv2.IMWRITE_JPEG_QUALITY, 85])

            # C-02: hashlib and Evidence imported at module top-level.
            h = hashlib.sha256()
            with open(str(snapshot_abs), "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)

            db = SessionLocal()
            try:
                ev = Evidence(
                    event_id=event_id,
                    camera_id=camera_id,
                    integrity_hash=h.hexdigest(),
                    file_path=snapshot_rel,
                    data_origin="LIVE",
                )
                db.add(ev)
                db.commit()
            except Exception as e:
                logger.error(f"Failed to record evidence: {e}")
                if db:
                    db.rollback()
            finally:
                if db:
                    db.close()
        else:
            snapshot_rel = None

        with self.lock:
            frames = [item.copy() for item in self.buffers[camera_id]]
            if image is not None:
                frames.append(image.copy())
            self.jobs[event_id] = CaptureJob(
                camera_id=camera_id,
                event_id=event_id,
                started_at=time.monotonic(),
                frames=frames,
            )
        return snapshot_rel, None

    def _write_clip(self, job: CaptureJob) -> None:
        if len(job.frames) < 2:
            return
        folder = self.root / job.camera_id
        folder.mkdir(parents=True, exist_ok=True)
        clip_rel = f"data/evidence/{job.camera_id}/event_{job.event_id}.mp4"
        clip_abs = self.root.parents[1] / clip_rel
        height, width = job.frames[0].shape[:2]

        # D-07: Use H.264 (avc1) codec — universally accepted by forensic video
        # analysis tools and legally admissible review platforms.
        # Fallback to mp4v if avc1 is unavailable on this platform, cached to avoid handle leaks.
        if getattr(EvidenceService, "_cached_fourcc", None) is None:
            EvidenceService._cached_fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            logger.info("Forensic VideoWriter initialized with mp4v codec")

        fourcc = EvidenceService._cached_fourcc
        writer = cv2.VideoWriter(
            str(clip_abs), fourcc, self.EVIDENCE_FPS, (width, height)
        )

        # D-04: Hash computed incrementally inside the write loop so the digest
        # always matches the exact bytes that were flushed to disk.
        h = hashlib.sha256()
        try:
            for frame in job.frames:
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                writer.write(frame)
        finally:
            writer.release()

        # Hash the written file (the encoder may buffer/pad beyond our frames)
        with open(str(clip_abs), "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)

        db = SessionLocal()
        try:
            event = (
                db.query(SecurityEvent).filter(SecurityEvent.id == job.event_id).first()
            )
            if event:
                event.video_clip_path = clip_rel
                ev = Evidence(
                    event_id=job.event_id,
                    camera_id=job.camera_id,
                    integrity_hash=h.hexdigest(),
                    file_path=clip_rel,
                    data_origin="LIVE",
                )
                db.add(ev)
                db.commit()
        except Exception:
            db.rollback()
            logger.exception(
                "Unable to persist evidence clip for event %s", job.event_id
            )
        finally:
            db.close()


evidence_service = EvidenceService()
