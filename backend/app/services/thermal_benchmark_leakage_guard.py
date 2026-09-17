"""
TRINETRA Phase XV — Thermal Benchmark Leakage Guard
Prevents any benchmark thermal frame or near-duplicate frame from leaking into:
training, validation splits, hyperparameter tuning, or active learning queues.

Implements 5 layers of defense:
1. Exact SHA-256 matching against quarantined benchmark hashes
2. Sequence-level temporal window checks (blocks frame N if frame N+1 is in benchmark)
3. Source-video / sequence ID isolation
4. Perceptual hash (dHash) hamming distance checks
5. Hard failure with BENCHMARK_LEAKAGE_REJECTED exception
"""

from __future__ import annotations
import math
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any

logger = logging.getLogger("ThermalBenchmarkLeakageGuard")


class BenchmarkLeakageError(Exception):
    """Raised when any benchmark frame or near-duplicate attempts to enter training."""
    pass


@dataclass
class QuarantinedThermalSample:
    benchmark_id: str
    sample_id: str
    sha256_hash: str
    source_video_id: str
    sequence_id: str
    timestamp: float
    perceptual_hash: int  # 64-bit integer hash


class ThermalBenchmarkLeakageGuard:
    """
    Guarantees absolute independence of the thermal ground truth benchmark.
    Fails closed with BENCHMARK_LEAKAGE_REJECTED on any leakage violation.
    """

    def __init__(self, sequence_window_s: float = 5.0, phash_distance_threshold: int = 5):
        self.sequence_window_s = sequence_window_s
        self.phash_distance_threshold = phash_distance_threshold
        self._quarantined_samples: Dict[str, QuarantinedThermalSample] = {}
        self._quarantined_shas: Set[str] = set()
        self._quarantined_videos: Set[str] = set()

    def register_benchmark_sample(
        self,
        benchmark_id: str,
        sample_id: str,
        sha256_hash: str,
        source_video_id: str,
        sequence_id: str,
        timestamp: float,
        payload_bytes: Optional[bytes] = None,
    ) -> QuarantinedThermalSample:
        """Adds a sample to the permanently quarantined benchmark registry."""
        phash = self._compute_simple_dhash(payload_bytes) if payload_bytes else int(sha256_hash[:16], 16)
        item = QuarantinedThermalSample(
            benchmark_id=benchmark_id,
            sample_id=sample_id,
            sha256_hash=sha256_hash.upper(),
            source_video_id=source_video_id,
            sequence_id=sequence_id,
            timestamp=timestamp,
            perceptual_hash=phash,
        )
        self._quarantined_samples[sample_id] = item
        self._quarantined_shas.add(item.sha256_hash)
        if source_video_id:
            self._quarantined_videos.add(source_video_id)
        logger.info(f"Quarantined benchmark sample {sample_id} [SHA: {sha256_hash[:12]}...]")
        return item

    def audit_candidate_sample(
        self,
        candidate_id: str,
        sha256_hash: str,
        source_video_id: Optional[str] = None,
        sequence_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        payload_bytes: Optional[bytes] = None,
    ) -> Tuple[bool, str]:
        """
        Audits a training/candidate sample against all quarantined benchmark samples.
        Returns (is_clean, reason). If leakage is found, raises or returns False.
        """
        cand_sha = sha256_hash.upper()

        # 1. Exact SHA-256 check
        if cand_sha in self._quarantined_shas:
            msg = f"BENCHMARK_LEAKAGE_REJECTED: Exact SHA-256 match found with quarantined benchmark sample."
            logger.error(f"Sample {candidate_id} REJECTED: {msg}")
            return False, msg

        # 2. Source-video / sequence ID check
        if source_video_id and source_video_id in self._quarantined_videos:
            msg = f"BENCHMARK_LEAKAGE_REJECTED: Candidate originates from quarantined benchmark video source ({source_video_id})."
            logger.error(f"Sample {candidate_id} REJECTED: {msg}")
            return False, msg

        # 3. Sequence-level temporal window check (frame N vs frame N+1)
        if sequence_id and timestamp is not None:
            for q_sample in self._quarantined_samples.values():
                if q_sample.sequence_id == sequence_id:
                    delta_t = abs(timestamp - q_sample.timestamp)
                    if delta_t < self.sequence_window_s:
                        msg = (
                            f"BENCHMARK_LEAKAGE_REJECTED: Sequence-level temporal leakage! Candidate timestamp "
                            f"{timestamp:.2f} is within {delta_t:.3f}s (< {self.sequence_window_s}s) of benchmark sample {q_sample.sample_id}."
                        )
                        logger.error(f"Sample {candidate_id} REJECTED: {msg}")
                        return False, msg

        # 4. Perceptual hash check (near-duplicate detection)
        if payload_bytes:
            cand_phash = self._compute_simple_dhash(payload_bytes)
            for q_sample in self._quarantined_samples.values():
                distance = bin(cand_phash ^ q_sample.perceptual_hash).count("1")
                if distance <= self.phash_distance_threshold:
                    msg = (
                        f"BENCHMARK_LEAKAGE_REJECTED: Near-duplicate detected! Perceptual hash distance {distance} "
                        f"<= threshold {self.phash_distance_threshold} to benchmark sample {q_sample.sample_id}."
                    )
                    logger.error(f"Sample {candidate_id} REJECTED: {msg}")
                    return False, msg

        return True, "CLEAN_NO_LEAKAGE_DETECTED"

    def assert_no_leakage(
        self,
        candidate_id: str,
        sha256_hash: str,
        source_video_id: Optional[str] = None,
        sequence_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        payload_bytes: Optional[bytes] = None,
    ):
        """Helper that raises BenchmarkLeakageError immediately upon leakage."""
        is_clean, reason = self.audit_candidate_sample(
            candidate_id=candidate_id,
            sha256_hash=sha256_hash,
            source_video_id=source_video_id,
            sequence_id=sequence_id,
            timestamp=timestamp,
            payload_bytes=payload_bytes,
        )
        if not is_clean:
            raise BenchmarkLeakageError(reason)

    def _compute_simple_dhash(self, payload_bytes: bytes) -> int:
        """Fast difference hash proxy over raw byte sequence."""
        if not payload_bytes or len(payload_bytes) < 64:
            return 0
        step = max(len(payload_bytes) // 64, 1)
        sampled = [payload_bytes[i * step] for i in range(64)]
        diff_bits = 0
        for i in range(63):
            if sampled[i] > sampled[i + 1]:
                diff_bits |= (1 << i)
        return diff_bits

    def reset(self):
        self._quarantined_samples.clear()
        self._quarantined_shas.clear()
        self._quarantined_videos.clear()


thermal_benchmark_leakage_guard = ThermalBenchmarkLeakageGuard()
