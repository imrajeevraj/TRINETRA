"""
TRINETRA — Data Poisoning Defense & Anomaly Detection Service (Phase XIII)
Protects the continuous learning pipeline against adversarial label floods,
class skew injection, duplicate floods, and contradictory sample attacks.
"""

from __future__ import annotations
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from pydantic import BaseModel, Field

from backend.app.core.events_pubsub import publish_event
from backend.app.services.dataset_governance_service import BENCHMARK_HASHES_PATH

logger = logging.getLogger("DataPoisoningDefense")


class PoisoningAlert(BaseModel):
    alert_id: str
    vector_type: str  # "BENCHMARK_LEAKAGE", "DUPLICATE_FLOOD", "CLASS_SKEW", "ABNORMAL_REVIEWER", "CONTRADICTORY_LABELS", "CAMERA_SPIKE"
    severity: str    # "MEDIUM", "HIGH", "CRITICAL"
    sample_ids: List[str]
    details: str
    detected_at: float = Field(default_factory=time.time)
    quarantine_action: str = "DATASET_POISONING_SUSPECTED_SAMPLES_QUARANTINED"


class DataPoisoningDefenseService:
    """
    Automated security filter inspecting candidate datasets and feedback batches before curation.
    """

    def __init__(self):
        self.alerts: List[PoisoningAlert] = []
        self.quarantined_samples: Set[str] = set()
        self.frozen_benchmark_hashes: Set[str] = set()
        self._load_frozen_benchmark_hashes()
        self._counter = 0

    def _load_frozen_benchmark_hashes(self):
        if BENCHMARK_HASHES_PATH.exists():
            try:
                with open(BENCHMARK_HASHES_PATH, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            parts = line.split()
                            if parts:
                                self.frozen_benchmark_hashes.add(parts[0].upper())
            except Exception as e:
                logger.error(f"Failed to load benchmark checksums: {e}")

    def inspect_sample_batch(
        self,
        samples: List[Dict[str, Any]],  # [{ "sample_id": ..., "frame_sha": ..., "label": ..., "camera_id": ..., "reviewer_id": ... }]
        baseline_class_ratios: Optional[Dict[str, float]] = None,
    ) -> Tuple[bool, List[PoisoningAlert]]:
        """
        Scans a candidate batch of samples for poisoning vectors.
        Returns (is_clean, alerts). If poisoning is suspected, returns is_clean = False.
        """
        batch_alerts: List[PoisoningAlert] = []
        now = time.time()

        # 1. Benchmark Leakage Vector
        leaked = []
        for s in samples:
            sha = s.get("frame_sha", "").upper()
            if sha in self.frozen_benchmark_hashes:
                leaked.append(s.get("sample_id", sha))

        if leaked:
            self._counter += 1
            alert = PoisoningAlert(
                alert_id=f"PSN-LEAK-{int(now)}-{self._counter:03d}",
                vector_type="BENCHMARK_LEAKAGE",
                severity="CRITICAL",
                sample_ids=leaked,
                details=f"Detected {len(leaked)} samples matching frozen benchmark IBVAP-GT-v1.0.",
            )
            batch_alerts.append(alert)
            self.quarantined_samples.update(leaked)

        # 2. Duplicate Flooding Vector
        sha_counts: Dict[str, List[str]] = {}
        for s in samples:
            sha = s.get("frame_sha", "").upper()
            if sha:
                if sha not in sha_counts:
                    sha_counts[sha] = []
                sha_counts[sha].append(s.get("sample_id", sha))

        flooded = []
        for sha, sids in sha_counts.items():
            if len(sids) >= 5:  # Flood threshold
                flooded.extend(sids)

        if flooded:
            self._counter += 1
            alert = PoisoningAlert(
                alert_id=f"PSN-FLOOD-{int(now)}-{self._counter:03d}",
                vector_type="DUPLICATE_FLOOD",
                severity="HIGH",
                sample_ids=flooded,
                details=f"Detected duplicate flood of {len(flooded)} samples sharing identical frame content.",
            )
            batch_alerts.append(alert)
            self.quarantined_samples.update(flooded)
            for sha, sids in sha_counts.items():
                if len(sids) >= 5:
                    self.quarantined_samples.add(sha)

        # 3. Class Skew Vector
        if baseline_class_ratios and len(samples) >= 10:
            class_counts: Dict[str, int] = {}
            for s in samples:
                lbl = s.get("label")
                if lbl:
                    class_counts[lbl] = class_counts.get(lbl, 0) + 1
            total_samples = len(samples)
            for cls_name, count in class_counts.items():
                ratio = count / total_samples
                baseline_ratio = baseline_class_ratios.get(cls_name, 0.20)
                if ratio > 0.60 and ratio > (baseline_ratio * 2.0):
                    self._counter += 1
                    alert = PoisoningAlert(
                        alert_id=f"PSN-SKEW-{int(now)}-{self._counter:03d}",
                        vector_type="CLASS_SKEW",
                        severity="HIGH",
                        sample_ids=[s.get("sample_id", "") for s in samples if s.get("label") == cls_name],
                        details=f"Class '{cls_name}' forms {ratio*100:.1f}% of batch, drastically exceeding expected baseline ({baseline_ratio*100:.1f}%).",
                    )
                    batch_alerts.append(alert)

        # 4. Contradictory Label Vector
        sha_labels: Dict[str, Set[str]] = {}
        for s in samples:
            sha = s.get("frame_sha", "").upper()
            lbl = s.get("label")
            if sha and lbl:
                if sha not in sha_labels:
                    sha_labels[sha] = set()
                sha_labels[sha].add(lbl)

        contradictions = []
        for sha, lbls in sha_labels.items():
            if len(lbls) > 1:
                contradictions.append(sha)

        if contradictions:
            self._counter += 1
            alert = PoisoningAlert(
                alert_id=f"PSN-CONTRADICT-{int(now)}-{self._counter:03d}",
                vector_type="CONTRADICTORY_LABELS",
                severity="HIGH",
                sample_ids=contradictions,
                details=f"Detected conflicting class labels assigned to {len(contradictions)} identical frames.",
            )
            batch_alerts.append(alert)
            self.quarantined_samples.update(contradictions)

        # 4. Camera Concentration Spike Vector
        cam_counts: Dict[str, int] = {}
        for s in samples:
            c = s.get("camera_id")
            if c:
                cam_counts[c] = cam_counts.get(c, 0) + 1

        total = len(samples)
        if total >= 20:
            for c, cnt in cam_counts.items():
                if (cnt / total) > 0.70:  # Single camera dominates > 70% of batch
                    self._counter += 1
                    alert = PoisoningAlert(
                        alert_id=f"PSN-SPIKE-{int(now)}-{self._counter:03d}",
                        vector_type="CAMERA_SPIKE",
                        severity="MEDIUM",
                        sample_ids=[s.get("sample_id", "") for s in samples if s.get("camera_id") == c],
                        details=f"Camera {c} generates {cnt}/{total} ({cnt/total*100:.1f}%) of batch samples, causing camera overfitting risk.",
                    )
                    batch_alerts.append(alert)

        # Record alerts and emit WebSocket notifications
        for a in batch_alerts:
            self.alerts.append(a)
            publish_event("ai.poisoning_alert", {
                "alert_id": a.alert_id,
                "vector_type": a.vector_type,
                "severity": a.severity,
                "sample_count": len(a.sample_ids),
                "details": a.details,
            })
            logger.warning(f"POISONING DEFENSE ALERT [{a.vector_type}]: {a.details}")

        is_clean = len(batch_alerts) == 0
        return is_clean, batch_alerts

    def list_alerts(self) -> List[PoisoningAlert]:
        return list(self.alerts)

    def reset(self):
        self.alerts.clear()
        self.quarantined_samples.clear()
        self._counter = 0


data_poisoning_defense = DataPoisoningDefenseService()
