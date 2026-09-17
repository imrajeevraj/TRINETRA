"""
TRINETRA Phase XIV — Cross-Spectral Failure Analysis Service
Classifies and clusters multimodal perception failures.
Feeds validated failure samples into Phase XIII active learning queues.
"""

from __future__ import annotations
import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from backend.app.services.active_learning_queue import active_learning_queue

logger = logging.getLogger("CrossSpectralFailureService")


class CrossSpectralFailureType(str, Enum):
    THERMAL_MISSED_BY_OPTICAL = "THERMAL_MISSED_BY_OPTICAL"
    OPTICAL_MISSED_BY_THERMAL = "OPTICAL_MISSED_BY_THERMAL"
    FALSE_CROSS_ASSOCIATION = "FALSE_CROSS_ASSOCIATION"
    REGISTRATION_FAILURE = "REGISTRATION_FAILURE"
    TEMPORAL_MISMATCH = "TEMPORAL_MISMATCH"
    THERMAL_CLUTTER = "THERMAL_CLUTTER"
    HOT_BACKGROUND_CONFUSION = "HOT_BACKGROUND_CONFUSION"
    REFLECTION_ARTIFACTS = "REFLECTION_ARTIFACTS"
    OCCLUSION_MISMATCH = "OCCLUSION_MISMATCH"


@dataclass
class CrossSpectralFailureCluster:
    cluster_id: str
    failure_type: CrossSpectralFailureType
    description: str
    sample_count: int = 0
    severity: str = "LOW"             # LOW, HIGH, CRITICAL
    affected_cameras: List[str] = field(default_factory=list)
    sample_ids: List[str] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)


class CrossSpectralFailureService:
    """
    Manages multimodal failure patterns and escalates severity.
    """

    def __init__(self):
        self._clusters: Dict[str, CrossSpectralFailureCluster] = {}
        self._initialize_clusters()

    def _initialize_clusters(self):
        for ft in CrossSpectralFailureType:
            cid = f"CLS-MM-{ft.value}"
            self._clusters[cid] = CrossSpectralFailureCluster(
                cluster_id=cid,
                failure_type=ft,
                description=f"Automated cluster for {ft.value}",
            )

    def record_failure(
        self,
        failure_type: CrossSpectralFailureType,
        camera_id: str,
        sample_id: str,
        optical_conf: Optional[float] = None,
        thermal_conf: Optional[float] = None,
        class_name: str = "person",
        enqueue_active_learning: bool = True,
    ) -> CrossSpectralFailureCluster:
        """
        Assigns failure to appropriate cluster, updates severity, and feeds active learning queue.
        """
        cid = f"CLS-MM-{failure_type.value}"
        cluster = self._clusters.get(cid)
        if not cluster:
            cluster = CrossSpectralFailureCluster(
                cluster_id=cid,
                failure_type=failure_type,
                description=f"Cluster for {failure_type.value}",
            )
            self._clusters[cid] = cluster

        cluster.sample_count += 1
        if camera_id not in cluster.affected_cameras:
            cluster.affected_cameras.append(camera_id)
        cluster.sample_ids.append(sample_id)
        cluster.last_updated = time.time()

        # Dynamic severity escalation
        if cluster.sample_count >= 50:
            cluster.severity = "CRITICAL"
        elif cluster.sample_count >= 20:
            cluster.severity = "HIGH"
        else:
            cluster.severity = "LOW"

        # Active Learning Prioritization (Phase XIII integration)
        if enqueue_active_learning:
            active_learning_queue.enqueue_sample(
                sample_id=sample_id,
                model_domain="MULTIMODAL",
                camera_id=camera_id,
                failure_type=failure_type.value,
                confidence=optical_conf if optical_conf is not None else (thermal_conf or 0.5),
                champion_conf=optical_conf,
                challenger_conf=thermal_conf,
                class_name=class_name,
                is_disputed=True if failure_type == CrossSpectralFailureType.FALSE_CROSS_ASSOCIATION else False,
            )

        return cluster

    def get_cluster(self, cluster_id: str) -> Optional[CrossSpectralFailureCluster]:
        return self._clusters.get(cluster_id)

    def list_clusters(self) -> List[CrossSpectralFailureCluster]:
        return list(self._clusters.values())

    def reset(self):
        self._clusters.clear()
        self._initialize_clusters()


cross_spectral_failure_service = CrossSpectralFailureService()
