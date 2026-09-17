"""
TRINETRA — Failure Clustering Service (Phase XIII)
Clusters operational failure patterns across domains to target retraining campaigns.
"""

from __future__ import annotations
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from pydantic import BaseModel, Field

from backend.app.core.events_pubsub import publish_event

logger = logging.getLogger("FailureClusteringService")


class FailureClusterRecord(BaseModel):
    cluster_id: str
    name: str
    domain: str
    failure_type: str
    sample_count: int = 0
    sample_ids: List[str] = Field(default_factory=list)
    affected_cameras: List[str] = Field(default_factory=list)
    affected_models: List[str] = Field(default_factory=list)
    first_seen: float = Field(default_factory=time.time)
    last_seen: float = Field(default_factory=time.time)
    severity: str = "MEDIUM"  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    status: str = "OPEN"      # "OPEN", "CURATING", "RESOLVED"
    target_dataset_version: Optional[str] = None
    curation_notes: str = ""


class FailureClusteringService:
    """
    Manages the taxonomy and dynamic assignment of operational failures into clusters.
    """

    def __init__(self):
        self.clusters: Dict[str, FailureClusterRecord] = {}
        self._init_standard_clusters()

    def _init_standard_clusters(self):
        """Initializes the baseline operational failure clusters."""
        standard_defs = [
            {
                "id": "CLS-SECURITY-TOOL-DISTRACTOR",
                "name": "Handheld Tools vs Firearms Confusion",
                "domain": "SECURITY_ITEM",
                "failure_type": "MISCLASSIFICATION",
                "severity": "HIGH",
                "target_dataset": "IBVAP-SECURITY-FEEDBACK-v1",
                "notes": "Cordless drills, impact drivers, and angle grinders confused for firearms",
            },
            {
                "id": "CLS-GROUND-SMALL-SHADOW-PERSON",
                "name": "Small Distant Pedestrians in Perimeter Shadows",
                "domain": "GROUND",
                "failure_type": "FALSE_NEGATIVE",
                "severity": "CRITICAL",
                "target_dataset": "IBVAP-GROUND-FEEDBACK-v1",
                "notes": "Low-contrast distant pedestrians (<32x32px) in perimeter shadow zones",
            },
            {
                "id": "CLS-GROUND-RIDER-VEHICLE-AMBIGUITY",
                "name": "Person / Vehicle Ambiguity",
                "domain": "GROUND",
                "failure_type": "MISCLASSIFICATION",
                "severity": "MEDIUM",
                "target_dataset": "IBVAP-GROUND-FEEDBACK-v1",
                "notes": "Cyclists, quad bikes, and small ATVs causing bounding box co-occurrence ambiguity",
            },
            {
                "id": "CLS-AIRBORNE-BIRD-SWARM",
                "name": "Airborne Clutter & Bird Swarms",
                "domain": "AIRBORNE",
                "failure_type": "FALSE_POSITIVE",
                "severity": "HIGH",
                "target_dataset": "IBVAP-AIRBORNE-FEEDBACK-v1",
                "notes": "Soaring birds and kites triggering false drone alerts in bright sky sector",
            },
            {
                "id": "CLS-AIRBORNE-CLUTTER-BIRDS",
                "name": "Airborne Clutter & Bird False Positives",
                "domain": "AIRBORNE",
                "failure_type": "FALSE_POSITIVE",
                "severity": "HIGH",
                "target_dataset": "IBVAP-AIRBORNE-FEEDBACK-v1",
                "notes": "Soaring birds and kites triggering false drone alerts in bright sky sector",
            },
            {
                "id": "CLS-CAMERA-LOW-LIGHT-DEGRADATION",
                "name": "Camera-Specific Low-Light Degradation",
                "domain": "GROUND",
                "failure_type": "FALSE_NEGATIVE",
                "severity": "MEDIUM",
                "target_dataset": "IBVAP-GROUND-FEEDBACK-v1",
                "notes": "IR illumination dropoff during dusk/night on specific perimeter camera sectors",
            },
        ]

        now = time.time()
        for d in standard_defs:
            self.clusters[d["id"]] = FailureClusterRecord(
                cluster_id=d["id"],
                name=d["name"],
                domain=d["domain"],
                failure_type=d["failure_type"],
                severity=d["severity"],
                target_dataset_version=d["target_dataset"],
                curation_notes=d["notes"],
                first_seen=now,
                last_seen=now,
            )

    def assign_sample_to_cluster(
        self,
        sample_id: str,
        domain: str,
        camera_id: str,
        model_id: str,
        failure_type: str = "FALSE_POSITIVE",
        trigger_or_reason: str = "",
        cluster_id: Optional[str] = None,
        **kwargs,
    ) -> Tuple[bool, str, Optional[FailureClusterRecord]]:
        """
        Assigns an operational sample to a matching failure cluster.
        """
        if cluster_id and cluster_id in self.clusters:
            target_cluster_id = cluster_id
        else:
            # Determine matching cluster
            target_cluster_id = None
            reason_lower = (trigger_or_reason or "").lower()

            if domain == "SECURITY_ITEM" and any(k in reason_lower for k in ["tool", "drill", "grinder", "wrench", "distractor"]):
                target_cluster_id = "CLS-SECURITY-TOOL-DISTRACTOR"
            elif domain == "GROUND" and any(k in reason_lower for k in ["small", "shadow", "distant", "small_person"]):
                target_cluster_id = "CLS-GROUND-SMALL-SHADOW-PERSON"
            elif domain == "GROUND" and any(k in reason_lower for k in ["bike", "rider", "cyclist", "atv", "ambiguity"]):
                target_cluster_id = "CLS-GROUND-RIDER-VEHICLE-AMBIGUITY"
            elif domain == "AIRBORNE" and any(k in reason_lower for k in ["bird", "clutter", "sky", "kite"]):
                target_cluster_id = "CLS-AIRBORNE-CLUTTER-BIRDS"
            elif any(k in reason_lower for k in ["night", "low_light", "lux", "illumination"]):
                target_cluster_id = "CLS-CAMERA-LOW-LIGHT-DEGRADATION"

        if not target_cluster_id:
            # Fallback to domain default
            if domain == "GROUND":
                target_cluster_id = "CLS-GROUND-SMALL-SHADOW-PERSON"
            elif domain == "AIRBORNE":
                target_cluster_id = "CLS-AIRBORNE-CLUTTER-BIRDS"
            else:
                target_cluster_id = "CLS-SECURITY-TOOL-DISTRACTOR"

        cluster = self.clusters.get(target_cluster_id)
        if not cluster:
            return False, f"CLUSTER_{target_cluster_id}_NOT_FOUND", None

        now = time.time()
        cluster.sample_count += 1
        if sample_id not in cluster.sample_ids:
            cluster.sample_ids.append(sample_id)
        if camera_id not in cluster.affected_cameras:
            cluster.affected_cameras.append(camera_id)
        if model_id not in cluster.affected_models:
            cluster.affected_models.append(model_id)
        cluster.last_seen = now

        # Update severity if sample count grows
        if cluster.sample_count >= 50 and cluster.severity != "CRITICAL":
            cluster.severity = "HIGH"
        if cluster.sample_count >= 100:
            cluster.severity = "CRITICAL"

        logger.info(f"Assigned sample {sample_id} to cluster {cluster.cluster_id} (Count: {cluster.sample_count})")

        publish_event("ai.failure_cluster_updated", {
            "cluster_id": cluster.cluster_id,
            "sample_count": cluster.sample_count,
            "severity": cluster.severity,
            "affected_cameras": cluster.affected_cameras,
        })

        return True, "SAMPLE_ASSIGNED_TO_CLUSTER", cluster

    def get_cluster(self, cluster_id: str) -> Optional[FailureClusterRecord]:
        return self.clusters.get(cluster_id)

    def list_clusters(self, domain: Optional[str] = None) -> List[FailureClusterRecord]:
        res = list(self.clusters.values())
        if domain:
            res = [c for c in res if c.domain == domain.upper()]
        return res

    def reset(self):
        self.clusters.clear()
        self._init_standard_clusters()


failure_clustering_service = FailureClusteringService()
