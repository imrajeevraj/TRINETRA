"""
TRINETRA — Incident Correlation & Multi-Sensor Fusion Engine (Phase VIII)
Correlates Ground, Airborne, Security Item, ANPR, Face, and PTZ evidence across
cameras, time, and space into unified, explainable incidents (INC-2026-XXXXXX).
"""

from __future__ import annotations
import time
import json
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

from backend.app.services.camera_topology import camera_topology_service
from backend.app.services.entity_association_engine import entity_association_engine, GlobalEntity
from backend.app.core.events_pubsub import publish_event

logger = logging.getLogger("IncidentCorrelationEngine")


class IncidentStatus(str, Enum):
    OPEN = "OPEN"
    INVESTIGATING = "INVESTIGATING"
    ESCALATED = "ESCALATED"
    CONFIRMED = "CONFIRMED"
    RESOLVED = "RESOLVED"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    ARCHIVED = "ARCHIVED"


class IncidentSeverity(str, Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class IncidentTimelineEntry:
    timestamp: float
    camera_id: str
    event_type: str
    description: str
    track_id: Optional[str] = None
    global_id: Optional[str] = None
    sensor_type: str = "GROUND"
    risk_delta: int = 0
    evidence_id: Optional[str] = None


@dataclass
class IncidentRecord:
    incident_id: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    status: IncidentStatus = IncidentStatus.OPEN
    severity: IncidentSeverity = IncidentSeverity.LOW
    risk_score: int = 0
    summary: str = ""
    primary_zone: str = "ZONE-PERIMETER"
    primary_camera: str = "CAM-001"
    cameras_involved: List[str] = field(default_factory=list)
    tracks_involved: List[str] = field(default_factory=list)
    global_entities: List[str] = field(default_factory=list)
    sensors_involved: List[str] = field(default_factory=list)
    timeline: List[IncidentTimelineEntry] = field(default_factory=list)
    evidence_hashes: Dict[str, str] = field(default_factory=dict) # evidence_id -> sha256
    risk_reasons: List[str] = field(default_factory=list)
    operator_notes: Optional[str] = None
    handled_by: Optional[str] = None


class IncidentCorrelationEngine:
    def __init__(
        self,
        correlation_window_sec: float = 120.0,
        high_risk_threshold: int = 65,
        critical_risk_threshold: int = 85
    ):
        self.correlation_window_sec = correlation_window_sec
        self.high_risk_thresh = high_risk_threshold
        self.critical_risk_thresh = critical_risk_threshold

        self.incidents: Dict[str, IncidentRecord] = {}
        self._incident_counter: int = 0

    def _generate_incident_id(self) -> str:
        self._incident_counter += 1
        year = time.strftime("%Y", time.gmtime())
        return f"INC-{year}-{self._incident_counter:06d}"

    def correlate_event(
        self,
        camera_id: str,
        event_type: str,
        class_name: str,
        confidence: float,
        bbox: List[float],
        track_id: Optional[str] = None,
        zone_id: Optional[str] = None,
        sensor_type: str = "GROUND",
        evidence_path: Optional[str] = None,
        evidence_hash: Optional[str] = None,
        plate_number: Optional[str] = None,
        face_id: Optional[str] = None,
        risk_score_hint: int = 0,
        timestamp: Optional[float] = None
    ) -> Tuple[IncidentRecord, bool]:
        """
        Main multi-sensor correlation entrypoint.
        Ingests an observation, cross-associates with active global entities,
        and binds it to an existing incident or spawns a new unified incident.
        Returns: (incident, is_new: bool)
        """
        now = timestamp if timestamp is not None else time.time()
        local_track_key = f"{camera_id}:{track_id}" if track_id else f"{camera_id}:evt"

        # 1. Cross-Camera Entity Ingestion
        global_entity: Optional[GlobalEntity] = None
        if track_id:
            global_entity, assoc_dec, assoc_reason = entity_association_engine.ingest_observation(
                camera_id=camera_id,
                track_id=track_id,
                class_name=class_name,
                bbox=bbox,
                timestamp=now,
                plate_number=plate_number,
                face_id=face_id,
                confidence=confidence
            )

        # 2. Search for existing Active Incident within correlation window
        matching_incident: Optional[IncidentRecord] = None

        for inc in self.incidents.values():
            if inc.status in [IncidentStatus.RESOLVED, IncidentStatus.FALSE_POSITIVE, IncidentStatus.ARCHIVED]:
                continue
            if (now - inc.updated_at) > self.correlation_window_sec:
                continue

            # Criteria A: Matches same global entity
            if global_entity and global_entity.global_id in inc.global_entities:
                matching_incident = inc
                break

            # Criteria B: Matches same camera within 30s
            if camera_id in inc.cameras_involved and (now - inc.updated_at) < 30.0:
                matching_incident = inc
                break

            # Criteria C: Matches adjacent camera in topology within travel window
            for exist_cam in inc.cameras_involved:
                feasible, _ = camera_topology_service.is_transition_feasible(exist_cam, camera_id, now - inc.created_at)
                if feasible:
                    matching_incident = inc
                    break
            if matching_incident:
                break

            # Criteria D: Multi-sensor cross-correlation (Airborne drone or Firearm near active ground incident in same or adjacent camera)
            if sensor_type in ["AIRBORNE", "SECURITY_ITEM"] and (now - inc.updated_at) < 60.0:
                cam_relevant = (camera_id in inc.cameras_involved) or any(
                    camera_topology_service.is_transition_feasible(c, camera_id, 15.0)[0] for c in inc.cameras_involved
                )
                if cam_relevant:
                    matching_incident = inc
                    break

        is_new = False
        if not matching_incident:
            # Create new incident
            is_new = True
            inc_id = self._generate_incident_id()
            matching_incident = IncidentRecord(
                incident_id=inc_id,
                created_at=now,
                updated_at=now,
                primary_camera=camera_id,
                primary_zone=zone_id or "ZONE-PERIMETER",
                summary=f"Incident initiated: {event_type} ({class_name}) on {camera_id}"
            )
            self.incidents[inc_id] = matching_incident

        # 3. Augment Incident Metadata
        inc = matching_incident
        inc.updated_at = now
        if camera_id not in inc.cameras_involved:
            inc.cameras_involved.append(camera_id)
        if local_track_key not in inc.tracks_involved:
            inc.tracks_involved.append(local_track_key)
        if sensor_type not in inc.sensors_involved:
            inc.sensors_involved.append(sensor_type)
        if global_entity and global_entity.global_id not in inc.global_entities:
            inc.global_entities.append(global_entity.global_id)

        # 4. Dynamic Risk Re-evaluation
        prev_risk = inc.risk_score
        risk_delta = 0

        if event_type in ["VIRTUAL_FENCE_CROSSING", "RESTRICTED_ZONE_BREACH"]:
            risk_delta += 40
            inc.risk_reasons.append(f"Virtual fence breach on {camera_id} (+40)")
        elif "INTRUSION" in event_type.upper():
            risk_delta += 30
            inc.risk_reasons.append(f"Intrusion event on {camera_id} (+30)")

        if sensor_type == "SECURITY_ITEM":
            risk_delta += 45
            inc.risk_reasons.append(f"Security item weapon alert on {camera_id} (+45)")
        elif sensor_type == "AIRBORNE":
            risk_delta += 35
            inc.risk_reasons.append(f"Airborne target detected on {camera_id} (+35)")
        elif plate_number:
            inc.risk_reasons.append(f"Vehicle plate correlated: {plate_number}")

        if risk_score_hint > 0 and risk_score_hint > prev_risk:
            risk_delta += (risk_score_hint - prev_risk)

        inc.risk_score = min(100, max(prev_risk, prev_risk + risk_delta))

        # 5. Severity Mapping
        if inc.risk_score >= self.critical_risk_thresh or "SECURITY_ITEM" in inc.sensors_involved:
            inc.severity = IncidentSeverity.CRITICAL
            inc.status = IncidentStatus.ESCALATED
        elif inc.risk_score >= self.high_risk_thresh:
            inc.severity = IncidentSeverity.HIGH
            inc.status = IncidentStatus.ESCALATED
        elif inc.risk_score >= 40:
            inc.severity = IncidentSeverity.MEDIUM
        else:
            inc.severity = IncidentSeverity.LOW

        # 6. Add Timeline Entry
        timeline_desc = f"[{sensor_type}] {event_type} ({class_name}, conf={confidence:.2f})"
        if plate_number:
            timeline_desc += f" [Plate: {plate_number}]"
        if global_entity:
            timeline_desc += f" -> Global: {global_entity.global_id}"

        tl_entry = IncidentTimelineEntry(
            timestamp=now,
            camera_id=camera_id,
            event_type=event_type,
            description=timeline_desc,
            track_id=local_track_key,
            global_id=global_entity.global_id if global_entity else None,
            sensor_type=sensor_type,
            risk_delta=risk_delta,
            evidence_id=evidence_hash
        )
        inc.timeline.append(tl_entry)

        # 7. Record Cryptographic Evidence
        if evidence_hash and evidence_path:
            inc.evidence_hashes[evidence_path] = evidence_hash

        # Update Summary
        inc.summary = f"Multi-camera incident involving {len(inc.cameras_involved)} cameras, {len(inc.sensors_involved)} sensors, risk score {inc.risk_score}."

        # Emit Real-Time PubSub Event
        publish_event("incident.updated" if not is_new else "incident.created", {
            "incident_id": inc.incident_id,
            "status": inc.status.value,
            "severity": inc.severity.value,
            "risk_score": inc.risk_score,
            "cameras": inc.cameras_involved,
            "sensors": inc.sensors_involved,
            "global_entities": inc.global_entities,
            "summary": inc.summary
        })

        return inc, is_new

    def get_incident(self, incident_id: str) -> Optional[IncidentRecord]:
        return self.incidents.get(incident_id)

    def list_incidents(
        self,
        status: Optional[str] = None,
        severity: Optional[str] = None
    ) -> List[IncidentRecord]:
        res = list(self.incidents.values())
        if status:
            res = [i for i in res if i.status.value.upper() == status.upper()]
        if severity:
            res = [i for i in res if i.severity.value.upper() == severity.upper()]
        res.sort(key=lambda x: x.updated_at, reverse=True)
        return res

    def record_disposition(
        self,
        incident_id: str,
        action: str,
        actor: str,
        notes: Optional[str] = None
    ) -> Optional[IncidentRecord]:
        """Operator disposition (ACKNOWLEDGE, CONFIRM, REJECT, RESOLVE)."""
        inc = self.incidents.get(incident_id)
        if not inc:
            return None

        action_u = action.upper()
        if action_u == "ACKNOWLEDGE":
            inc.status = IncidentStatus.INVESTIGATING
        elif action_u == "CONFIRM":
            inc.status = IncidentStatus.CONFIRMED
        elif action_u == "REJECT":
            inc.status = IncidentStatus.FALSE_POSITIVE
        elif action_u == "RESOLVE":
            inc.status = IncidentStatus.RESOLVED

        inc.handled_by = actor
        inc.operator_notes = notes
        inc.updated_at = time.time()

        publish_event("incident.disposition", {
            "incident_id": incident_id,
            "action": action_u,
            "actor": actor,
            "status": inc.status.value,
            "notes": notes
        })
        return inc


incident_correlation_engine = IncidentCorrelationEngine()
