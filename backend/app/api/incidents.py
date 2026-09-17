"""
TRINETRA — Unified Incident & Cross-Camera Intelligence REST API (Phase VIII)
Exposes endpoints for incidents, cross-camera entities, evidence graphs, and operator dispositions.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from backend.app.core.security import get_current_user, require_roles
from backend.app.services.incident_correlation_engine import (
    incident_correlation_engine,
    IncidentRecord,
    IncidentStatus,
    IncidentSeverity
)
from backend.app.services.entity_association_engine import (
    entity_association_engine,
    GlobalEntity
)
from backend.app.services.camera_topology import camera_topology_service
from backend.app.services.evidence_graph_service import evidence_graph_service

router = APIRouter(prefix="/incidents", tags=["incidents"])


class DispositionRequest(BaseModel):
    notes: Optional[str] = Field(default="", max_length=512, description="Operator investigation notes")


@router.get("", response_model=List[Dict[str, Any]])
def list_incidents(
    status: Optional[str] = Query(None, description="Filter by status (OPEN, ESCALATED, CONFIRMED, etc.)"),
    severity: Optional[str] = Query(None, description="Filter by severity (INFO, LOW, MEDIUM, HIGH, CRITICAL)"),
    current_user=Depends(require_roles("ADMIN", "OPERATOR", "VIEWER"))
):
    incidents = incident_correlation_engine.list_incidents(status=status, severity=severity)
    return [
        {
            "incident_id": inc.incident_id,
            "created_at": inc.created_at,
            "updated_at": inc.updated_at,
            "status": inc.status.value,
            "severity": inc.severity.value,
            "risk_score": inc.risk_score,
            "summary": inc.summary,
            "cameras": inc.cameras_involved,
            "sensors": inc.sensors_involved,
            "global_entities": inc.global_entities,
            "evidence_count": len(inc.evidence_hashes)
        }
        for inc in incidents
    ]


@router.get("/topology", response_model=Dict[str, Any])
def get_camera_topology(
    current_user=Depends(require_roles("ADMIN", "OPERATOR", "VIEWER"))
):
    return camera_topology_service.get_topology_graph()


@router.get("/{incident_id}", response_model=Dict[str, Any])
def get_incident(
    incident_id: str,
    current_user=Depends(require_roles("ADMIN", "OPERATOR", "VIEWER"))
):
    inc = incident_correlation_engine.get_incident(incident_id)
    if not inc:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")

    return {
        "incident_id": inc.incident_id,
        "created_at": inc.created_at,
        "updated_at": inc.updated_at,
        "status": inc.status.value,
        "severity": inc.severity.value,
        "risk_score": inc.risk_score,
        "summary": inc.summary,
        "primary_camera": inc.primary_camera,
        "primary_zone": inc.primary_zone,
        "cameras_involved": inc.cameras_involved,
        "tracks_involved": inc.tracks_involved,
        "global_entities": inc.global_entities,
        "sensors_involved": inc.sensors_involved,
        "risk_reasons": inc.risk_reasons,
        "operator_notes": inc.operator_notes,
        "handled_by": inc.handled_by
    }


@router.get("/{incident_id}/timeline", response_model=List[Dict[str, Any]])
def get_incident_timeline(
    incident_id: str,
    current_user=Depends(require_roles("ADMIN", "OPERATOR", "VIEWER"))
):
    inc = incident_correlation_engine.get_incident(incident_id)
    if not inc:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")

    return [
        {
            "timestamp": tl.timestamp,
            "camera_id": tl.camera_id,
            "event_type": tl.event_type,
            "description": tl.description,
            "track_id": tl.track_id,
            "global_id": tl.global_id,
            "sensor_type": tl.sensor_type,
            "risk_delta": tl.risk_delta,
            "evidence_id": tl.evidence_id
        }
        for tl in sorted(inc.timeline, key=lambda x: x.timestamp)
    ]


@router.get("/{incident_id}/evidence", response_model=Dict[str, Any])
def get_incident_evidence(
    incident_id: str,
    current_user=Depends(require_roles("ADMIN", "OPERATOR", "VIEWER"))
):
    inc = incident_correlation_engine.get_incident(incident_id)
    if not inc:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")

    graph = evidence_graph_service.build_incident_graph(incident_id)
    return {
        "incident_id": inc.incident_id,
        "evidence_hashes": inc.evidence_hashes,
        "graph": graph
    }


@router.get("/{incident_id}/entities", response_model=List[Dict[str, Any]])
def get_incident_entities(
    incident_id: str,
    current_user=Depends(require_roles("ADMIN", "OPERATOR", "VIEWER"))
):
    inc = incident_correlation_engine.get_incident(incident_id)
    if not inc:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")

    entities = []
    for gid in inc.global_entities:
        e = entity_association_engine.global_entities.get(gid)
        if e:
            entities.append({
                "global_id": e.global_id,
                "entity_type": e.entity_type,
                "first_seen": e.first_seen,
                "last_seen": e.last_seen,
                "confidence": e.confidence,
                "observations_count": len(e.observations)
            })
    return entities


# --- Entity Specific Routes ---

entities_router = APIRouter(prefix="/entities", tags=["entities"])


@entities_router.get("/{entity_id}", response_model=Dict[str, Any])
def get_entity_profile(
    entity_id: str,
    current_user=Depends(require_roles("ADMIN", "OPERATOR", "VIEWER"))
):
    e = entity_association_engine.global_entities.get(entity_id)
    if not e:
        raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found")

    return {
        "global_id": e.global_id,
        "entity_type": e.entity_type,
        "first_seen": e.first_seen,
        "last_seen": e.last_seen,
        "confidence": e.confidence,
        "observation_count": len(e.observations),
        "status": e.status
    }


@entities_router.get("/{entity_id}/timeline", response_model=List[Dict[str, Any]])
def get_entity_movement_sequence(
    entity_id: str,
    current_user=Depends(require_roles("ADMIN", "OPERATOR", "VIEWER"))
):
    timeline = entity_association_engine.get_entity_timeline(entity_id)
    if timeline is None:
        raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found")
    return timeline


# --- Operator Review & Disposition Actions ---

@router.post("/{incident_id}/acknowledge", response_model=Dict[str, Any])
def acknowledge_incident(
    incident_id: str,
    payload: DispositionRequest,
    current_user=Depends(require_roles("ADMIN", "OPERATOR"))
):
    inc = incident_correlation_engine.record_disposition(
        incident_id, "ACKNOWLEDGE", current_user.username, payload.notes
    )
    if not inc:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")
    return {"incident_id": inc.incident_id, "status": inc.status.value, "handled_by": inc.handled_by}


@router.post("/{incident_id}/confirm", response_model=Dict[str, Any])
def confirm_incident(
    incident_id: str,
    payload: DispositionRequest,
    current_user=Depends(require_roles("ADMIN", "OPERATOR"))
):
    inc = incident_correlation_engine.record_disposition(
        incident_id, "CONFIRM", current_user.username, payload.notes
    )
    if not inc:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")
    return {"incident_id": inc.incident_id, "status": inc.status.value, "handled_by": inc.handled_by}


@router.post("/{incident_id}/reject", response_model=Dict[str, Any])
def reject_incident(
    incident_id: str,
    payload: DispositionRequest,
    current_user=Depends(require_roles("ADMIN", "OPERATOR"))
):
    inc = incident_correlation_engine.record_disposition(
        incident_id, "REJECT", current_user.username, payload.notes
    )
    if not inc:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")
    return {"incident_id": inc.incident_id, "status": inc.status.value, "handled_by": inc.handled_by}


@router.post("/{incident_id}/resolve", response_model=Dict[str, Any])
def resolve_incident(
    incident_id: str,
    payload: DispositionRequest,
    current_user=Depends(require_roles("ADMIN", "OPERATOR"))
):
    inc = incident_correlation_engine.record_disposition(
        incident_id, "RESOLVE", current_user.username, payload.notes
    )
    if not inc:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")
    return {"incident_id": inc.incident_id, "status": inc.status.value, "handled_by": inc.handled_by}
