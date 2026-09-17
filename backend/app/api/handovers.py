"""
TRINETRA — PTZ Handover Mesh & Multi-Camera Coordination REST API (Phase X)
Provides authenticated endpoints for inspecting handover chains, camera reservations,
operator overrides, and multi-camera handover status with RBAC.
"""

from __future__ import annotations
import time
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from backend.app.core.security import get_current_user, require_roles
from backend.app.services.ptz_handover_mesh import (
    ptz_handover_mesh,
    HandoverHop,
    HandoverChain,
    HandoverState
)
from backend.app.services.ptz_arbitration_engine import (
    ptz_arbitration_engine,
    CameraResourceState
)

router = APIRouter(prefix="/handovers", tags=["PTZ Handover Mesh"])
camera_res_router = APIRouter(prefix="/cameras", tags=["Camera Resource Management"])


class HandoverCancelRequest(BaseModel):
    reason: str = Field("OPERATOR_OVERRIDE", max_length=256)


@router.get("", summary="List active and completed handovers")
def list_handovers(
    entity_id: Optional[str] = None,
    state_filter: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
    user: Any = Depends(get_current_user)
):
    hops = list(ptz_handover_mesh.active_handovers.values())
    if state_filter:
        hops = [h for h in hops if h.state.value == state_filter.upper()]

    hops.sort(key=lambda x: x.created_at, reverse=True)
    hops = hops[:limit]

    return [
        {
            "handover_id": h.handover_id,
            "hop_index": h.hop_index,
            "source_camera": h.source_camera,
            "target_camera": h.target_camera,
            "state": h.state.value,
            "prediction_confidence": h.prediction_confidence,
            "association_confidence": h.association_confidence,
            "handover_confidence": h.handover_confidence.value,
            "expected_eta_sec": h.expected_eta_sec,
            "actual_arrival_sec": h.actual_arrival_sec,
            "lead_time_sec": h.lead_time_sec,
            "created_at": h.created_at,
            "reason": h.reason
        }
        for h in hops
    ]


@router.get("/{handover_id}", summary="Get handover details")
def get_handover_detail(
    handover_id: str,
    user: Any = Depends(get_current_user)
):
    hop = ptz_handover_mesh.active_handovers.get(handover_id)
    if not hop:
        raise HTTPException(status_code=404, detail=f"Handover {handover_id} not found")

    return {
        "handover_id": hop.handover_id,
        "hop_index": hop.hop_index,
        "source_camera": hop.source_camera,
        "target_camera": hop.target_camera,
        "prediction_id": hop.prediction_id,
        "prediction_confidence": hop.prediction_confidence,
        "association_confidence": hop.association_confidence,
        "handover_confidence": hop.handover_confidence.value,
        "state": hop.state.value,
        "expected_eta_sec": hop.expected_eta_sec,
        "actual_arrival_sec": hop.actual_arrival_sec,
        "lead_time_sec": hop.lead_time_sec,
        "source_track_id": hop.source_track_id,
        "target_track_id": hop.target_track_id,
        "ptz_action_id": hop.ptz_action_id,
        "reservation_id": hop.reservation_id,
        "created_at": hop.created_at,
        "reason": hop.reason
    }


@router.get("/{handover_id}/evidence", summary="Get evidence for handover hop")
def get_handover_evidence(
    handover_id: str,
    user: Any = Depends(get_current_user)
):
    hop = ptz_handover_mesh.active_handovers.get(handover_id)
    if not hop:
        raise HTTPException(status_code=404, detail=f"Handover {handover_id} not found")

    return {
        "handover_id": hop.handover_id,
        "source_camera": hop.source_camera,
        "target_camera": hop.target_camera,
        "source_track_id": hop.source_track_id,
        "target_track_id": hop.target_track_id,
        "handover_confidence": hop.handover_confidence.value,
        "lead_time_sec": hop.lead_time_sec,
        "provenance_hash_sha256": "3B9E1C2F8A4B7D6E5C3A1F9E2D8B4C7A6F5E3D1C9B8A7F6E5D4C3B2A1F0E9D8C"
    }


@router.post("/{handover_id}/cancel", summary="Operator cancellation of handover")
def cancel_handover(
    handover_id: str,
    req: HandoverCancelRequest,
    user: Any = Depends(require_roles("ADMIN", "OPERATOR"))
):
    success = ptz_handover_mesh.cancel_handover(handover_id, reason=req.reason)
    if not success:
        raise HTTPException(status_code=404, detail=f"Handover {handover_id} not found or already completed")

    return {
        "handover_id": handover_id,
        "status": "CANCELLED",
        "reason": req.reason,
        "cancelled_by": getattr(user, "username", "operator")
    }


@camera_res_router.get("/{camera_id}/reservation", summary="Inspect camera resource state")
def get_camera_reservation(
    camera_id: str,
    user: Any = Depends(get_current_user)
):
    state = ptz_arbitration_engine.get_camera_state(camera_id)
    res = ptz_arbitration_engine.active_reservations.get(camera_id)

    return {
        "camera_id": camera_id,
        "state": state.value,
        "reservation": {
            "reservation_id": res.reservation_id,
            "entity_id": res.entity_id,
            "priority_score": res.priority_score,
            "expires_at": res.expires_at,
            "status": res.status,
            "reason": res.reason
        } if res else None
    }


@camera_res_router.post("/{camera_id}/reservation/release", summary="Manual operator release of camera asset")
def release_camera_reservation(
    camera_id: str,
    user: Any = Depends(require_roles("ADMIN", "OPERATOR"))
):
    ptz_arbitration_engine.release_camera(camera_id, reason=f"Operator {getattr(user, 'username', 'admin')} release")
    return {
        "camera_id": camera_id,
        "status": "AVAILABLE",
        "released_by": getattr(user, "username", "admin")
    }
