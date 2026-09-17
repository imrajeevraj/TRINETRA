"""
TRINETRA — Edge Camera Mesh & Distributed Intelligence REST API (Phase XI)
Provides authenticated endpoints for inspecting edge nodes, network partition status,
peer handovers, outbox backlogs, and operator emergency global locks with RBAC.
"""

from __future__ import annotations
import time
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from backend.app.core.security import get_current_user, require_roles
from backend.app.services.edge_camera_node import (
    edge_node_manager,
    EdgeCameraNode,
    NetworkPartitionMode,
    NodeHealth
)
from backend.app.services.camera_topology import camera_topology_service
from backend.app.services.edge_state_reconciliation import edge_state_reconciliation_service

router = APIRouter(prefix="/edge", tags=["Edge Camera Mesh"])


class GlobalLockRequest(BaseModel):
    reason: str = Field("OPERATOR_EMERGENCY_OVERRIDE", max_length=256)


@router.get("/nodes", summary="List all edge camera nodes")
def list_edge_nodes(
    health_filter: Optional[str] = None,
    user: Any = Depends(get_current_user)
):
    nodes = []
    for node in edge_node_manager.nodes.values():
        if health_filter and node.health.value != health_filter.upper():
            continue
        hb = node.send_heartbeat()
        nodes.append({
            "node_id": node.node_id,
            "camera_id": node.camera_id,
            "health": node.health.value,
            "network_state": node.network_state.value,
            "is_enabled": node.is_enabled,
            "clock_offset_ms": node.clock_offset_ms,
            "lease_valid": hb["lease_valid"],
            "lease_expires_in_sec": hb["lease_expires_in_sec"],
            "outbox_pending_count": hb["outbox_depth"],
            "sensor_type": node.capabilities.get("sensor_type", "OPTICAL"),
            "ptz_capable": node.capabilities.get("ptz", False),
            "thermal_capable": node.capabilities.get("thermal", False),
            "neighbors": node.neighbor_cameras
        })
    return nodes


@router.get("/nodes/{node_id}", summary="Get detailed state of a specific edge node")
def get_edge_node(node_id: str, user: Any = Depends(get_current_user)):
    # Match either node_id or camera_id
    target_node = None
    for n in edge_node_manager.nodes.values():
        if n.node_id == node_id or n.camera_id == node_id:
            target_node = n
            break

    if not target_node:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Edge node '{node_id}' not found."
        )

    hb = target_node.send_heartbeat()
    return {
        "node_id": target_node.node_id,
        "camera_id": target_node.camera_id,
        "health": target_node.health.value,
        "network_state": target_node.network_state.value,
        "is_enabled": target_node.is_enabled,
        "global_ptz_lock_active": target_node.global_ptz_lock_active,
        "clock_offset_ms": target_node.clock_offset_ms,
        "capabilities": target_node.capabilities,
        "neighbors": target_node.neighbor_cameras,
        "lease": {
            "lease_id": target_node.authorization_lease.lease_id if target_node.authorization_lease else None,
            "valid": hb["lease_valid"],
            "expires_in_sec": hb["lease_expires_in_sec"]
        },
        "outbox_depth": hb["outbox_depth"],
        "active_handovers": list(target_node.local_handovers.values())
    }


@router.get("/topology", summary="Get 8-camera mesh topology with sensor specs")
def get_mesh_topology(user: Any = Depends(get_current_user)):
    return camera_topology_service.get_topology_graph()


@router.get("/health", summary="System-wide edge mesh health status")
def get_edge_health(user: Any = Depends(get_current_user)):
    total_nodes = len(edge_node_manager.nodes)
    online = sum(1 for n in edge_node_manager.nodes.values() if n.health == NodeHealth.ONLINE)
    degraded = sum(1 for n in edge_node_manager.nodes.values() if n.health == NodeHealth.DEGRADED)
    offline = sum(1 for n in edge_node_manager.nodes.values() if n.health == NodeHealth.OFFLINE)
    total_outbox = sum(len([o for o in n.outbox if o.status == "PENDING"]) for n in edge_node_manager.nodes.values())

    net_states = {n.network_state.value for n in edge_node_manager.nodes.values()}
    aggregate_net = "NORMAL"
    if "PARTITIONED" in net_states:
        aggregate_net = "PARTITIONED"
    elif "DEGRADED" in net_states:
        aggregate_net = "DEGRADED"
    elif "RECOVERING" in net_states:
        aggregate_net = "RECOVERING"

    return {
        "total_nodes": total_nodes,
        "online_nodes": online,
        "degraded_nodes": degraded,
        "offline_nodes": offline,
        "network_partition_state": aggregate_net,
        "global_ptz_lock": edge_node_manager.global_ptz_lock,
        "total_pending_outbox_events": total_outbox,
        "timestamp": time.time()
    }


@router.get("/handover/{handover_id}", summary="Get local edge handover status")
def get_edge_handover(handover_id: str, user: Any = Depends(get_current_user)):
    for node in edge_node_manager.nodes.values():
        if handover_id in node.local_handovers:
            return {
                "node_id": node.node_id,
                "camera_id": node.camera_id,
                "handover": node.local_handovers[handover_id]
            }
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Handover '{handover_id}' not found in active edge mesh nodes."
    )


@router.get("/outbox/{node_id}", summary="Inspect pending outbox buffer for a node")
def get_node_outbox(
    node_id: str,
    limit: int = Query(50, ge=1, le=200),
    user: Any = Depends(get_current_user)
):
    target_node = None
    for n in edge_node_manager.nodes.values():
        if n.node_id == node_id or n.camera_id == node_id:
            target_node = n
            break

    if not target_node:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Edge node '{node_id}' not found."
        )

    items = list(target_node.outbox)
    items.sort(key=lambda x: x.created_at, reverse=True)
    items = items[:limit]

    return [
        {
            "event_id": item.event_id,
            "event_type": item.event_type,
            "status": item.status,
            "attempt_count": item.attempt_count,
            "created_at": item.created_at,
            "payload": item.payload
        }
        for item in items
    ]


@router.post("/nodes/{node_id}/disable", summary="Disable edge node autonomy")
def disable_edge_node(
    node_id: str,
    user: Any = Depends(require_roles("ADMIN", "OPERATOR"))
):
    target_node = None
    for n in edge_node_manager.nodes.values():
        if n.node_id == node_id or n.camera_id == node_id:
            target_node = n
            break

    if not target_node:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Node '{node_id}' not found.")

    target_node.is_enabled = False
    return {"status": "SUCCESS", "node_id": target_node.node_id, "is_enabled": False}


@router.post("/nodes/{node_id}/enable", summary="Enable edge node autonomy")
def enable_edge_node(
    node_id: str,
    user: Any = Depends(require_roles("ADMIN", "OPERATOR"))
):
    target_node = None
    for n in edge_node_manager.nodes.values():
        if n.node_id == node_id or n.camera_id == node_id:
            target_node = n
            break

    if not target_node:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Node '{node_id}' not found.")

    target_node.is_enabled = True
    return {"status": "SUCCESS", "node_id": target_node.node_id, "is_enabled": True}


@router.post("/global-lock", summary="Activate emergency global PTZ lock across all nodes")
def activate_global_ptz_lock(
    req: GlobalLockRequest,
    user: Any = Depends(require_roles("ADMIN", "OPERATOR"))
):
    edge_node_manager.set_global_ptz_lock(True)
    return {
        "status": "LOCKED",
        "global_ptz_lock": True,
        "reason": req.reason,
        "operator": getattr(user, "username", "operator"),
        "timestamp": time.time()
    }


@router.post("/global-unlock", summary="Release emergency global PTZ lock across all nodes")
def release_global_ptz_lock(
    user: Any = Depends(require_roles("ADMIN", "OPERATOR"))
):
    edge_node_manager.set_global_ptz_lock(False)
    return {
        "status": "UNLOCKED",
        "global_ptz_lock": False,
        "operator": getattr(user, "username", "operator"),
        "timestamp": time.time()
    }
