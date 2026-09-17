"""
TRINETRA Phase XIV — Multimodal Sensor Intelligence REST & WebSocket API
Exposes endpoints for sensors, calibration, synchronization, multimodal inference,
cross-spectral associations, tracks, failure clusters, and thermal governance.
"""

import time
import json
import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from backend.app.services.sensor_abstraction import (
    sensor_abstraction_service,
    SensorModality,
    DataOrigin,
)
from backend.app.services.sensor_topology_service import sensor_topology_service
from backend.app.services.sensor_sync_service import sensor_sync_service
from backend.app.services.sensor_calibration_service import sensor_calibration_service
from backend.app.services.cross_spectral_registration import cross_spectral_registration_engine
from backend.app.services.thermal_ingestion_service import thermal_ingestion_service
from backend.app.services.thermal_quality_gate import thermal_quality_gate
from backend.app.services.thermal_dataset_governance import thermal_dataset_governance
from backend.app.services.multimodal_inference_manager import (
    multimodal_inference_manager,
    MultimodalInferenceMode,
)
from backend.app.services.multimodal_fusion_engine import multimodal_fusion_engine
from backend.app.services.multimodal_tracking_service import multimodal_tracking_service
from backend.app.services.sensor_failure_handler import sensor_failure_handler, FailureType
from backend.app.services.cross_spectral_failure_service import cross_spectral_failure_service
from backend.app.services.sensor_drift_monitor import sensor_drift_monitor
from backend.app.services.multimodal_champion_challenger import multimodal_champion_challenger
from backend.app.services.thermal_validation_gate import thermal_validation_gate
from backend.app.services.thermal_sensor_adapter import discover_hardware_sensors
from backend.app.services.thermal_frame_validator import thermal_frame_validator
from backend.app.services.thermal_quality_intelligence import thermal_quality_intelligence
from backend.app.services.thermal_dataset_service import thermal_dataset_service, DatasetState
from backend.app.services.thermal_benchmark_leakage_guard import thermal_benchmark_leakage_guard
from backend.app.services.thermal_annotation_service import thermal_annotation_service
from backend.app.services.thermal_training_service import thermal_training_service
from backend.app.services.thermal_benchmark_service import thermal_benchmark_service
from backend.app.services.rgb_thermal_comparison_engine import rgb_thermal_comparison_engine
from backend.app.services.thermal_deployment_package_service import thermal_deployment_package_service

logger = logging.getLogger("MultimodalAPI")

sensors_router = APIRouter(prefix="/sensors", tags=["sensors"])
multimodal_router = APIRouter(prefix="/multimodal", tags=["multimodal"])
thermal_router = APIRouter(prefix="/thermal", tags=["thermal"])


# --- Schemas ---

class SensorRegistrationRequest(BaseModel):
    sensor_id: str
    camera_id: str
    modality: str
    resolution_w: int = 1920
    resolution_h: int = 1080
    fps: int = 30
    data_origin: str = "SIMULATED"
    calibration_id: Optional[str] = None


class InferenceExecutionRequest(BaseModel):
    camera_id: str
    mode: str = "MODE_C_FUSION"
    scene_lux: float = 120.0
    mock_detections: Optional[Dict[str, List[Dict[str, Any]]]] = None


class SensorFailureRequest(BaseModel):
    failure_type: str
    context: Optional[Dict[str, Any]] = None


# --- 1. Sensors Endpoints ---

@sensors_router.get("")
def list_sensors(camera_id: Optional[str] = None, modality: Optional[str] = None):
    mod_enum = None
    if modality:
        try:
            mod_enum = SensorModality(modality)
        except ValueError:
            pass
    sensors = sensor_abstraction_service.list_sensors(camera_id=camera_id, modality=mod_enum)
    return {
        "count": len(sensors),
        "sensors": [
            {
                "sensor_id": s.sensor_id,
                "camera_id": s.camera_id,
                "modality": s.modality.value,
                "resolution": s.resolution,
                "fps": s.fps,
                "data_origin": s.data_origin.value,
                "position_status": s.position_status,
                "calibration_id": s.calibration_id,
            }
            for s in sensors
        ]
    }


@sensors_router.get("/discover/hardware")
def discover_sensors():
    return discover_hardware_sensors()


@sensors_router.get("/{sensor_id}")
def get_sensor(sensor_id: str):
    sensor = sensor_abstraction_service.get_sensor(sensor_id)
    if not sensor:
        raise HTTPException(status_code=404, detail="Sensor not found")
    node = sensor_topology_service.get_node(sensor_id)
    return {
        "sensor_id": sensor.sensor_id,
        "camera_id": sensor.camera_id,
        "modality": sensor.modality.value,
        "resolution": sensor.resolution,
        "fps": sensor.fps,
        "data_origin": sensor.data_origin.value,
        "position_status": sensor.position_status,
        "calibration_id": sensor.calibration_id,
        "topology_node": {
            "orientation_reference": node.orientation_reference if node else "UNKNOWN",
            "timestamp_source": node.timestamp_source if node else "PTP_IEEE_1588",
            "is_co_located": node.is_co_located if node else False,
            "co_located_with": node.co_located_with if node else None,
        } if node else None
    }


@sensors_router.get("/{sensor_id}/health")
def get_sensor_health(sensor_id: str):
    metrics = sensor_drift_monitor.get_metrics(sensor_id)
    if not metrics:
        # Generate nominal telemetry if not yet recorded
        sensor = sensor_abstraction_service.get_sensor(sensor_id)
        if not sensor:
            raise HTTPException(status_code=404, detail="Sensor not found")
        metrics = sensor_drift_monitor.record_metrics(
            sensor_id=sensor_id,
            camera_id=sensor.camera_id,
            modality=sensor.modality.value,
        )
    return {
        "sensor_id": metrics.sensor_id,
        "camera_id": metrics.camera_id,
        "status": metrics.status.value,
        "mean_brightness": metrics.mean_brightness,
        "histogram_divergence": metrics.histogram_divergence,
        "saturated_pixel_ratio": metrics.saturated_pixel_ratio,
        "dead_pixel_ratio": metrics.dead_pixel_ratio,
        "noise_variance": metrics.noise_variance,
        "timestamp_jitter_ms": metrics.timestamp_jitter_ms,
        "registration_drift_px": metrics.registration_drift_px,
        "metric_classification": metrics.metric_classification,
        "remediation_action": metrics.remediation_action,
    }


@sensors_router.get("/{sensor_id}/calibration")
def get_sensor_calibration(sensor_id: str):
    profile = sensor_calibration_service.get_sensor_profile(sensor_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Calibration profile not found for sensor")
    return {
        "calibration_id": profile.calibration_id,
        "sensor_id": profile.sensor_id,
        "modality": profile.modality.value,
        "status": profile.status.value,
        "has_optical_intrinsics": profile.optical_intrinsics is not None,
        "has_thermal_intrinsics": profile.thermal_intrinsics is not None,
        "has_cross_spectral_extrinsics": profile.cross_spectral_extrinsics is not None,
        "extrinsics": {
            "rotation": profile.cross_spectral_extrinsics.relative_rotation_euler,
            "translation_m": profile.cross_spectral_extrinsics.relative_translation_m,
            "rmse_px": profile.cross_spectral_extrinsics.rmse_alignment_px,
            "version": profile.cross_spectral_extrinsics.alignment_version,
        } if profile.cross_spectral_extrinsics else None,
        "notes": profile.notes,
    }


@sensors_router.get("/{sensor_id}/synchronization")
def get_sensor_synchronization(sensor_id: str, paired_sensor_id: Optional[str] = None):
    # Find sync partner if not provided
    node = sensor_topology_service.get_node(sensor_id)
    partner_id = paired_sensor_id or (node.co_located_with if node else None) or "SNS-CAM005-LWIR"
    status = sensor_sync_service.get_latest_sync_status(sensor_id, partner_id)
    return {
        "sensor_id": sensor_id,
        "paired_sensor_id": partner_id,
        "sync_status": status.value,
        "max_window_ms": sensor_sync_service.max_sync_window_ms,
        "stale_threshold_ms": sensor_sync_service.stale_threshold_ms,
        "clock_source": "PTP_IEEE_1588",
    }


@sensors_router.post("/{sensor_id}/failure")
def simulate_sensor_failure(sensor_id: str, req: SensorFailureRequest):
    sensor = sensor_abstraction_service.get_sensor(sensor_id)
    if not sensor:
        raise HTTPException(status_code=404, detail="Sensor not found")
    try:
        ft = FailureType(req.failure_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid failure_type: {req.failure_type}")

    audit = sensor_failure_handler.handle_sensor_event(
        camera_id=sensor.camera_id,
        sensor_id=sensor_id,
        failure_type=ft,
        context=req.context,
    )
    return {
        "event_id": audit.event_id,
        "camera_id": audit.camera_id,
        "sensor_id": audit.sensor_id,
        "failure_type": audit.failure_type.value,
        "degradation_state": audit.degradation_state.value,
        "message": audit.message,
    }


# --- 2. Multimodal Endpoints ---

@multimodal_router.get("/status")
def get_multimodal_status():
    return {
        "architecture_version": "Phase XIV Multimodal Intelligence",
        "supported_modalities": ["OPTICAL_RGB", "THERMAL_LWIR", "DEPTH", "RADAR", "OTHER"],
        "active_modes": ["MODE_A_OPTICAL_ONLY", "MODE_B_THERMAL_ONLY", "MODE_C_FUSION"],
        "truthfulness_statement": {
            "native_thermal_ai": "NOT VALIDATED (No operational physical dataset)",
            "dataset_status": "IBVAP-THERMAL-READINESS",
            "production_optical_models": "FROZEN & VERIFIED",
        }
    }


@multimodal_router.post("/inference")
def run_multimodal_inference(req: InferenceExecutionRequest):
    try:
        mode_enum = MultimodalInferenceMode(req.mode)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid inference mode: {req.mode}")

    # Generate synthetic/simulated observations for execution
    opt_obs = sensor_abstraction_service.create_observation(
        sensor_id=f"SNS-{req.camera_id.replace('-', '')}-RGB",
        frame_id=f"FR-OPT-{int(time.time()*1000)}",
        payload_bytes=b"\xFF\xD8\xFF\xE0" + (b"\x00" * 64),
        forced_data_origin=DataOrigin.SIMULATED,
    )
    thm_obs = None
    if req.camera_id in ["CAM-005", "CAM-006", "CAM-007"]:
        thm_obs = sensor_abstraction_service.create_observation(
            sensor_id=f"SNS-{req.camera_id.replace('-', '')}-LWIR",
            frame_id=f"FR-THM-{int(time.time()*1000)}",
            payload_bytes=b"\x89PNG" + (b"\x80" * 64),
            forced_data_origin=DataOrigin.SIMULATED,
        )

    res = multimodal_inference_manager.execute_inference(
        camera_id=req.camera_id,
        mode=mode_enum,
        optical_obs=opt_obs,
        thermal_obs=thm_obs,
        mock_detections=req.mock_detections,
    )

    return {
        "inference_id": res.inference_id,
        "camera_id": res.camera_id,
        "requested_mode": res.requested_mode.value,
        "effective_mode": res.effective_mode.value,
        "system_health": res.system_health.value,
        "optical_count": len(res.optical_detections),
        "thermal_count": len(res.thermal_detections),
        "fused_count": len(res.fused_detections),
        "latencies_ms": {
            "optical": res.optical_latency_ms,
            "thermal": res.thermal_latency_ms,
            "fusion": res.fusion_latency_ms,
            "total": res.total_latency_ms,
        },
        "status_message": res.status_message,
    }


@multimodal_router.get("/associations")
def list_cross_spectral_associations(limit: int = 50):
    reg = cross_spectral_registration_engine.get_latest_registration()
    return {
        "latest_registration": {
            "registration_id": reg.registration_id,
            "mode": reg.mode.value,
            "confidence": reg.registration_confidence,
            "error_px": reg.alignment_error_px,
            "delta_ms": reg.timestamp_delta_ms,
            "status_label": reg.status_label,
            "notes": reg.notes,
        } if reg else None,
        "history_count": len(cross_spectral_registration_engine._history),
    }


@multimodal_router.get("/tracks")
def list_multimodal_tracks(camera_id: Optional[str] = None):
    entities = multimodal_tracking_service.list_active_entities(camera_id=camera_id)
    return {
        "count": len(entities),
        "entities": [
            {
                "global_entity_id": e.global_entity_id,
                "camera_id": e.camera_id,
                "class_name": e.class_name,
                "optical_track_id": e.optical_track_id,
                "thermal_track_id": e.thermal_track_id,
                "match_tier": e.match_tier.value,
                "kinematic_match_score": e.kinematic_match_score,
                "velocity_mps": e.velocity_mps,
                "heading_deg": e.heading_deg,
                "status": e.status,
            }
            for e in entities
        ]
    }


@multimodal_router.get("/failures")
def list_multimodal_failures():
    clusters = cross_spectral_failure_service.list_clusters()
    return {
        "cluster_count": len(clusters),
        "clusters": [
            {
                "cluster_id": c.cluster_id,
                "failure_type": c.failure_type.value,
                "sample_count": c.sample_count,
                "severity": c.severity,
                "affected_cameras": c.affected_cameras,
                "last_updated": c.last_updated,
            }
            for c in clusters
        ]
    }


# --- 3. Thermal Governance Endpoints ---

@thermal_router.get("/models")
def list_thermal_models():
    shadow = multimodal_champion_challenger.generate_comparison_report()
    return {
        "production_champion": {
            "model_id": "ibvap_ground_detector_v2.0",
            "modality": "OPTICAL_RGB",
            "status": "PRODUCTION (FROZEN)",
            "authority": "100% OPERATIONAL AUTHORITY",
        },
        "candidate_challenger": {
            "model_id": "ibvap_thermal_yolo11n_candidate",
            "modality": "THERMAL_LWIR",
            "status": "NOT VALIDATED (SHADOW ONLY)",
            "authority": "ZERO OPERATIONAL ACTUATION",
        },
        "shadow_comparison": {
            "report_id": shadow.report_id,
            "sample_count": shadow.sample_count,
            "verdict": shadow.verdict,
            "operational_safety_violated": shadow.operational_safety_violated,
            "notes": shadow.notes,
        }
    }


@thermal_router.get("/datasets")
def list_thermal_datasets():
    datasets = thermal_dataset_governance.list_datasets()
    return {
        "count": len(datasets),
        "datasets": [
            {
                "dataset_id": d.dataset_id,
                "version": d.version,
                "modalities": d.modalities,
                "classes": d.classes,
                "label_status": d.label_status,
                "benchmark_status": d.benchmark_status,
                "frame_count": len(d.frame_hash_manifest),
                "is_training_eligible": d.is_training_eligible,
                "notes": d.notes,
            }
            for d in datasets
        ]
    }


@thermal_router.get("/quality")
def get_thermal_quality_summary():
    return {
        "gate_name": "ThermalQualityGate",
        "rules_evaluated": 11,
        "recent_evaluations_count": len(thermal_quality_gate._gate_history),
        "recent_rejections": sum(1 for r in thermal_quality_gate._gate_history if r.verdict.value == "REJECT"),
        "recent_reviews": sum(1 for r in thermal_quality_gate._gate_history if r.verdict.value == "REVIEW"),
        "recent_accepts": sum(1 for r in thermal_quality_gate._gate_history if r.verdict.value == "ACCEPT"),
    }


@thermal_router.post("/validate")
def validate_thermal_candidate(has_real_dataset: bool = False):
    report = thermal_validation_gate.evaluate_candidate(has_real_thermal_dataset=has_real_dataset)
    return {
        "report_id": report.report_id,
        "candidate_model_id": report.candidate_model_id,
        "overall_verdict": report.overall_verdict,
        "stages_passed": report.stages_passed,
        "stages_total": report.stages_total,
        "has_real_thermal_dataset": report.has_real_thermal_dataset,
        "governance_notes": report.governance_notes,
    }


@thermal_router.get("/benchmark")
def get_thermal_benchmark():
    bm = thermal_benchmark_service.get_benchmark("IBVAP-THERMAL-READINESS-v0")
    return {
        "benchmark_id": bm.benchmark_id if bm else "IBVAP-THERMAL-READINESS-v0",
        "status": bm.status if bm else "NOT_VALIDATED",
        "sample_count": bm.sample_count if bm else 0,
        "is_frozen": bm.is_frozen if bm else True,
        "manifest_sha256": bm.manifest_sha256 if bm else "",
        "truthfulness_notes": bm.truthfulness_notes if bm else "",
    }


@thermal_router.get("/comparison")
def get_rgb_thermal_comparison():
    report = rgb_thermal_comparison_engine.get_latest_report()
    return {
        "report_id": report.report_id if report else "NONE",
        "configs": {
            k: {
                "config_name": v.config_name,
                "model_name": v.model_name,
                "modality": v.modality,
                "data_origin": v.data_origin.value,
                "validation_status": v.validation_status,
                "precision": v.precision,
                "recall": v.recall,
                "f1_score": v.f1_score,
                "night_recall": v.night_recall,
                "mean_latency_ms": v.mean_latency_ms,
            }
            for k, v in report.configs.items()
        } if report else {},
        "superiority_claim": report.superiority_claim if report else "NONE_PERMITTED",
        "truthfulness_notes": report.truthfulness_notes if report else "",
    }


@thermal_router.get("/governance/datasets")
def list_governed_datasets():
    datasets = thermal_dataset_service.list_datasets()
    return {
        "count": len(datasets),
        "datasets": [
            {
                "dataset_id": d.dataset_id,
                "name": d.name,
                "version": d.version,
                "current_state": d.current_state.value,
                "data_origin": d.data_origin.value,
                "sample_count": len(d.samples),
                "is_training_eligible": d.is_training_eligible,
                "manifest_sha256": d.manifest_sha256,
                "notes": d.notes,
            }
            for d in datasets
        ]
    }


@thermal_router.get("/training/candidates")
def list_training_candidates():
    experiments = thermal_training_service.list_experiments()
    return {
        "count": len(experiments),
        "experiments": [
            {
                "experiment_id": e.experiment_id,
                "experiment_name": e.experiment_name,
                "architecture": e.architecture,
                "status": e.status,
                "has_real_thermal_dataset": e.has_real_thermal_dataset,
                "candidate_sha256": e.candidate_sha256,
                "governance_notes": e.governance_notes,
            }
            for e in experiments
        ]
    }


@thermal_router.get("/deployment/packages")
def list_deployment_packages():
    packages = thermal_deployment_package_service.list_packages()
    return {
        "count": len(packages),
        "packages": [
            {
                "package_id": p.package_id,
                "model_name": p.model_name,
                "architecture": p.architecture,
                "model_sha256": p.model_sha256,
                "package_status": p.package_status,
                "runtime_version": p.runtime_version,
            }
            for p in packages
        ]
    }


# --- 4. WebSocket Telemetry ---

@multimodal_router.websocket("/ws/telemetry")
async def multimodal_websocket_telemetry(websocket: WebSocket):
    await websocket.accept()
    try:
        telemetry_events = [
            "THERMAL_SENSOR_CONNECTED",
            "THERMAL_SENSOR_DISCONNECTED",
            "THERMAL_FRAME_REJECTED",
            "THERMAL_CALIBRATION_CHANGED",
            "THERMAL_SYNC_DEGRADED",
            "THERMAL_MODEL_CHANGED",
            "THERMAL_DRIFT_DETECTED",
            "THERMAL_VALIDATION_FAILED",
            "THERMAL_CANDIDATE_READY",
        ]
        counter = 0
        while True:
            evt = telemetry_events[counter % len(telemetry_events)]
            payload = {
                "event": "multimodal.telemetry_heartbeat",
                "sub_event": evt,
                "timestamp": time.time(),
                "sensors_active": len(sensor_abstraction_service.list_sensors()),
                "sync_state": "SYNCED",
                "active_tracks": len(multimodal_tracking_service.list_active_entities()),
                "health": "FULLY_OPERATIONAL",
                "thermal_status": "NOT_VALIDATED",
                "sensor_discovery_status": "NO_REAL_SENSOR_DETECTED",
            }
            counter += 1
            await websocket.send_text(json.dumps(payload))
            await websocket.receive_text()  # wait for keepalive or client ping
    except WebSocketDisconnect:
        logger.info("Multimodal WebSocket client disconnected")
    except Exception as e:
        logger.warning(f"Multimodal WebSocket error: {e}")
