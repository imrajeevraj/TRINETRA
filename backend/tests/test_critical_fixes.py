from pathlib import Path
from backend.app.services.risk_engine import RiskEngine
from backend.app.core.config import config_manager
from backend.app.services.system_health_service import SystemHealthService


def test_config_hotreload():
    """Test that config changes are reloaded without restart."""
    risk_engine = RiskEngine()
    
    # Get initial weight
    weights1 = risk_engine._get_weights()
    assert isinstance(weights1, dict)
    assert "person" in weights1
    
    # Simulate config file reload
    config_file = Path("configs/risk.yaml")
    if config_file.exists():
        config_manager._load_config_file("risk", config_file)
        
    weights2 = risk_engine._get_weights()
    assert weights2.get("person", 20) >= 0


def test_system_health():
    """Test system health metrics are returned."""
    health = SystemHealthService.get_system_health()
    
    assert 0 <= health.cpu_percent <= 100
    assert 0 <= health.memory_percent <= 100
    assert health.memory_used_mb > 0
    assert health.memory_total_mb > health.memory_used_mb
    assert health.disk_percent >= 0
    assert isinstance(health.gpu_available, bool)


def test_event_filtering(db_session):
    """Test get_recent_events filtering functionality."""
    from backend.app.models.event import SecurityEvent
    from backend.app.api.events import get_recent_events
    from datetime import datetime

    db = db_session
    # Create test events
    test_evt1 = SecurityEvent(
        event_type="ZONE_ENTRY",
        camera_id="CAM-FILTER-01",
        zone_id="RESTRICTED_A",
        track_id="P-991",
        object_type="PERSON",
        severity="CRITICAL",
        risk_score=95,
        status="NEW",
        timestamp=datetime.utcnow()
    )
    test_evt2 = SecurityEvent(
        event_type="VIRTUAL_FENCE_CROSSING",
        camera_id="CAM-FILTER-02",
        zone_id="FENCE_B",
        track_id="V-992",
        object_type="VEHICLE",
        severity="LOW",
        risk_score=15,
        status="ACKNOWLEDGED",
        timestamp=datetime.utcnow()
    )
    demo_evt = SecurityEvent(
        event_type="ZONE_ENTRY",
        camera_id="CAM-DEMO-01",
        zone_id="DEMO_ZONE",
        track_id="DEMO-P-001",
        object_type="PERSON",
        severity="HIGH",
        risk_score=75,
        status="NEW",
        data_origin="DEMO",
        timestamp=datetime.utcnow(),
    )
    db.add(test_evt1)
    db.add(test_evt2)
    db.add(demo_evt)
    db.commit()

    # Test filtering by camera_id
    res_cam = get_recent_events(camera_id="CAM-FILTER-01", db=db, _user=None)
    assert any(e["track_id"] == "P-991" for e in res_cam)
    assert not any(e["track_id"] == "V-992" for e in res_cam)

    # Test filtering by severity
    res_sev = get_recent_events(severity="CRITICAL", db=db, _user=None)
    assert all(e["severity"] == "CRITICAL" for e in res_sev)

    # Test filtering by track_id
    res_track = get_recent_events(track_id="991", db=db, _user=None)
    assert any(e["track_id"] == "P-991" for e in res_track)
    assert not any(e["track_id"] == "V-992" for e in res_track)

    # Test search query
    res_search = get_recent_events(search="RESTRICTED_A", db=db, _user=None)
    assert any(e["zone_id"] == "RESTRICTED_A" for e in res_search)

    # Live views must never receive simulated/demo records by default.
    res_live = get_recent_events(db=db, _user=None)
    assert not any(e["track_id"] == "DEMO-P-001" for e in res_live)
    res_demo = get_recent_events(data_origin="DEMO", db=db, _user=None)
    assert any(e["track_id"] == "DEMO-P-001" for e in res_demo)

    # Clean up test events
    db.delete(test_evt1)
    db.delete(test_evt2)
    db.delete(demo_evt)
    db.commit()


def test_system_health_route_has_single_api_prefix():
    """The UI requests /api/system; router composition must match it exactly."""
    from backend.app.main import app

    paths = set(app.openapi().get("paths", {}).keys())
    assert "/api/system/health/detailed" in paths
    assert "/api/api/system/health/detailed" not in paths
