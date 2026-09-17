"""T-02: Comprehensive API Endpoints Test Suite
Validates authentication, role enforcement, camera feeds, events lifecycle,
ANPR plate retrieval, system health, and Prometheus metrics endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.core.security import hash_password, create_access_token
from backend.app.models.user import User
from backend.app.models.camera import Camera
from backend.app.models.event import SecurityEvent, PlateEvent


@pytest.fixture
def client(db_session):
    return TestClient(app)


@pytest.fixture
def admin_user(db_session):
    user = User(
        username="admin_test",
        hashed_password=hash_password("SecretPass123!"),
        role="ADMIN",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def viewer_user(db_session):
    user = User(
        username="viewer_test",
        hashed_password=hash_password("ViewerPass123!"),
        role="VIEWER",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def admin_headers(admin_user):
    token = create_access_token(admin_user.username, admin_user.role)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def viewer_headers(viewer_user):
    token = create_access_token(viewer_user.username, viewer_user.role)
    return {"Authorization": f"Bearer {token}"}


# ── 1. Root & Health Checks ──────────────────────────────────────────────────
def test_root_endpoint(client):
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ONLINE"
    assert "TRINETRA" in data["app"]


def test_health_check_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code in [200, 503]
    data = res.json()
    assert "status" in data
    assert "checks" in data


def test_prometheus_metrics_endpoint(client):
    res = client.get("/metrics")
    assert res.status_code == 200
    assert "text/plain" in res.headers["content-type"]
    assert "# HELP" in res.text or "# TYPE" in res.text or len(res.text) > 0


# ── 2. Authentication & Roles ────────────────────────────────────────────────
def test_login_success(client, admin_user):
    res = client.post("/api/auth/login", data={"username": "admin_test", "password": "SecretPass123!"})
    assert res.status_code == 200
    data = res.json()
    assert data["username"] == "admin_test"
    assert data["role"] == "ADMIN"
    assert "ibvap_access_token" in res.cookies


def test_login_invalid_password(client, admin_user):
    res = client.post("/api/auth/login", data={"username": "admin_test", "password": "WrongPassword"})
    assert res.status_code == 401
    assert "Invalid username or password" in res.json()["detail"]


def test_auth_me_endpoint(client, admin_headers):
    res = client.get("/api/auth/me", headers=admin_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["username"] == "admin_test"
    assert data["role"] == "ADMIN"


def test_unauthenticated_request_rejected(client):
    res = client.get("/api/cameras/")
    assert res.status_code == 401


# ── 3. Camera Endpoints ──────────────────────────────────────────────────────
def test_list_cameras(client, db_session, admin_headers):
    cam = Camera(
        id="CAM-TEST-001",
        name="North Watchtower",
        source="data/videos/cam1.mp4",
        location="Sector 4",
        resolution="1920x1080",
        fps=30,
        is_active=True,
        status="ONLINE",
    )
    db_session.add(cam)
    db_session.commit()

    res = client.get("/api/cameras/", headers=admin_headers)
    assert res.status_code == 200
    cameras = res.json()
    assert len(cameras) >= 1
    assert any(c["id"] == "CAM-TEST-001" for c in cameras)


def test_get_camera_by_id(client, db_session, viewer_headers):
    cam = Camera(
        id="CAM-TEST-002",
        name="Perimeter Gate",
        source="data/videos/cam2.mp4",
        location="Gate 1",
        resolution="1280x720",
        fps=25,
        is_active=True,
        status="ONLINE",
    )
    db_session.add(cam)
    db_session.commit()

    res = client.get("/api/cameras/CAM-TEST-002", headers=viewer_headers)
    assert res.status_code == 200
    assert res.json()["id"] == "CAM-TEST-002"

    res_missing = client.get("/api/cameras/CAM-NONEXISTENT", headers=viewer_headers)
    assert res_missing.status_code == 404


# ── 4. Security Events & Auditing ────────────────────────────────────────────
def test_recent_events_and_disposition(client, db_session, admin_headers):
    evt = SecurityEvent(
        event_type="ZONE_ENTRY",
        camera_id="CAM-TEST-001",
        zone_id="restricted_zone",
        track_id="P-101",
        object_type="PERSON",
        confidence=0.92,
        risk_score=85,
        severity="HIGH",
        status="NEW",
        data_origin="LIVE",
    )
    db_session.add(evt)
    db_session.commit()
    db_session.refresh(evt)

    # Fetch recent events
    res = client.get("/api/events/recent?limit=10&data_origin=LIVE", headers=admin_headers)
    assert res.status_code == 200
    events = res.json()
    assert len(events) >= 1

    # Update disposition
    patch_res = client.patch(
        f"/api/events/{evt.id}/disposition",
        json={"status": "ACKNOWLEDGED", "operator_notes": "Patrol dispatched to sector."},
        headers=admin_headers,
    )
    assert patch_res.status_code == 200
    assert patch_res.json()["status"] == "ACKNOWLEDGED"


# ── 5. ANPR Endpoints ────────────────────────────────────────────────────────
def test_get_recent_plates(client, db_session, viewer_headers):
    plate = PlateEvent(
        camera_id="CAM-TEST-001",
        track_id="V-202",
        plate_number="DL01AB1234",
        confidence=0.96,
        watchlist_status="CLEAN",
        data_origin="LIVE",
    )
    db_session.add(plate)
    db_session.commit()

    res = client.get("/api/anpr/plates?limit=10&data_origin=LIVE", headers=viewer_headers)
    assert res.status_code == 200
    plates = res.json()
    assert len(plates) >= 1
    assert any(p["plate_number"] == "DL01AB1234" for p in plates)


# ── 6. System Health Endpoints ───────────────────────────────────────────────
def test_system_health_endpoints(client, admin_headers):
    res_quick = client.get("/api/system/health/quick", headers=admin_headers)
    assert res_quick.status_code == 200
    assert "status" in res_quick.json()

    res_detailed = client.get("/api/system/health/detailed", headers=admin_headers)
    assert res_detailed.status_code == 200
    data = res_detailed.json()
    assert "cpu" in data
    assert "memory" in data
    assert "stage_profiling_ms" in data
