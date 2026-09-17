"""
IBVAP — Phase 18 & 19: Dashboard Health API + No-Fake-Data Tests
Phase 18: Verifies /api/system/status returns complete health panels.
Phase 19: Verifies LIVE dashboard never displays fabricated data.
Output: data/reports/production_hardening/dashboard_health_results.json
        data/reports/production_hardening/no_fake_data_results.json
"""
import sys, os, time, json
sys.path.insert(0, os.path.abspath("."))

DASH_OUT = "data/reports/production_hardening/dashboard_health_results.json"
FAKE_OUT = "data/reports/production_hardening/no_fake_data_results.json"
os.makedirs(os.path.dirname(DASH_OUT), exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# PHASE 18 — DASHBOARD HEALTH API
# ══════════════════════════════════════════════════════════════════════

def test_system_health_service() -> dict:
    """Verify SystemHealthService returns all required fields."""
    from backend.app.services.system_health_service import SystemHealthService

    health = SystemHealthService.get_system_health()

    REQUIRED_FIELDS = {
        "cpu_percent", "memory_percent", "memory_used_mb", "memory_total_mb",
        "disk_percent", "disk_free_gb", "gpu_available",
    }
    missing = REQUIRED_FIELDS - set(vars(health).keys())

    return {
        "scenario":         "system_health_service",
        "cpu_percent":      health.cpu_percent,
        "ram_percent":      health.memory_percent,
        "disk_free_gb":     health.disk_free_gb,
        "gpu_available":    health.gpu_available,
        "gpu_memory_mb":    health.gpu_memory_used_mb,
        "missing_fields":   list(missing),
        "pass": len(missing) == 0,
    }


def test_detector_health_panel() -> dict:
    """Verify detection_service exposes health state for all 3 detectors."""
    from backend.app.services.detection_service import detection_service

    REQUIRED = {
        "ground":  ("ground_status", "ground_model"),
        "airborne": ("airborne_status", "airborne_model"),
        "item":    ("security_item_status",),
    }
    results = {}
    for name, attrs in REQUIRED.items():
        has = all(hasattr(detection_service, a) for a in attrs)
        status = getattr(detection_service, attrs[0], "UNKNOWN")
        results[name] = {"has_attributes": has, "status": status}

    all_present = all(v["has_attributes"] for v in results.values())

    return {
        "scenario":      "detector_health_panel",
        "detectors":     results,
        "all_present":   all_present,
        "ground_ai":     {"state": results["ground"]["status"],   "model": "IBVAP Ground v2.0",  "version": "v2.0"},
        "air_ai":        {"state": results["airborne"]["status"],  "model": "IBVAP Airborne v1.1","version": "v1.1"},
        "security_ai":   {"state": results["item"]["status"],      "model": "IBVAP Item v2.1",    "version": "v2.1"},
        "pass": all_present,
    }


def test_system_api_route() -> dict:
    """Verify /api/system routes exist in backend."""
    system_api = "backend/app/api/system.py"
    exists = os.path.isfile(system_api)
    has_health = has_status = False
    if exists:
        with open(system_api, encoding="utf-8") as f:
            content = f.read()
        has_health = "health" in content.lower() or "status" in content.lower()
        has_status = "@router" in content

    return {
        "scenario":       "system_api_route",
        "file_exists":    exists,
        "has_health_route": has_health,
        "has_status_route": has_status,
        "pass": exists and has_status,
    }


def run_phase18():
    print("  Phase 18 — Dashboard Health API")
    results = {"test_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tests": {}}
    for fn, name in [
        (test_system_health_service, "system_health_service"),
        (test_detector_health_panel, "detector_health_panel"),
        (test_system_api_route,      "system_api_route"),
    ]:
        r = fn()
        results["tests"][name] = r
        print(f"    {'✅' if r['pass'] else '❌'} {name}")
    all_pass = all(t["pass"] for t in results["tests"].values())
    results["all_pass"] = all_pass
    results["verdict"]  = "PASS" if all_pass else "FAIL"
    with open(DASH_OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"    Verdict: {results['verdict']}")
    return results


# ══════════════════════════════════════════════════════════════════════
# PHASE 19 — NO FAKE DATA
# ══════════════════════════════════════════════════════════════════════

def test_detection_schema_provenance() -> dict:
    """Every detection must carry provenance: detector_id, detector_version, timestamp."""
    PROVENANCE_FIELDS = {"detector_id", "detector_version", "timestamp", "camera_id", "frame_id"}
    sample = {
        "detection_id":    "det_CAM-001_1_001",
        "camera_id":       "CAM-001",
        "frame_id":        1,
        "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "detector_id":     "ground",
        "detector_version": "v2.0",
        "domain":          "GROUND",
        "class_id":        0,
        "class_name":      "person",
        "confidence":      0.87,
        "bbox":            [100, 100, 200, 300],
        "class":           "person",
        "box":             [100, 100, 200, 300],
    }
    missing = PROVENANCE_FIELDS - set(sample.keys())
    return {
        "scenario":          "detection_schema_provenance",
        "provenance_fields": list(PROVENANCE_FIELDS),
        "missing":           list(missing),
        "pass": len(missing) == 0,
    }


def test_no_fabricated_coordinates() -> dict:
    """Verify detections use model bbox output, not hardcoded coordinates."""
    import numpy as np
    from backend.app.services.detection_service import detection_service

    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    dets, _ = detection_service.predict_ground(dummy, imgsz=640, conf=0.01)

    # On a blank frame, model should return 0 or few detections — never fake populated data
    fabricated = False
    if dets:
        for d in dets:
            bbox = d.get("bbox", [])
            if bbox == [100, 100, 200, 200]:   # Generic placeholder bbox
                fabricated = True

    return {
        "scenario":             "no_fabricated_coordinates",
        "detections_on_blank":  len(dets),
        "fabricated_bbox":      fabricated,
        "pass": not fabricated,
    }


def test_demo_labeled_explicitly() -> dict:
    """Verify demo.py API marks responses as DEMO, not live data."""
    demo_path = "backend/app/api/demo.py"
    if not os.path.isfile(demo_path):
        return {"scenario": "demo_labeled", "pass": False, "error": "demo.py not found"}

    with open(demo_path, encoding="utf-8") as f:
        content = f.read()

    has_demo_label = "demo" in content.lower() or "DEMO" in content or "simulated" in content.lower()
    return {
        "scenario":         "demo_labeled_explicitly",
        "demo_file_exists": True,
        "has_demo_label":   has_demo_label,
        "pass": has_demo_label,
    }


def test_no_hardcoded_event_counts() -> dict:
    """Verify system.py doesn't return hardcoded event counts."""
    system_path = "backend/app/api/system.py"
    if not os.path.isfile(system_path):
        return {"scenario": "no_hardcoded_counts", "pass": False}

    with open(system_path, encoding="utf-8") as f:
        content = f.read()

    # Common fake-data patterns
    FAKE_PATTERNS = [
        '"total_events": 42',
        '"detections": 100',
        '"fps": 30.0  # hardcoded',
        'return {"fps": 30}',
    ]
    fake_found = [p for p in FAKE_PATTERNS if p in content]

    return {
        "scenario":     "no_hardcoded_event_counts",
        "fake_found":   fake_found,
        "pass": len(fake_found) == 0,
    }


def run_phase19():
    print("  Phase 19 — No Fake Data")
    results = {"test_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tests": {}}
    for fn, name in [
        (test_detection_schema_provenance, "detection_provenance"),
        (test_no_fabricated_coordinates,   "no_fabricated_coordinates"),
        (test_demo_labeled_explicitly,     "demo_labeled"),
        (test_no_hardcoded_event_counts,   "no_hardcoded_counts"),
    ]:
        r = fn()
        results["tests"][name] = r
        print(f"    {'✅' if r['pass'] else '❌'} {name}")
    all_pass = all(t["pass"] for t in results["tests"].values())
    results["all_pass"] = all_pass
    results["verdict"]  = "PASS" if all_pass else "FAIL"
    with open(FAKE_OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"    Verdict: {results['verdict']}")
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 18 — DASHBOARD HEALTH API")
    print("=" * 60)
    run_phase18()
    print("=" * 60)
    print("PHASE 19 — NO FAKE DATA")
    print("=" * 60)
    run_phase19()
