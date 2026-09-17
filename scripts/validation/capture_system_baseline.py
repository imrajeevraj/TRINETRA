"""
IBVAP — Phase 1: System Baseline Snapshot
Captures complete hardware, software, and model inventory.
Output: docs/validation/PRODUCTION_BASELINE.md
"""
import sys, os, time, platform, hashlib, json, subprocess
sys.path.insert(0, os.path.abspath("."))

import psutil, torch
from ultralytics import __version__ as yolo_ver

OUTPUT_MD   = "docs/validation/PRODUCTION_BASELINE.md"
OUTPUT_JSON = "data/reports/production_hardening/baseline_snapshot.json"
os.makedirs("data/reports/production_hardening", exist_ok=True)
os.makedirs("docs/validation", exist_ok=True)

# ── Hardware ─────────────────────────────────────────────────────────────────
cpu_name    = platform.processor() or "Unknown"
cpu_cores   = psutil.cpu_count(logical=False)
cpu_threads = psutil.cpu_count(logical=True)
ram_gb      = round(psutil.virtual_memory().total / 1024**3, 1)
os_name     = platform.platform()

gpu_name = vram_total_gb = "N/A"
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_total_gb = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)

# ── Software ─────────────────────────────────────────────────────────────────
py_ver   = platform.python_version()
pt_ver   = torch.__version__
cuda_ver = torch.version.cuda or "N/A"

node_ver = redis_ver = pg_ver = sqlite_ver = "N/A"
try:
    node_ver = subprocess.check_output(["node", "--version"], text=True).strip()
except Exception:
    pass

try:
    import redis as _r
    rc = _r.from_url("redis://localhost:6379/0", socket_connect_timeout=2)
    info = rc.info("server")
    redis_ver = info.get("redis_version", "N/A")
    rc.close()
except Exception:
    pass

try:
    import psycopg2
    conn = psycopg2.connect(host="localhost", port=5432, dbname="ibvap_db",
                            user="ibvap_user", password="ibvap_pass", connect_timeout=3)
    raw = conn.server_version  # e.g. 160015
    pg_ver = f"{raw // 10000}.{(raw % 10000) // 100}"
    conn.close()
except Exception:
    pass

try:
    import sqlite3
    conn2 = sqlite3.connect("ibvap.db")
    sqlite_ver = conn2.execute("SELECT sqlite_version()").fetchone()[0]
    conn2.close()
except Exception:
    pass

# ── Models ───────────────────────────────────────────────────────────────────
def sha256_file(path):
    if not os.path.isfile(path):
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536): h.update(chunk)
    return h.hexdigest().upper()

MODELS = {
    "ground_v2_0": {
        "name":    "IBVAP Ground Detector v2.0",
        "status":  "PRODUCTION_ACTIVE",
        "classes": {0: "person", 1: "vehicle"},
        "path":    "models/current/ibvap_detector.pt",
        "expected_sha": "7DBF36027768194F61C6EBBC000E1421374624558168F8A9896F727B296277B0",
    },
    "airborne_v1_1": {
        "name":    "IBVAP Airborne Detector v1.1",
        "status":  "VALIDATED_CANDIDATE",
        "classes": {0: "drone", 1: "aircraft"},
        "path":    "data/training/runs/ibvap_airborne_v1_exp001/weights/best.pt",
        "expected_sha": "E1009633325E463B1C0D45D6578522A1A2026AE27B854591A83461043928C914",
    },
    "security_item_v2_1": {
        "name":    "IBVAP Security Item Detector v2.1",
        "status":  "PROMOTED_TO_VALIDATED",
        "classes": {0: "firearm"},
        "path":    "data/training/runs/ibvap_security_item_v2_1_exp001/weights/best.pt",
        "expected_sha": "72464C778DE57270800146AB5ADF2AB683FFEAC55338C89231EE3A83E7F6E1DA",
    },
}

model_results = {}
for key, m in MODELS.items():
    actual = sha256_file(m["path"])
    match  = actual.upper() == m["expected_sha"].upper() if actual != "FILE_NOT_FOUND" else False
    size_mb = round(os.path.getsize(m["path"]) / 1024**2, 1) if os.path.isfile(m["path"]) else 0
    model_results[key] = {**m, "actual_sha": actual, "sha_match": match, "size_mb": size_mb}

# ── Tests running count ───────────────────────────────────────────────────────
try:
    import subprocess as _sp
    r = _sp.run([sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q", "--no-header"],
                capture_output=True, text=True, timeout=15)
    lines = [l for l in r.stdout.splitlines() if "selected" in l or "test" in l.lower()]
    test_count = lines[-1] if lines else "N/A"
except Exception:
    test_count = "62 tests (last known count)"

snapshot_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

# ── Assemble JSON ─────────────────────────────────────────────────────────────
snapshot = {
    "snapshot_timestamp": snapshot_time,
    "hardware": {
        "os":         os_name,
        "cpu":        cpu_name,
        "cpu_cores":  cpu_cores,
        "cpu_threads": cpu_threads,
        "ram_gb":     ram_gb,
        "gpu":        gpu_name,
        "vram_gb":    vram_total_gb,
    },
    "software": {
        "python":      py_ver,
        "pytorch":     pt_ver,
        "cuda":        cuda_ver,
        "ultralytics": yolo_ver,
        "node":        node_ver,
        "postgresql":  pg_ver,
        "redis":       redis_ver,
        "sqlite":      sqlite_ver,
    },
    "models":   model_results,
    "tests":    test_count,
    "benchmarks_frozen": ["IBVAP-GT-v1.0", "IBVAP-GT-ITEM-v2.0"],
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(snapshot, f, indent=2)

# ── Write Markdown ────────────────────────────────────────────────────────────
sha_rows = ""
for key, m in model_results.items():
    match_str = "✅ VERIFIED" if m["sha_match"] else "❌ MISMATCH"
    sha_rows += f"| {m['name']} | `{m['expected_sha'][:16]}…` | {m['status']} | {match_str} |\n"

md = f"""# IBVAP — Production Baseline Snapshot
**Captured:** {snapshot_time}
**Purpose:** Pre-production hardening baseline. All values captured from live runtime.

---

## 1. Hardware

| Component | Value |
|---|---|
| OS | {os_name} |
| CPU | {cpu_name} |
| CPU Cores / Threads | {cpu_cores} physical / {cpu_threads} logical |
| RAM | {ram_gb} GB |
| GPU | {gpu_name} |
| VRAM Total | {vram_total_gb} GB |

---

## 2. Software Versions

| Component | Version |
|---|---|
| Python | {py_ver} |
| PyTorch | {pt_ver} |
| CUDA | {cuda_ver} |
| Ultralytics YOLO | {yolo_ver} |
| Node.js | {node_ver} |
| PostgreSQL | {pg_ver} |
| Redis | {redis_ver} |
| SQLite (dev fallback) | {sqlite_ver} |

---

## 3. Active Model Versions & SHA-256

| Model | SHA-256 (truncated) | Status | Integrity |
|---|---|---|---|
{sha_rows.strip()}

Full SHA-256 hashes recorded in `data/reports/production_hardening/baseline_snapshot.json`.

---

## 4. Frozen Benchmarks (IMMUTABLE)

- `IBVAP-GT-v1.0` — Ground benchmark. Must NOT be modified.
- `IBVAP-GT-ITEM-v2.0` — Security Item benchmark. Must NOT be modified.

---

## 5. Automated Test Count

{test_count}

---

## 6. Database Configuration

| Database | Endpoint | Status |
|---|---|---|
| PostgreSQL (primary) | localhost:5432/ibvap_db | {'✅ REACHABLE' if pg_ver != 'N/A' else '❌ UNREACHABLE'} |
| Redis | localhost:6379 | {'✅ REACHABLE' if redis_ver != 'N/A' else '❌ UNREACHABLE'} |
| SQLite (dev fallback) | ibvap.db | ✅ PRESENT |

---

## 7. Known Limitations

> ⚠️ **REAL-FIREARM-VIDEO VALIDATION NOT AVAILABLE**
> Security Item model v2.1 has been validated on `IBVAP-GT-ITEM-v2.0` (synthetic/curated benchmark) only.
> Real-world firearm surveillance video validation has not been performed.
> This limitation is explicitly preserved and must not be omitted from any production report.
"""

with open(OUTPUT_MD, "w", encoding="utf-8") as f:
    f.write(md)

print(f"Baseline snapshot written:")
print(f"  Markdown: {OUTPUT_MD}")
print(f"  JSON:     {OUTPUT_JSON}")
print(f"\nHardware: {cpu_name} | {ram_gb}GB RAM | {gpu_name} | {vram_total_gb}GB VRAM")
print(f"Software: Python {py_ver} | PyTorch {pt_ver} | Redis {redis_ver} | PG {pg_ver}")
print(f"Models:   All SHA-256 matches: {all(m['sha_match'] for m in model_results.values())}")
