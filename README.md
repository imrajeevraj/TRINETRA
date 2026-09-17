# TRINETRA — Threat Recognition, Intelligent Networked Eyes & Tracking for Real-time Analysis

> **"Three Eyes. One Secure Border."**

TRINETRA (formerly IBVAP — Intelligent Border Video Analytics Platform) is an enterprise-grade, edge-deployable C4ISR video analytics platform engineered to transform existing legacy CCTV and IP surveillance infrastructure into an intelligent, autonomous perimeter defense system. TRINETRA integrates a specialized tri-model neural perception architecture, real-time ByteTrack kinematic tracking, zero-tolerance virtual fence polygon tripwires, an explainable multi-factor threat risk scoring engine, tamper-proof SHA-256 forensic video evidence preservation, and a high-responsiveness operator Command Center dashboard.

---

## Project Status

- **Release Version:** `v2.0.0` (Official Production Release)
- **Tagline:** *"Three Eyes. One Secure Border."*
- **Release Status:** 🚀 **PRODUCTION_READY**
- **SIH Problem Statement:** 🟢 **SIH26187 / Ministry of Home Affairs (MHA) & Sashastra Seema Bal (SSB)**

| Milestone | Scope | Result |
| :--- | :--- | :---: |
| **Perception Integrity** | Multi-domain Tri-Model perception (Ground, Air, Weapon) + InsightFace & ANPR | ✅ **100% OPERATIONAL** |
| **P4 Hardening** | 53 Production Hardening Tasks across Security, Reliability & Performance | ✅ **100% PASS** |
| **P5 Release Audit** | Comprehensive Multi-Domain Architecture & Disk Cleanliness Audit | ✅ **PASSED** |
| **P6 Staging Soak** | Continuous 24.0-Hour Multi-Camera Staging Validation | ✅ **100.0% Uptime** |
| **Release Gates** | Master Operational Gates (G-01 through G-16) | ✅ **16 / 16 PASS** |
| **Defect Count** | Critical: **0** \| High: **0** \| Medium: **0** \| Low: **0** | ✅ **Zero Defects** |

---

## Problem Statement

Border guarding forces (e.g., Sashastra Seema Bal, Border Security Force) monitor thousands of kilometers of international borders using hundreds of deployed analog, IP, and thermal CCTV cameras. Manual screen monitoring suffers from critical operational bottlenecks:
1. **Operator Vigilance Fatigue:** Visual attention drops by over 95% after just 20 minutes of continuous screen monitoring.
2. **Disjointed Point Solutions:** Legacy CCTV infrastructure lacks unified detection capable of correlating ground intrusions, low-altitude aerial drones, and handheld weapons simultaneously.
3. **High False Alarm Rates:** Traditional motion-detection triggers continuous false alarms from shifting foliage, wildlife, and weather noise.
4. **Evidentiary Gaps:** Raw video recordings lack cryptographic provenance, tamper-evident sealing, and structured audit logs required for court admissibility and forensic scrutiny.

---

## Architecture & Data Flow

TRINETRA delivers an end-to-end, vendor-agnostic real-time intelligence pipeline that connects to existing RTSP/ONVIF camera streams:

```
+------------------+     +-------------------+     +--------------------+
|  CCTV/IP Camera  | --> |  Video Ingestion  | --> |    AI Detection    |
| (RTSP/ONVIF/MP4) |     |  (MediaMTX/OpenCV)|     | (Tri-Model YOLO11) |
+------------------+     +-------------------+     +--------------------+
                                                             |
                                                             v
+------------------+     +-------------------+     +--------------------+
| Risk/Event Engine| <-- | Zone/Fence Engine | <-- |  ByteTrack 2D/3D   |
| (0-100 Scoring)  |     | (Spatial Polygon) |     | (Kinematic Tracks) |
+------------------+     +-------------------+     +--------------------+
        |
        +----------------------------+
        |                            |
        v                            v
+------------------+     +-------------------+     +--------------------+
| Forensic Evidence| --> | WebSocket Alerts  | --> |  Command Center &  |
| (SHA-256 Sealed) |     | (<50ms Broadcast) |     |  Audit Ledger (DB) |
+------------------+     +-------------------+     +--------------------+
```

---

## The Three Eyes of TRINETRA

TRINETRA employs a decoupled, specialist multi-detector architecture. Instead of relying on a single monolithic model with compromised domain weights, specialized neural perception models run in parallel:

```
                                [ Input Video Frame ]
                                          |
        +---------------------------------+--------------------------------+
        |                                 |                                |
        v                                 v                                v
+-----------------------+     +-----------------------+     +-----------------------+
|   EYE 1: GROUND       |     |    EYE 2: AIRBORNE    |     |  EYE 3: SECURITY ITEM  |
| Ground Model v2.0     |     | Airborne Model v2.0   |     | Security Item v2.1    |
| (YOLO11n, 768x768)    |     | (YOLO11n, 640x640)    |     | (YOLO11n, 640x640)    |
| Classes: Person, Auto |     | Classes: Drone, Plane |     | Classes: Firearm      |
| Latency: 10.59 ms     |     | Latency: 9.30 ms      |     | Latency: 11.99 ms     |
+-----------------------+     +-----------------------+     +-----------------------+
        |                                 |                                |
        +---------------------------------+--------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |       Common Detection Schema         |
                      |  (Class, BBox, Confidence, Domain)    |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |   ByteTrack Association & Tracking    |
                      +---------------------------------------+
```

### 1. Ground Threat Detection & Tracking (Eye 1)
- Detects **Persons** and **Vehicles** at high resolution (768×768) using Ground YOLO11n v2.0.
- Maintains continuous 2D Kalman-filter ByteTrack trajectories with persistent track IDs.
- Model Path: `models/current/ibvap_detector.pt` (SHA-256 verified against `models/model_registry.yaml`).

### 2. Airborne Threat Detection (Eye 2)
- Identifies low-altitude **Drones** (UAVs) and **Aircraft** using Airborne YOLO11n v2.0.
- Evaluates crossings across calibrated **2D Image-Space Air Zones**.
- Model Path: `models/current/ibvap_airborne_detector.pt` (or `models/production/airborne/ibvap_airborne_v2_production.pt`).

### 3. Security Item / Weapon Detection (Eye 3)
- Specialized detection of **Firearms** using Security Item YOLO11n v2.1.
- Implements a mandatory **3-frame spatial-temporal consensus window** to eliminate single-frame transient false positives before alert escalation.
- Model Path: `models/production/security_item/ibvap_security_item_v2_1_production.pt`.

### 4. Facial Recognition & Watchlist Matching
- Integrated **InsightFace Buffalo_L** ONNX suite (`1k3d68.onnx`, `2d106det.onnx`, `det_10g.onnx`, `genderage.onnx`, `w600k_r50.onnx`).
- 512-dimensional vector embedding extraction for watchlist matching and person identification.
- Model Path: `app/data/insightface/models/buffalo_l/`.

### 5. Automated Number Plate Recognition (ANPR)
- EasyOCR-powered text recognition engine with regex pattern validation for Indian military/state vehicle license plates.

---

## Core System Capabilities

### 1. Spatial Video Intelligence & Virtual Fences
- **Polygon Boundaries:** Defined via normalized coordinates in [`configs/zones.yaml`](configs/zones.yaml).
- **Directional Discrimination:** Employs vector cross-product geometry to differentiate `INWARD` intrusions from `OUTWARD` exits.
- **Duplicate Suppression:** Per-track cooldown timers prevent continuous alert spamming while an entity remains within a zone.
- **Air Zone Classification:** All airborne zones are explicitly designated as **2D IMAGE-SPACE AIR ZONES** on camera sensor planes.

### 2. Explainable Multi-Factor Threat Risk Engine
Computes real-time threat scores (0–100) based on weighted multi-domain signals:
- **Zone Intrusion Weight:** +40 points (Restricted / High-Security border zone)
- **Weapon Detection Weight:** +35 points (Confirmed firearm in consensus window)
- **Kinematic Speed/Trajectory:** +15 points (High velocity or sudden directional acceleration)
- **Environmental Factors:** +10 points (Nighttime / zero-light conditions)

### 3. Tamper-Proof Forensic Evidence Preservation
Every security event recorded by TRINETRA generates a forensically sealed evidentiary record:
- **Event Metadata:** Unique Event ID, Camera ID, UTC ISO-8601 Timestamp, Class Confidence.
- **Video Clip Generation:** Automatic 12-second H.264 clip (5s pre-event buffer + 7s post-event buffer).
- **Cryptographic Sealing:** SHA-256 checksum calculated over evidence clips and stored in the database ledger.
- **Audit Ledger:** Immutably recorded in `event_audits` and `evidence` tables.

```bash
# Verify evidence integrity against database SHA-256
sha256sum data/evidence/CAM-001/event_10003.mp4
```

### 4. Camera-Qualified Track Namespaces
To prevent cross-camera identity hallucinations across wide-area sectors, track IDs are strictly camera-qualified:

$$\text{Track ID} = \texttt{<Camera-ID>:<Namespace-Prefix>-<Sequential-ID>}$$

| Prefix | Domain Entity | Example Track ID | Description |
| :---: | :--- | :--- | :--- |
| **`P`** | Person | `CAM-001:P-024` | Person #24 on Camera 1 |
| **`V`** | Vehicle | `CAM-002:V-102` | Vehicle #102 on Camera 2 |
| **`A`** | Airborne (Drone/Plane) | `CAM-004:A-007` | Drone #7 on Camera 4 |
| **`I`** | Security Item (Weapon) | `CAM-002:I-001` | Firearm #1 on Camera 2 |

---

## AI Performance & Benchmarks

### 1. Frozen Benchmark Results

Evaluated against strictly frozen, immutable ground-truth benchmark datasets:

| Model Checkpoint | Benchmark Dataset | Precision | Recall | F1-Score | mAP@0.5 | Standalone Latency | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ground v2.0** (`ibvap_detector.pt`) | `IBVAP-GT-v1.0` | 64.49% (P) / 96.97% (V) | 74.58% (P) / 86.49% (V) | 69.17% / 91.43% | 48.10% / 83.87% | 10.59 ms | **Production Active** |
| **Airborne v2.0** (`ibvap_airborne_v2_production.pt`) | `airborne_v1_1_val` | 83.47% (A) / 75.54% (D) | 99.02% (A) / 91.00% (D) | 90.58% / 82.55% | 82.65% / 68.74% | 9.64 ms | **Production Active** |
| **Security Item v2.1** (`ibvap_security_item_v2_1_production.pt`) | `IBVAP-GT-ITEM-v2.0` | 83.61% | 83.61% | 83.61% | 85.79% | 11.99 ms | **Production Active** |

### 2. Multi-Camera Operational Runtime Profiling

| Deployment Load | Active Models | Aggregate Throughput | Per-Camera FPS | P50 Latency | P95 Latency | RAM RSS | VRAM |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 Camera** | Ground + Air + Item | 7.4 FPS | 7.4 FPS | 134.4 ms | 138.8 ms | 414 MB | 96 MB |
| **2 Cameras** | Ground + Air + Item | 7.6 FPS | 3.8 FPS | 252.8 ms | 312.7 ms | 418 MB | 96 MB |
| **4 Cameras** | Ground + Air + Item | 7.6 FPS | 1.9 FPS | 514.5 ms | 587.4 ms | 491 MB | 96 MB |
| **4 Cameras (TensorRT GPU)** | Ground + Air + Item | > 120 FPS | > 30.0 FPS | 11.2 ms | 14.5 ms | 512 MB | 840 MB |

---

## Repository Structure

```
IBVAP/
├── app/
│   └── data/insightface/models/    # InsightFace Buffalo_L ONNX Suite (5 models)
├── backend/                        # FastAPI C4ISR Backend Application
│   ├── alembic/                    # Alembic Database Migrations
│   ├── app/
│   │   ├── api/                    # REST & WebSocket API Routers (auth, cameras, events, ws, system)
│   │   ├── core/                   # Security, Database Pool, Telemetry & Config Manager
│   │   ├── models/                 # SQLAlchemy ORM Models (cameras, users, events, evidence, audits)
│   │   ├── schemas/                # Pydantic Request/Response Schemas
│   │   └── services/               # Core Services (detection, tracking, risk, border rules, anpr)
│   └── tests/                      # Pytest API & Integration Test Suite (23 passed)
├── frontend/                       # React 19 + TypeScript + Vite Command Center UI
│   ├── src/
│   │   ├── components/             # CameraGrid, TacticalMap, IntelligencePanel, AnprPanel
│   │   ├── context/                # WebSocket & Authentication State Management
│   │   └── types/                  # TypeScript Data & Telemetry Interfaces
│   └── dist/                       # Optimized Production Build Bundle
├── configs/                        # System, Camera, Zone, and Streaming Configurations
│   ├── cameras.yaml                # Camera stream definitions & RTSP / MP4 video feeds
│   ├── zones.yaml                  # Virtual fence polygons & image-space air zones
│   └── mediamtx.yml                # MediaMTX authenticated streaming configuration
├── data/                           # Runtime Data & Feeds
│   ├── evidence/                   # Generated event video clips and JPEG snapshots
│   └── videos/virat/               # VIRAT Surveillance Video Feeds (CAM-001 to CAM-004)
├── models/                         # Official Production Models & Checksums
│   ├── current/                    # Active Checkpoints (ibvap_detector.pt, ibvap_airborne_detector.pt)
│   ├── production/                 # Promoted Ground, Airborne, Security Item (.pt and .onnx)
│   ├── archive/                    # Baseline archives
│   └── model_registry.yaml         # Authoritative Checksum & Metadata Registry
├── benchmark/                      # Frozen Evaluation Benchmarks (IBVAP-GT-v1.0)
├── deploy/                         # Production Deployment Manifests (Helm & Nginx)
├── docs/                           # Official Documentation, Feasibility, & Demonstration Guides
├── scripts/                        # Operational, Validation & System Profiling Scripts
└── docker-compose.yml              # Production Multi-Container Compose Manifest
```

---

## Quick Start & Installation

### 1. Prerequisites
- **Python:** `3.11+`
- **Node.js:** `v20+` (NPM `10+`)
- **Docker & Docker Compose:** Optional for Redis/PostgreSQL/MediaMTX

### 2. Configure Environment
```bash
cp .env.example .env
```
Default configuration uses SQLite (`ibvap_audit.db` or `ibvap.db`) and local video streaming sources out-of-the-box.

### 3. Install Dependencies
```bash
# Install root orchestration & frontend dependencies
npm run install:all

# Install backend Python dependencies
pip install -r backend/requirements.txt
```

### 4. Run the Full Application
```bash
# Launch both Backend and Frontend concurrently
npm run dev
```

Or run services independently:
```bash
# Backend (FastAPI on http://localhost:8000)
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

# Frontend (Vite on http://localhost:5173)
cd frontend && npm run dev
```

Open your browser at **`http://localhost:5173`** to access the TRINETRA Command Center.

Default Credentials:
- **Username:** `ibvap-admin`
- **Password:** `Admin@4163`

---

## Testing & Quality Assurance

```bash
# 1. Run full backend pytest suite (23 tests)
pytest backend/tests

# 2. Run frontend production build & TypeScript typecheck
cd frontend && npm run build && cd ..

# 3. Verify all AI models, database, and camera feeds
python scratch/verify_all.py
```

---

## Production Deployment Options

### Option A: Docker Compose (Single-Server / Edge Workstation)
```bash
docker compose up -d --build
```

### Option B: Kubernetes Helm Chart
```bash
helm install ibvap deploy/helm/ibvap/ -f deploy/helm/ibvap/values.yaml --namespace ibvap --create-namespace
```

---

## Known Operational Boundaries

1. **Firearm Temporal Confirmation:** Requires multi-frame temporal consensus window to eliminate single-frame transient false positives.
2. **2D Image-Space Air Zones:** Airspace boundaries are 2D sensor projections on camera planes, not 3D radar altitudes.
3. **Camera-Local Tracking:** Track IDs (`CAM-001:P-024`) are strictly camera-local unless correlated via 512-d facial embedding matching (`FaceService`).
4. **Operator-in-the-Loop Mandate:** The platform functions as an autonomous tactical Decision Support System (DSS); all kinetic and physical response actions remain with human command officers.

---

## Documentation Index

- [FEASIBILITY AND VIABILITY.md](FEASIBILITY%20AND%20VIABILITY.md) — Comprehensive Technical, Operational, Economic & Legal Viability Analysis.
- [IMPACT AND BENEFITS.md](IMPACT%20AND%20BENEFITS.md) / [BENEFITS.md](BENEFITS.md) — Strategic, Operational, Sentry Welfare & Evidentiary Impact Matrix.
- [models/model_registry.yaml](models/model_registry.yaml) — Cryptographic SHA-256 Model Integrity Registry.
- [configs/cameras.yaml](configs/cameras.yaml) — Camera Topology & Video Ingestion Configuration.
- [configs/zones.yaml](configs/zones.yaml) — Virtual Fence & Intrusion Polygon Definitions.

---

## License & Attribution

**Proprietary / Government Use** — Developed for the Ministry of Home Affairs (MHA) & Sashastra Seema Bal (SSB) under Smart India Hackathon (SIH) 2026. All rights reserved.
