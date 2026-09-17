# IBVAP — Video Data Directory & Placement Guidelines
# TRINETRA — Video Data Directory & Placement Guidelines

## 1. Intentional Repository Exclusion
Surveillance video files (`*.mp4`, `*.avi`, `*.mkv`, etc.) are **strictly excluded from this Git repository** in accordance with:
- **GitHub Storage Limits:** Raw surveillance feeds exceed GitHub file size limitations (100 MB per file limit).
- **Data Governance & Licensing:** Video datasets may be subject to research distribution restrictions and export controls.
- **Repository Cleanliness:** Preventing multi-gigabyte bloat for developers cloning the software codebase.

---

## 2. Directory Layout & Expected File Locations

Place your local test and operational surveillance recordings in this directory hierarchy:

```
data/videos/
├── README.md               # (This guidance file - tracked in Git)
├── virat/                  # Local VIRAT evaluation footage (untracked)
│   ├── VIRAT_S_000001.mp4
│   ├── VIRAT_S_000002.mp4
│   ├── VIRAT_S_000003.mp4
│   └── VIRAT_S_000004.mp4
├── live_streams/           # Optional recorded RTSP feed dumps
└── phase1_test.mp4         # Local unit test video fixture
```

---

## 3. VIRAT Video Dataset Reference

IBVAP benchmarks and offline retrospective audits utilize the **VIRAT Video Dataset** (Release 2.0):
TRINETRA benchmarks and offline retrospective audits utilize the **VIRAT Video Dataset** (Release 2.0):
- **Source:** [VIRAT Video Dataset](https://viratdata.org/)
- **Description:** Real-world surveillance footage captured from high-angle CCTV and airborne cameras with realistic human-vehicle interactions, occlusions, and border outpost scenarios.
- **License / Terms of Use:** Research and educational use only under the VIRAT dataset agreement. Users must obtain the files directly from the official portal or authorized research mirrors.
- **Format:** H.264 / MP4 at 1080p / 720p resolution, 30 FPS.

---

## 4. Configuring Custom Video Feeds in IBVAP
## 4. Configuring Custom Video Feeds in TRINETRA

To configure local video files as emulated camera feeds in IBVAP:
To configure local video files as emulated camera feeds in TRINETRA:

1. Copy your test `.mp4` files into `data/videos/`.
2. Edit `configs/cameras.yaml` to specify the local path under the `source` parameter:
   ```yaml
   cameras:
     - id: "CAM-001"
       name: "Outpost Sector North"
       source: "data/videos/virat/VIRAT_S_000001.mp4"
       location: "BOP-TEST-01"
       resolution: "1920x1080"
       fps: 30
       is_active: true
   ```
3. Restart the backend service or launch the ingest engine via:
   ```bash
   python -m uvicorn backend.app.main:app --reload
   ```
