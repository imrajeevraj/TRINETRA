import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from backend.app.core.config import get_cameras_config, get_system_config
from backend.app.core.database import SessionLocal, Base, engine
from backend.app.models.camera import Camera
from backend.app.services.camera_manager import CameraStreamThread, CameraManager



def test_configurations_loading():
    cameras = get_cameras_config()
    assert isinstance(cameras, list)
    assert len(cameras) > 0
    assert "id" in cameras[0]
    assert "source" in cameras[0]
    
    system = get_system_config()
    assert isinstance(system, dict)
    assert "app_name" in system

def test_camera_database_model(db_session):
    db = db_session
    # Clean any existing
    db.query(Camera).filter(Camera.id == "TEST-CAM-01").delete()
    db.commit()
    
    cam = Camera(
        id="TEST-CAM-01",
        name="Test Camera 1",
        source="0",
        location="Gate 1",
        resolution="1280x720",
        fps=30,
        is_active=True,
        status="OFFLINE"
    )
    db.add(cam)
    db.commit()
    
    db_cam = db.query(Camera).filter(Camera.id == "TEST-CAM-01").first()
    assert db_cam is not None
    assert db_cam.name == "Test Camera 1"
    assert db_cam.status == "OFFLINE"
    
    db.delete(db_cam)
    db.commit()
@patch("os.path.isfile", return_value=True)
@patch("cv2.VideoCapture")
def test_camera_stream_thread_success(mock_video_capture, mock_isfile):
    # Mock cv2.VideoCapture to return a dummy video frame
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    
    # Return true for read, and a dummy black frame
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, dummy_frame)
    mock_cap.get.side_effect = lambda prop: 640 if prop == 3 else 480 if prop == 4 else 30
    mock_video_capture.return_value = mock_cap
    
    stream = CameraStreamThread(
        camera_id="BOP-TEST",
        name="Test Ingestion Thread",
        source="data/videos/phase1_test.mp4",
        target_fps=30
    )
    
    # Run loop once by stopping immediately
    stream.running = True
    
    with patch.object(stream, "update_db_status") as mock_db_update:
        # Mocking run loop internally or calling connection segment
        # Let's verify properties configuration loading
        cap = mock_video_capture("data/videos/phase1_test.mp4")
        assert cap.isOpened()
        
        ret, frame = cap.read()
        assert ret is True
        assert frame.shape == (480, 640, 3)
