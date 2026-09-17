from backend.app.core.database import Base
from backend.app.models.camera import Camera
from backend.app.models.user import User
from backend.app.models.event import (
    Track,
    DetectionEvent,
    PlateEvent,
    FaceEvent,
    WatchlistEntry,
    SecurityEvent,
    EventAudit,
)
from backend.app.models.incident import (
    Incident,
    IncidentEvent,
    GlobalEntityRecord,
    EntityObservationRecord,
    IncidentEvidenceLink,
    IncidentDisposition,
)

from backend.app.models.prediction import (
    PredictionRecord,
    PredictionHypothesis,
    PredictionOutcome,
    PredictivePTZActionRecord,
)

from backend.app.models.handover import (
    PTZHandoverChainRecord,
    PTZHandoverRecord,
    CameraReservationRecord,
    TerrainForecastRecord,
)

from backend.app.models.edge_mesh import (
    EdgeNodeRecord,
    EdgeLeaseRecord,
    EdgeOutboxRecord,
    CrossSpectralAssociationRecord,
    EvidenceSyncRecord,
)

from backend.app.models.model_governance import (
    ModelRecord,
    ModelVersionRecord,
    DatasetRecord,
    TrainingRunRecord,
    ValidationRunRecord,
    CanaryDeploymentRecord,
    EdgeDeploymentRecord,
    ModelHealthRecord,
    RollbackEventRecord,
    PromotionDecisionRecord,
)

from backend.app.models.ai_quality import (
    OperatorFeedbackEntity,
    FeedbackReviewAuditEntity,
    HardCaseEntity,
    FailureClusterEntity,
    CameraQualityEntity,
    PoisoningAlertEntity,
)

from backend.app.models.multimodal import (
    SensorModality,
    DataOrigin,
    CalibrationStatus,
    SynchronizationStatus,
    RegistrationMode,
    SensorHealthStatus,
    SensorRecord,
    SensorObservationRecord,
    SensorCalibrationRecord,
    SensorSynchronizationRecord,
    ThermalDatasetRecord,
    MultimodalSpectralAssociationEntity,
    MultimodalTrackEntity,
    SensorHealthEntity,
    ThermalDatasetSampleEntity,
    ThermalAnnotationEntity,
    ThermalEdgeDeploymentPackageEntity,
)

__all__ = [
    "Base",
    "Camera",
    "User",
    "Track",
    "DetectionEvent",
    "PlateEvent",
    "FaceEvent",
    "WatchlistEntry",
    "SecurityEvent",
    "EventAudit",
    "Incident",
    "IncidentEvent",
    "GlobalEntityRecord",
    "EntityObservationRecord",
    "IncidentEvidenceLink",
    "IncidentDisposition",
    "PredictionRecord",
    "PredictionHypothesis",
    "PredictionOutcome",
    "PredictivePTZActionRecord",
    "PTZHandoverChainRecord",
    "PTZHandoverRecord",
    "CameraReservationRecord",
    "TerrainForecastRecord",
    "EdgeNodeRecord",
    "EdgeLeaseRecord",
    "EdgeOutboxRecord",
    "CrossSpectralAssociationRecord",
    "EvidenceSyncRecord",
    "ModelRecord",
    "ModelVersionRecord",
    "DatasetRecord",
    "TrainingRunRecord",
    "ValidationRunRecord",
    "CanaryDeploymentRecord",
    "EdgeDeploymentRecord",
    "ModelHealthRecord",
    "RollbackEventRecord",
    "PromotionDecisionRecord",
    "OperatorFeedbackEntity",
    "FeedbackReviewAuditEntity",
    "HardCaseEntity",
    "FailureClusterEntity",
    "CameraQualityEntity",
    "PoisoningAlertEntity",
    "SensorModality",
    "DataOrigin",
    "CalibrationStatus",
    "SynchronizationStatus",
    "RegistrationMode",
    "SensorHealthStatus",
    "SensorRecord",
    "SensorObservationRecord",
    "SensorCalibrationRecord",
    "SensorSynchronizationRecord",
    "ThermalDatasetRecord",
    "MultimodalSpectralAssociationEntity",
    "MultimodalTrackEntity",
    "SensorHealthEntity",
    "ThermalDatasetSampleEntity",
    "ThermalAnnotationEntity",
    "ThermalEdgeDeploymentPackageEntity",
]


