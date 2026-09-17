"""
TRINETRA — Normalized Prediction, Hypothesis & Outcome Database Models (Phase IX)
PostgreSQL schema supporting trajectory forecasts, predictive PTZ actions, and actual-vs-predicted outcomes.
"""

from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship

from backend.app.core.database import Base


class PredictionRecord(Base):
    """Primary persistent record of a trajectory / camera transition prediction."""
    __tablename__ = "predictions"

    prediction_id = Column(String(32), primary_key=True, index=True) # e.g. "PRED-2026-000001"
    entity_id = Column(String(64), nullable=False, index=True)       # e.g. "GLOBAL-PERSON-00001"
    current_camera = Column(String(32), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    predicted_camera = Column(String(32), nullable=True)
    predicted_zone = Column(String(64), nullable=True)
    eta_min_sec = Column(Float, nullable=True)
    eta_max_sec = Column(Float, nullable=True)
    typical_eta_sec = Column(Float, nullable=True)
    confidence = Column(Float, default=0.5, nullable=False)
    status = Column(String(32), default="PREDICTED", index=True, nullable=False) # PREDICTED, HIT, PARTIAL_HIT, MISS, EXPIRED
    forecast_method = Column(String(32), default="KALMAN_CV", nullable=False)
    forecast_version = Column(String(16), default="v1.0", nullable=False)

    hypotheses = relationship("PredictionHypothesis", back_populates="prediction", cascade="all, delete-orphan")
    outcome = relationship("PredictionOutcome", uselist=False, back_populates="prediction", cascade="all, delete-orphan")


class PredictionHypothesis(Base):
    """Ranked multi-hypothesis destination alternatives."""
    __tablename__ = "prediction_hypotheses"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    prediction_id = Column(String(32), ForeignKey("predictions.prediction_id", ondelete="CASCADE"), nullable=False, index=True)
    rank = Column(Integer, default=1, nullable=False)
    candidate_camera = Column(String(32), nullable=False)
    candidate_zone = Column(String(64), nullable=True)
    eta_sec = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    reason = Column(String(256), nullable=True)

    prediction = relationship("PredictionRecord", back_populates="hypotheses")


class PredictionOutcome(Base):
    """Evaluation score comparing predicted transition with actual camera arrival."""
    __tablename__ = "prediction_outcomes"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    prediction_id = Column(String(32), ForeignKey("predictions.prediction_id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    actual_camera = Column(String(32), nullable=True)
    actual_arrival_sec = Column(Float, nullable=True)
    eta_error_sec = Column(Float, nullable=True)
    outcome = Column(String(32), nullable=False, index=True) # HIT, PARTIAL_HIT, MISS, EXPIRED, UNOBSERVABLE
    evaluated_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    details = Column(Text, nullable=True)

    prediction = relationship("PredictionRecord", back_populates="outcome")


class PredictivePTZActionRecord(Base):
    """Auditable log of anticipatory PTZ positioning commands."""
    __tablename__ = "predictive_ptz_actions"

    action_id = Column(String(32), primary_key=True, index=True) # e.g. "PRECUE-ACT-00001"
    prediction_id = Column(String(32), nullable=False, index=True)
    entity_id = Column(String(64), nullable=False)
    target_camera = Column(String(32), nullable=False)
    requested_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    ready_at = Column(DateTime, nullable=True)
    observed_at = Column(DateTime, nullable=True)
    lead_time_sec = Column(Float, nullable=True)
    state = Column(String(32), default="PRE_CUE_REQUESTED", nullable=False)
    pan = Column(Float, default=0.0)
    tilt = Column(Float, default=0.0)
    zoom = Column(Float, default=1.0)
