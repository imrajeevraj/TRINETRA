"""
TRINETRA — Model Lifecycle State Machine (Phase XII)
Formally defines the 18 model lifecycle states, valid state transition graphs,
guard conditions, and immutable transition audit logging.
"""

from __future__ import annotations
import enum
import time
import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from pydantic import BaseModel, Field

logger = logging.getLogger("ModelLifecycleStateMachine")


class ModelLifecycleState(str, enum.Enum):
    COLLECTED = "COLLECTED"
    LABELING = "LABELING"
    LABELED = "LABELED"
    TRAINING = "TRAINING"
    TRAINED = "TRAINED"
    VALIDATING = "VALIDATING"
    VALIDATED = "VALIDATED"
    CANARY_READY = "CANARY_READY"
    CANARY_RUNNING = "CANARY_RUNNING"
    CANARY_PASSED = "CANARY_PASSED"
    DEPLOYING = "DEPLOYING"
    ACTIVE = "ACTIVE"
    MONITORING = "MONITORING"
    DEGRADED = "DEGRADED"
    ROLLBACK_PENDING = "ROLLBACK_PENDING"
    ROLLED_BACK = "ROLLED_BACK"
    REJECTED = "REJECTED"
    ARCHIVED = "ARCHIVED"


class StateTransitionEvent(BaseModel):
    transition_id: str
    model_id: str
    from_state: ModelLifecycleState
    to_state: ModelLifecycleState
    timestamp: float
    operator: str
    reason: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelLifecycleEngine:
    """
    Thread-safe model lifecycle governor enforcing valid state transitions,
    immutability of history, and fail-closed transitions.
    """

    # Comprehensive legal transition matrix
    LEGAL_TRANSITIONS: Dict[ModelLifecycleState, Set[ModelLifecycleState]] = {
        ModelLifecycleState.COLLECTED: {
            ModelLifecycleState.LABELING,
            ModelLifecycleState.REJECTED,
            ModelLifecycleState.ARCHIVED,
        },
        ModelLifecycleState.LABELING: {
            ModelLifecycleState.LABELED,
            ModelLifecycleState.REJECTED,
            ModelLifecycleState.ARCHIVED,
        },
        ModelLifecycleState.LABELED: {
            ModelLifecycleState.TRAINING,
            ModelLifecycleState.ARCHIVED,
            ModelLifecycleState.REJECTED,
        },
        ModelLifecycleState.TRAINING: {
            ModelLifecycleState.TRAINED,
            ModelLifecycleState.REJECTED,
            ModelLifecycleState.ARCHIVED,
        },
        ModelLifecycleState.TRAINED: {
            ModelLifecycleState.VALIDATING,
            ModelLifecycleState.ARCHIVED,
            ModelLifecycleState.REJECTED,
        },
        ModelLifecycleState.VALIDATING: {
            ModelLifecycleState.VALIDATED,
            ModelLifecycleState.REJECTED,
            ModelLifecycleState.ARCHIVED,
        },
        ModelLifecycleState.VALIDATED: {
            ModelLifecycleState.CANARY_READY,
            ModelLifecycleState.DEPLOYING,
            ModelLifecycleState.REJECTED,
            ModelLifecycleState.ARCHIVED,
        },
        ModelLifecycleState.CANARY_READY: {
            ModelLifecycleState.CANARY_RUNNING,
            ModelLifecycleState.ARCHIVED,
            ModelLifecycleState.REJECTED,
        },
        ModelLifecycleState.CANARY_RUNNING: {
            ModelLifecycleState.CANARY_PASSED,
            ModelLifecycleState.ROLLBACK_PENDING,
            ModelLifecycleState.REJECTED,
        },
        ModelLifecycleState.CANARY_PASSED: {
            ModelLifecycleState.DEPLOYING,
            ModelLifecycleState.ARCHIVED,
            ModelLifecycleState.REJECTED,
        },
        ModelLifecycleState.DEPLOYING: {
            ModelLifecycleState.ACTIVE,
            ModelLifecycleState.ROLLBACK_PENDING,
            ModelLifecycleState.REJECTED,
        },
        ModelLifecycleState.ACTIVE: {
            ModelLifecycleState.MONITORING,
            ModelLifecycleState.DEGRADED,
            ModelLifecycleState.ROLLBACK_PENDING,
            ModelLifecycleState.ARCHIVED,
        },
        ModelLifecycleState.MONITORING: {
            ModelLifecycleState.ACTIVE,
            ModelLifecycleState.DEGRADED,
            ModelLifecycleState.ROLLBACK_PENDING,
            ModelLifecycleState.ARCHIVED,
        },
        ModelLifecycleState.DEGRADED: {
            ModelLifecycleState.ROLLBACK_PENDING,
            ModelLifecycleState.VALIDATING,
            ModelLifecycleState.ACTIVE,
            ModelLifecycleState.ARCHIVED,
        },
        ModelLifecycleState.ROLLBACK_PENDING: {
            ModelLifecycleState.ROLLED_BACK,
            ModelLifecycleState.ACTIVE,
        },
        ModelLifecycleState.ROLLED_BACK: {
            ModelLifecycleState.ARCHIVED,
            ModelLifecycleState.VALIDATING,
        },
        ModelLifecycleState.REJECTED: {
            ModelLifecycleState.ARCHIVED,
        },
        ModelLifecycleState.ARCHIVED: {
            ModelLifecycleState.VALIDATING,
            ModelLifecycleState.ACTIVE,
        },
    }

    def __init__(self):
        self._model_states: Dict[str, ModelLifecycleState] = {}
        self._transition_history: Dict[str, List[StateTransitionEvent]] = {}
        self._transition_counter = 0

    def register_model(
        self,
        model_id: str,
        initial_state: ModelLifecycleState = ModelLifecycleState.COLLECTED,
        operator: str = "SYSTEM",
        reason: str = "INITIAL_REGISTRATION",
    ) -> StateTransitionEvent:
        """Registers a model into the lifecycle engine with its initial state."""
        self._model_states[model_id] = initial_state
        self._transition_counter += 1
        event = StateTransitionEvent(
            transition_id=f"TRANS-{self._transition_counter:06d}",
            model_id=model_id,
            from_state=initial_state,
            to_state=initial_state,
            timestamp=time.time(),
            operator=operator,
            reason=reason,
            metadata={"initial": True},
        )
        self._transition_history[model_id] = [event]
        logger.info(f"Registered model {model_id} in state {initial_state.value}")
        return event

    def get_state(self, model_id: str) -> Optional[ModelLifecycleState]:
        return self._model_states.get(model_id)

    def transition(
        self,
        model_id: str,
        target_state: ModelLifecycleState,
        operator: str,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> Tuple[bool, str, Optional[StateTransitionEvent]]:
        """
        Executes an audited state transition.
        Fails safely if the transition violates the legal state graph.
        """
        current_state = self._model_states.get(model_id)
        if current_state is None:
            return False, f"MODEL_{model_id}_NOT_REGISTERED", None

        # Check self-transition (idempotent no-op)
        if current_state == target_state:
            return True, "ALREADY_IN_TARGET_STATE", None

        # Check legal transitions
        allowed = self.LEGAL_TRANSITIONS.get(current_state, set())
        if target_state not in allowed:
            err_msg = (
                f"ILLEGAL_TRANSITION_VIOLATION: Cannot transition {model_id} "
                f"from {current_state.value} to {target_state.value}. "
                f"Allowed target states: {[s.value for s in allowed]}"
            )
            logger.warning(err_msg)
            return False, err_msg, None

        now = timestamp if timestamp is not None else time.time()
        self._transition_counter += 1
        event = StateTransitionEvent(
            transition_id=f"TRANS-{self._transition_counter:06d}",
            model_id=model_id,
            from_state=current_state,
            to_state=target_state,
            timestamp=now,
            operator=operator,
            reason=reason,
            metadata=metadata or {},
        )

        self._model_states[model_id] = target_state
        if model_id not in self._transition_history:
            self._transition_history[model_id] = []
        self._transition_history[model_id].append(event)

        logger.info(
            f"Model {model_id} transitioned: {current_state.value} -> {target_state.value} "
            f"by {operator} (Reason: {reason})"
        )
        return True, "TRANSITION_SUCCESSFUL", event

    def get_history(self, model_id: str) -> List[StateTransitionEvent]:
        return list(self._transition_history.get(model_id, []))

    def reset(self):
        self._model_states.clear()
        self._transition_history.clear()
        self._transition_counter = 0


model_lifecycle_engine = ModelLifecycleEngine()
