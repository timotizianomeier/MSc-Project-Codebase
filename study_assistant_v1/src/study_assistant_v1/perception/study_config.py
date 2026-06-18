"""StudyConfig dataclass and perception factory.

This is research-critical code — understand it before running a study session.

StudyConfig captures the condition and backend choices for a single session.
It is loaded from environment variables at startup (see .env).
Every CSV log row includes the condition name, so your analysis can filter
by condition without needing any metadata outside the log file itself.

Condition semantics (as of 2026-06-18, may be refined before data collection):
  intervention  — full system: real perception backends, robot intervenes
  control       — Null backends, robot present but does not intervene

Adding a new backend later:
  1. Write a class that satisfies the relevant Protocol (EmotionRecognizer etc.)
  2. Add a new elif branch in the corresponding _build_* helper below
  3. Add the new backend name to the .env.example comment
  No other files need to change.
"""

import os
from dataclasses import dataclass, field

from study_assistant_v1.perception.emotion_recognizer import EmotionRecognizer, NullEmotionRecognizer
from study_assistant_v1.perception.engagement_detector import EngagementDetector, NullEngagementDetector
from study_assistant_v1.perception.context_provider import ContextProvider, NullContextProvider


VALID_CONDITIONS = {"intervention", "control"}


@dataclass
class StudyConfig:
    """Per-session study configuration.

    Loaded from environment variables at session start.
    All fields are written into the event log so the CSV is self-describing.
    """

    # Which experimental condition this session runs under.
    # Must be one of VALID_CONDITIONS.
    condition: str = field(default="control")

    # A unique identifier for this participant, e.g. "p01", "p02".
    # Used to link log rows to participant-level data (ADHD vs neurotypical group
    # is recorded in a separate participant registry, not in the log itself).
    participant_id: str = field(default="unknown")

    # Which perception backend to use for each module.
    # "null" selects the Null implementation (no-op).
    # Additional backends will be added as they are implemented.
    emotion_backend: str = field(default="null")
    engagement_backend: str = field(default="null")
    context_backend: str = field(default="null")

    # Master switch: if False, the robot never triggers an LLM intervention
    # regardless of what the perception modules report.
    intervention_enabled: bool = field(default=False)

    def __post_init__(self) -> None:
        if self.condition not in VALID_CONDITIONS:
            raise ValueError(
                f"Invalid condition '{self.condition}'. Must be one of: {sorted(VALID_CONDITIONS)}"
            )
        # Enforce internal consistency: the control condition must have all
        # Null backends and no interventions.  This makes it harder to
        # accidentally run a mixed configuration.
        if self.condition == "control":
            if self.intervention_enabled:
                raise ValueError("condition='control' requires intervention_enabled=False")
            if any(b != "null" for b in [self.emotion_backend, self.engagement_backend, self.context_backend]):
                raise ValueError("condition='control' requires all backends set to 'null'")


def load_study_config_from_env() -> StudyConfig:
    """Build a StudyConfig from environment variables.

    Called once at session startup. All values come from .env (or the shell
    environment), so the researcher sets the condition by editing .env before
    each participant session.

    Environment variables:
        STUDY_CONDITION          intervention | control  (default: control)
        STUDY_PARTICIPANT_ID     e.g. p01               (default: unknown)
        STUDY_EMOTION_BACKEND    null | deepface         (default: null)
        STUDY_ENGAGEMENT_BACKEND null | mediapipe        (default: null)
        STUDY_CONTEXT_BACKEND    null | pdf              (default: null)
        STUDY_INTERVENTION       true | false            (default: false)
    """
    condition = os.getenv("STUDY_CONDITION", "control").strip().lower()
    participant_id = os.getenv("STUDY_PARTICIPANT_ID", "unknown").strip()
    emotion_backend = os.getenv("STUDY_EMOTION_BACKEND", "null").strip().lower()
    engagement_backend = os.getenv("STUDY_ENGAGEMENT_BACKEND", "null").strip().lower()
    context_backend = os.getenv("STUDY_CONTEXT_BACKEND", "null").strip().lower()

    raw_intervention = os.getenv("STUDY_INTERVENTION", "false").strip().lower()
    intervention_enabled = raw_intervention in {"1", "true", "yes", "on"}

    return StudyConfig(
        condition=condition,
        participant_id=participant_id,
        emotion_backend=emotion_backend,
        engagement_backend=engagement_backend,
        context_backend=context_backend,
        intervention_enabled=intervention_enabled,
    )


def _build_emotion_recognizer(backend: str) -> EmotionRecognizer:
    if backend == "null":
        return NullEmotionRecognizer()
    raise ValueError(f"Unknown emotion_backend: '{backend}'. Available: 'null'")


def _build_engagement_detector(backend: str) -> EngagementDetector:
    if backend == "null":
        return NullEngagementDetector()
    raise ValueError(f"Unknown engagement_backend: '{backend}'. Available: 'null'")


def _build_context_provider(backend: str) -> ContextProvider:
    if backend == "null":
        return NullContextProvider()
    raise ValueError(f"Unknown context_backend: '{backend}'. Available: 'null'")


def build_perception(
    config: StudyConfig,
) -> tuple[EmotionRecognizer, EngagementDetector, ContextProvider]:
    """Instantiate the three perception modules from a StudyConfig.

    This is the single place where backend strings become live objects.
    To add a new backend: add an elif branch in the relevant _build_* helper.
    """
    return (
        _build_emotion_recognizer(config.emotion_backend),
        _build_engagement_detector(config.engagement_backend),
        _build_context_provider(config.context_backend),
    )
