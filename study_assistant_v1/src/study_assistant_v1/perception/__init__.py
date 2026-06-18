"""Perception module interfaces and implementations.

Each module defines a Protocol (structural interface) plus a Null implementation
that returns safe no-op values. Real implementations (DeepFace, MediaPipe, etc.)
are added later and selected via StudyConfig.
"""

from study_assistant_v1.perception.emotion_recognizer import EmotionRecognizer, NullEmotionRecognizer
from study_assistant_v1.perception.engagement_detector import EngagementDetector, NullEngagementDetector
from study_assistant_v1.perception.context_provider import ContextProvider, NullContextProvider
from study_assistant_v1.perception.study_config import StudyConfig, build_perception

__all__ = [
    "EmotionRecognizer",
    "NullEmotionRecognizer",
    "EngagementDetector",
    "NullEngagementDetector",
    "ContextProvider",
    "NullContextProvider",
    "StudyConfig",
    "build_perception",
]
