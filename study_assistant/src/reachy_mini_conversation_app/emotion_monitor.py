"""Rolling emotion-window tracking and intervention decision, following Lalwani et al."""

from __future__ import annotations
from dataclasses import dataclass


NEGATIVE_EMOTIONS: frozenset[str] = frozenset({"angry", "disgust", "fear", "sad"})


@dataclass(frozen=True)
class EmotionSample:
    """A single classified frame's dominant emotion."""

    emotion: str
    timestamp: float
