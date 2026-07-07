"""Rolling emotion-window tracking and intervention decision, following Lalwani et al."""

from __future__ import annotations
from typing import ClassVar
from dataclasses import dataclass


NEGATIVE_EMOTIONS: frozenset[str] = frozenset({"angry", "disgust", "fear", "sad"})


@dataclass(frozen=True)
class EmotionSample:
    """A single classified frame's dominant emotion."""

    emotion: str
    timestamp: float


class EmotionMonitor:
    """Tracks recent dominant emotions and decides when sustained negative affect warrants intervention."""

    WINDOW_SECONDS: ClassVar[float] = 30.0
    NEGATIVE_THRESHOLD: ClassVar[float] = 0.40
    INTERACTION_COOLDOWN_SECONDS: ClassVar[float] = 60.0
    INTERVENTION_COOLDOWN_SECONDS: ClassVar[float] = 60.0

    def __init__(self) -> None:
        """Initialize an empty rolling window with no prior trigger."""
        self._samples: list[EmotionSample] = []
        self._last_trigger_time: float | None = None

    def record(self, emotion: str, timestamp: float) -> None:
        """Add a classified frame's dominant emotion, dropping samples older than WINDOW_SECONDS."""
        self._samples.append(EmotionSample(emotion, timestamp))
        cutoff = timestamp - self.WINDOW_SECONDS
        self._samples = [sample for sample in self._samples if sample.timestamp >= cutoff]

    def negative_share(self) -> float:
        """Return the fraction of samples in the current window that are negative emotions."""
        if not self._samples:
            return 0.0
        negative_count = sum(1 for sample in self._samples if sample.emotion in NEGATIVE_EMOTIONS)
        return negative_count / len(self._samples)

    def should_intervene(self, now: float, response_done: bool, last_activity_time: float) -> bool:
        """Return whether negative affect, an idle conversation, and cooldown elapsed all hold."""
        if self.negative_share() <= self.NEGATIVE_THRESHOLD:
            return False
        if not response_done:
            return False
        if now - last_activity_time <= self.INTERACTION_COOLDOWN_SECONDS:
            return False
        if self._last_trigger_time is not None and now - self._last_trigger_time <= self.INTERVENTION_COOLDOWN_SECONDS:
            return False
        return True

    def mark_intervened(self, now: float) -> None:
        """Record that an intervention was just sent, starting the intervention cooldown."""
        self._last_trigger_time = now
