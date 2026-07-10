"""Rolling emotion-window tracking and intervention decision, following Lalwani et al."""

from typing import ClassVar

from reachy_mini_conversation_app.intervention_monitor import InterventionMonitor


NEGATIVE_EMOTIONS: frozenset[str] = frozenset({"angry", "disgust", "fear", "sad"})


class EmotionMonitor(InterventionMonitor[str]):
    """Tracks recent dominant emotions and decides when sustained negative affect warrants intervention."""

    NEGATIVE_THRESHOLD: ClassVar[float] = 0.40

    def negative_share(self) -> float:
        """Return the fraction of samples in the current window that are negative emotions."""
        if not self._samples:
            return 0.0
        negative_count = sum(1 for sample in self._samples if sample.value in NEGATIVE_EMOTIONS)
        return negative_count / len(self._samples)

    def _signal_active(self) -> bool:
        """Return whether the negative-emotion share exceeds the threshold."""
        return self.negative_share() > self.NEGATIVE_THRESHOLD
