"""Rolling emotion-window tracking and intervention decision, following Lalwani et al."""

from reachy_mini_conversation_app.intervention_monitor import InterventionMonitor


NEGATIVE_EMOTIONS: frozenset[str] = frozenset({"angry", "disgust", "fear", "sad"})


def negative_mass(scores: dict[str, float]) -> float:
    """Sum the probability mass of the negative emotion classes in a 0-1 scores dict."""
    return sum(scores.get(emotion, 0.0) for emotion in NEGATIVE_EMOTIONS)


class EmotionMonitor(InterventionMonitor[float]):
    """Tracks windowed negative-emotion probability mass and decides when sustained negative affect warrants intervention.

    Samples are per-poll negative masses (see negative_mass), so an ambiguous frame
    contributes its actual evidence instead of a full vote for whichever label won.
    """

    def __init__(self, negative_threshold: float = 0.40) -> None:
        """Initialize with the windowed-mass threshold above which the signal is active."""
        super().__init__()
        self.NEGATIVE_THRESHOLD = negative_threshold

    def negative_share(self) -> float:
        """Return the mean negative probability mass over the current window (0-1)."""
        if not self._samples:
            return 0.0
        return sum(sample.value for sample in self._samples) / len(self._samples)

    def _signal_active(self) -> bool:
        """Return whether the windowed negative mass exceeds the threshold."""
        return self.negative_share() > self.NEGATIVE_THRESHOLD
