"""Rolling engagement-score tracking and re-engagement decision, following Lalwani et al."""

from typing import ClassVar

from reachy_mini_conversation_app.intervention_monitor import InterventionMonitor


class EngagementMonitor(InterventionMonitor[float]):
    """Tracks recent engagement scores and decides when sustained disengagement warrants intervention."""

    ENGAGEMENT_THRESHOLD: ClassVar[float] = 0.93

    def average_score(self) -> float | None:
        """Return the mean engagement score over the current window, or None when empty."""
        if not self._samples:
            return None
        return sum(sample.value for sample in self._samples) / len(self._samples)

    def _signal_active(self) -> bool:
        """Return whether the windowed average has fallen below the engagement threshold."""
        average = self.average_score()
        return average is not None and average < self.ENGAGEMENT_THRESHOLD
