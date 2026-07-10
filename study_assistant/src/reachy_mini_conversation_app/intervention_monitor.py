"""Shared rolling-window gating for signal-driven intervention monitors."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, ClassVar
from dataclasses import dataclass


SampleValueT = TypeVar("SampleValueT")


@dataclass(frozen=True)
class MonitorSample(Generic[SampleValueT]):
    """A single timestamped observation."""

    value: SampleValueT
    timestamp: float


class InterventionMonitor(ABC, Generic[SampleValueT]):
    """Rolling sample window plus the gates deciding when an intervention may fire."""

    WINDOW_SECONDS: ClassVar[float] = 30.0
    INTERACTION_COOLDOWN_SECONDS: ClassVar[float] = 60.0
    INTERVENTION_COOLDOWN_SECONDS: ClassVar[float] = 60.0

    def __init__(self) -> None:
        """Initialize an empty rolling window with no prior trigger."""
        self._samples: list[MonitorSample[SampleValueT]] = []
        self._last_trigger_time: float | None = None

    def record(self, value: SampleValueT, timestamp: float) -> None:
        """Add an observation, dropping samples older than WINDOW_SECONDS."""
        self._samples.append(MonitorSample(value, timestamp))
        cutoff = timestamp - self.WINDOW_SECONDS
        self._samples = [sample for sample in self._samples if sample.timestamp >= cutoff]

    @abstractmethod
    def _signal_active(self) -> bool:
        """Return whether the windowed signal currently warrants an intervention."""
        ...

    def should_intervene(self, now: float, response_done: bool, last_activity_time: float) -> bool:
        """Return whether the signal, an idle conversation, and both cooldowns all hold."""
        if not self._signal_active():
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

    @property
    def last_trigger_time(self) -> float | None:
        """Return the monotonic time of the last intervention, or None if it hasn't fired yet."""
        return self._last_trigger_time
