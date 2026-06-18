"""EngagementDetector interface and Null implementation."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class EngagementDetector(Protocol):
    """Estimates whether the student is engaged with their work.

    Real implementations use signals such as gaze direction, head pose,
    or activity level from the robot's camera.
    The Null implementation always reports full engagement so no
    inattention events are emitted in the control condition.
    """

    def get_engagement(self) -> float | None:
        """Return an engagement score in [0.0, 1.0], or None if unavailable.

        1.0 = fully engaged, 0.0 = fully disengaged.
        Implementations define their own scale and threshold.
        """
        ...

    def is_inattentive(self) -> bool:
        """Return True if the student appears inattentive.

        This is the method your intervention logic should call.
        The threshold is owned by each implementation so it can be
        tuned per-backend without changing the caller.
        """
        ...

    def start(self) -> None:
        """Start any background threads or resources."""
        ...

    def stop(self) -> None:
        """Release resources and stop background threads."""
        ...


class NullEngagementDetector:
    """No-op implementation used in the control condition.

    Always reports full engagement (score=1.0, is_inattentive=False)
    so no inattention events are generated and the robot never intervenes.
    """

    def get_engagement(self) -> float | None:
        return 1.0

    def is_inattentive(self) -> bool:
        return False

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass
