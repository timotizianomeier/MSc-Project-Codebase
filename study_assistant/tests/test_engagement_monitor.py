import pytest

from reachy_mini_conversation_app.engagement_monitor import EngagementMonitor


def test_average_score_with_no_samples_is_none() -> None:
    """An empty window has no average — None, not a fabricated score."""
    monitor = EngagementMonitor()

    assert monitor.average_score() is None


def test_should_intervene_false_on_empty_window() -> None:
    """A fresh monitor with no scores must not fire, even with every gate open."""
    monitor = EngagementMonitor()

    result = monitor.should_intervene(now=100.0, response_done=True, last_activity_time=0.0)

    assert result is False


def test_average_score_computes_mean_of_scores() -> None:
    """The window average should be the plain mean of the recorded scores."""
    monitor = EngagementMonitor()
    for score in (0.5, 0.9, 1.0):
        monitor.record(score, timestamp=0.0)

    assert monitor.average_score() == pytest.approx(0.8)


def test_record_evicts_scores_older_than_window() -> None:
    """A score past WINDOW_SECONDS should no longer drag the average down."""
    monitor = EngagementMonitor()
    monitor.record(0.1, timestamp=0.0)

    monitor.record(1.0, timestamp=EngagementMonitor.WINDOW_SECONDS + 1.0)

    assert monitor.average_score() == pytest.approx(1.0)


def test_should_intervene_false_when_average_exactly_at_threshold() -> None:
    """An average exactly at ENGAGEMENT_THRESHOLD must not fire — it must fall below."""
    monitor = EngagementMonitor()
    monitor.record(EngagementMonitor.ENGAGEMENT_THRESHOLD, timestamp=0.0)

    result = monitor.should_intervene(now=100.0, response_done=True, last_activity_time=0.0)

    assert result is False


def test_should_intervene_true_when_disengaged_and_gates_open() -> None:
    """A window averaging below the threshold with an idle conversation should fire."""
    monitor = EngagementMonitor()
    for score in (0.4, 0.5, 0.6):
        monitor.record(score, timestamp=0.0)

    result = monitor.should_intervene(now=100.0, response_done=True, last_activity_time=0.0)

    assert result is True


def test_should_intervene_false_when_response_not_done() -> None:
    """The inherited gating must still hold: an active response blocks intervention."""
    monitor = EngagementMonitor()
    for score in (0.4, 0.5, 0.6):
        monitor.record(score, timestamp=0.0)

    result = monitor.should_intervene(now=100.0, response_done=False, last_activity_time=0.0)

    assert result is False
