from __future__ import annotations

from reachy_mini_conversation_app.emotion_monitor import EmotionMonitor


def test_negative_share_with_no_samples_is_zero() -> None:
    """An empty window should read as no negative signal, not divide by zero."""
    monitor = EmotionMonitor()

    assert monitor.negative_share() == 0.0


def test_record_evicts_samples_older_than_window() -> None:
    """A sample past WINDOW_SECONDS should no longer count toward negative_share()."""
    monitor = EmotionMonitor()
    monitor.record("sad", timestamp=0.0)

    monitor.record("happy", timestamp=EmotionMonitor.WINDOW_SECONDS + 1.0)

    assert monitor.negative_share() == 0.0


def test_negative_share_computes_fraction_of_negative_samples() -> None:
    """2 of 5 samples negative should report exactly 0.4."""
    monitor = EmotionMonitor()
    for emotion in ("angry", "sad", "happy", "neutral", "surprise"):
        monitor.record(emotion, timestamp=0.0)

    assert monitor.negative_share() == 0.4


def test_should_intervene_true_when_all_gates_pass() -> None:
    """Negative share above threshold, idle response, and elapsed cooldowns should trigger."""
    monitor = EmotionMonitor()
    for emotion in ("angry", "sad", "sad", "happy", "neutral"):
        monitor.record(emotion, timestamp=0.0)

    result = monitor.should_intervene(now=100.0, response_done=True, last_activity_time=0.0)

    assert result is True


def test_should_intervene_false_when_share_exactly_at_threshold() -> None:
    """Negative share exactly at NEGATIVE_THRESHOLD should not trigger — it must be exceeded, not met."""
    monitor = EmotionMonitor()
    for emotion in ("angry", "sad", "happy", "neutral", "surprise"):
        monitor.record(emotion, timestamp=0.0)

    result = monitor.should_intervene(now=100.0, response_done=True, last_activity_time=0.0)

    assert result is False


def test_should_intervene_false_when_response_not_done() -> None:
    """An active model response should block intervention even if emotion/timing gates pass."""
    monitor = EmotionMonitor()
    for emotion in ("angry", "sad", "sad", "happy", "neutral"):
        monitor.record(emotion, timestamp=0.0)

    result = monitor.should_intervene(now=100.0, response_done=False, last_activity_time=0.0)

    assert result is False


def test_should_intervene_false_when_interaction_cooldown_not_elapsed() -> None:
    """A recent user interaction should block intervention even if emotion/response gates pass."""
    monitor = EmotionMonitor()
    for emotion in ("angry", "sad", "sad", "happy", "neutral"):
        monitor.record(emotion, timestamp=0.0)

    result = monitor.should_intervene(now=100.0, response_done=True, last_activity_time=95.0)

    assert result is False


def test_should_intervene_false_when_intervention_cooldown_not_elapsed() -> None:
    """A recent prior intervention should block a new one even if all other gates pass."""
    monitor = EmotionMonitor()
    for emotion in ("angry", "sad", "sad", "happy", "neutral"):
        monitor.record(emotion, timestamp=0.0)
    monitor.mark_intervened(now=100.0)

    result = monitor.should_intervene(now=110.0, response_done=True, last_activity_time=0.0)

    assert result is False
