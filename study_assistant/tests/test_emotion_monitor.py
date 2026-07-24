from __future__ import annotations

from reachy_mini_conversation_app.emotion_monitor import EmotionMonitor, negative_mass


def test_negative_mass_sums_negative_classes_only() -> None:
    """Only the four negative classes contribute; unknown labels (e.g. contempt) do not."""
    scores = {"sad": 0.3, "fear": 0.2, "neutral": 0.4, "contempt": 0.1}

    assert negative_mass(scores) == 0.5


def test_negative_share_with_no_samples_is_zero() -> None:
    """An empty window should read as no negative signal, not divide by zero."""
    monitor = EmotionMonitor()

    assert monitor.negative_share() == 0.0


def test_record_evicts_samples_older_than_window() -> None:
    """A sample past WINDOW_SECONDS should no longer count toward negative_share()."""
    monitor = EmotionMonitor()
    monitor.record(1.0, timestamp=0.0)

    monitor.record(0.0, timestamp=EmotionMonitor.WINDOW_SECONDS + 1.0)

    assert monitor.negative_share() == 0.0


def test_negative_share_is_mean_of_masses() -> None:
    """Ambiguous frames contribute their actual mass, not a full vote: mean(0.5, 0.3, 0.4) = 0.4."""
    monitor = EmotionMonitor()
    for mass in (0.5, 0.3, 0.4):
        monitor.record(mass, timestamp=0.0)

    assert abs(monitor.negative_share() - 0.4) < 1e-9


def test_should_intervene_true_when_all_gates_pass() -> None:
    """Negative share above threshold, idle response, and elapsed cooldowns should trigger."""
    monitor = EmotionMonitor()
    for mass in (1.0, 1.0, 1.0, 0.0, 0.0):
        monitor.record(mass, timestamp=0.0)

    result = monitor.should_intervene(now=100.0, response_done=True, last_activity_time=0.0)

    assert result is True


def test_should_intervene_false_when_share_exactly_at_threshold() -> None:
    """Negative share exactly at the threshold should not trigger — it must be exceeded, not met."""
    monitor = EmotionMonitor()
    for mass in (1.0, 1.0, 0.0, 0.0, 0.0):
        monitor.record(mass, timestamp=0.0)

    result = monitor.should_intervene(now=100.0, response_done=True, last_activity_time=0.0)

    assert result is False


def test_constructor_threshold_overrides_default() -> None:
    """A raised threshold (study calibration knob) should silence a share the default would fire on."""
    monitor = EmotionMonitor(negative_threshold=0.55)
    for mass in (1.0, 1.0, 0.0, 0.0):
        monitor.record(mass, timestamp=0.0)

    result = monitor.should_intervene(now=100.0, response_done=True, last_activity_time=0.0)

    assert result is False


def test_should_intervene_false_when_response_not_done() -> None:
    """An active model response should block intervention even if emotion/timing gates pass."""
    monitor = EmotionMonitor()
    for mass in (1.0, 1.0, 1.0, 0.0, 0.0):
        monitor.record(mass, timestamp=0.0)

    result = monitor.should_intervene(now=100.0, response_done=False, last_activity_time=0.0)

    assert result is False


def test_should_intervene_false_when_interaction_cooldown_not_elapsed() -> None:
    """A recent user interaction should block intervention even if emotion/response gates pass."""
    monitor = EmotionMonitor()
    for mass in (1.0, 1.0, 1.0, 0.0, 0.0):
        monitor.record(mass, timestamp=0.0)

    result = monitor.should_intervene(now=100.0, response_done=True, last_activity_time=95.0)

    assert result is False


def test_should_intervene_false_when_intervention_cooldown_not_elapsed() -> None:
    """A recent prior intervention should block a new one even if all other gates pass."""
    monitor = EmotionMonitor()
    for mass in (1.0, 1.0, 1.0, 0.0, 0.0):
        monitor.record(mass, timestamp=0.0)
    monitor.mark_intervened(now=100.0)

    result = monitor.should_intervene(now=110.0, response_done=True, last_activity_time=0.0)

    assert result is False
