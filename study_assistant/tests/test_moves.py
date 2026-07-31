import time
import threading
from unittest.mock import MagicMock, call
from collections.abc import Callable

import numpy as np
import pytest

from reachy_mini.utils import create_head_pose
from reachy_mini.utils.interpolation import compose_world_offset
from reachy_mini_conversation_app.moves import MovementManager
from reachy_mini_conversation_app.dance_emotion_moves import EmotionQueueMove


class _FakeMove:
    """Minimal non-emotion Move stub returning a fixed head pose."""

    def __init__(self, head: np.ndarray) -> None:
        self._head = head
        self.duration = 10.0

    def evaluate(self, t: float):
        return (self._head, np.array([0.0, 0.0]), 0.0)


def _wait_for(predicate: Callable[[], bool], timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_stop_can_skip_neutral_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sleep shutdown should stop the movement loop without undoing the sleep pose."""
    robot = MagicMock()
    manager = MovementManager(robot)
    started = threading.Event()

    def fake_working_loop() -> None:
        started.set()
        while not manager._stop_event.is_set():
            time.sleep(0.001)

    monkeypatch.setattr(manager, "working_loop", fake_working_loop)

    manager.start()
    assert started.wait(timeout=1.0)

    manager.stop(reset_to_neutral=False)

    assert manager._thread is None
    robot.goto_target.assert_not_called()


def test_head_tracking_follows_speaking() -> None:
    """Once enabled, tracking owns the head when idle and releases it while the assistant speaks."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = np.eye(4)
    robot.get_current_joint_positions.return_value = ([0.0] * 6, [0.0, 0.0])
    manager = MovementManager(robot)
    manager.start()
    try:
        # The head_tracking tool enables tracking with full weight.
        manager.set_head_tracking(True)
        assert _wait_for(lambda: call(weight=1.0) in robot.start_head_tracking.call_args_list)

        # Speaking with a locked face captures the anchor and releases the head.
        manager.set_speaking(True)
        assert _wait_for(lambda: call(weight=0.0) in robot.start_head_tracking.call_args_list)
        assert _wait_for(lambda: manager._track_anchor is not None)

        # Done speaking hands the head back to tracking.
        robot.start_head_tracking.reset_mock()
        manager.set_speaking(False)
        assert _wait_for(lambda: call(weight=1.0) in robot.start_head_tracking.call_args_list)
        assert _wait_for(lambda: manager._track_anchor is None)
    finally:
        manager.stop(reset_to_neutral=False)

    robot.stop_head_tracking.assert_called_once()


def test_speaking_anchor_composes_emotions_and_holds_dances_from_neutral() -> None:
    """While speaking: hold the anchor, compose emotions onto it, play dances from neutral."""
    robot = MagicMock()
    manager = MovementManager(robot)
    anchor = create_head_pose(0, 0, 0, 0, 0, 20, degrees=True)
    manager._track_anchor = anchor

    # No move: the head holds the captured look-at anchor.
    manager.state.current_move = None
    head, _, _ = manager._get_primary_pose(manager._now())
    assert np.allclose(head, anchor)

    # Emotion: composed onto the anchor exactly like the daemon wobble.
    emotion_head = create_head_pose(0, 0, 0, 0, 0, 15, degrees=True)
    recorded = MagicMock()
    recorded.get.return_value = _FakeMove(emotion_head)
    manager.state.current_move = EmotionQueueMove("happy", recorded)
    manager.state.move_start_time = manager._now()
    head, _, _ = manager._get_primary_pose(manager._now())
    assert np.allclose(head, compose_world_offset(anchor, emotion_head))

    # Any other move (e.g. a dance) plays from its own neutral base, ignoring the anchor.
    dance_head = create_head_pose(0, 0, 0, 0, 25, 0, degrees=True)
    manager.state.current_move = _FakeMove(dance_head)
    manager.state.move_start_time = manager._now()
    head, _, _ = manager._get_primary_pose(manager._now())
    assert np.allclose(head, dance_head)


def test_pitch_trim_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """With HEAD_PITCH_TRIM_DEG at 0, poses reach the robot untouched."""
    from reachy_mini_conversation_app.config import config

    monkeypatch.setattr(config, "HEAD_PITCH_TRIM_DEG", 0.0)
    robot = MagicMock()
    manager = MovementManager(robot)
    head = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)

    manager._issue_control_command(head, (0.0, 0.0), 0.0)

    sent = robot.set_target.call_args.kwargs["head"]
    assert np.allclose(sent, head)


def test_pitch_trim_tilts_every_outgoing_pose(monkeypatch: pytest.MonkeyPatch) -> None:
    """A nonzero trim composes the pitch offset under whatever pose is being sent."""
    from reachy_mini_conversation_app.config import config

    monkeypatch.setattr(config, "HEAD_PITCH_TRIM_DEG", 8.0)
    robot = MagicMock()
    manager = MovementManager(robot)
    head = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)

    manager._issue_control_command(head, (0.0, 0.0), 0.0)

    sent = robot.set_target.call_args.kwargs["head"]
    expected = compose_world_offset(create_head_pose(0, 0, 0, 0, 8.0, 0, degrees=True), head)
    assert np.allclose(sent, expected)
    assert not np.allclose(sent, head)


def _heartbeat_manager(
    monkeypatch: pytest.MonkeyPatch, *, enabled: bool = True, control: bool = False
) -> MovementManager:
    from reachy_mini_conversation_app.config import config

    monkeypatch.setattr(config, "ANTENNA_HEARTBEAT_ENABLED", enabled)
    monkeypatch.setattr(config, "CONTROL_MODE", control)
    return MovementManager(MagicMock())


def test_heartbeat_move_starts_and_ends_at_held_antennas() -> None:
    """The flutter must begin and end exactly at the held positions — no snap."""
    from reachy_mini_conversation_app.moves import AntennaHeartbeatMove

    hold = (create_head_pose(0, 0, 0, 0, 0, 0, degrees=True), (-0.1745, 0.1745), 0.0)
    move = AntennaHeartbeatMove(hold)

    _, start_antennas, _ = move.evaluate(0.0)
    _, end_antennas, _ = move.evaluate(move.duration)
    _, mid_antennas, _ = move.evaluate(move.duration * 0.15)

    assert np.allclose(start_antennas, hold[1], atol=1e-9)
    assert np.allclose(end_antennas, hold[1], atol=1e-9)
    assert not np.allclose(mid_antennas, hold[1], atol=1e-3)


def test_heartbeat_move_never_moves_head_or_body() -> None:
    """Only antennas animate; head pose and body yaw are held for the whole move."""
    from reachy_mini_conversation_app.moves import AntennaHeartbeatMove

    hold_head = create_head_pose(0, 0, 0, 0, -8, 0, degrees=True)
    move = AntennaHeartbeatMove((hold_head, (-0.1745, 0.1745), 0.3))

    for t in (0.0, 0.3, 0.6, 0.9, 1.2):
        head, _, body_yaw = move.evaluate(t)
        assert np.allclose(head, hold_head)
        assert body_yaw == 0.3


def test_heartbeat_queues_when_due_and_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enabled + due + fully idle -> exactly one flutter lands in the move queue."""
    from reachy_mini_conversation_app.moves import AntennaHeartbeatMove

    manager = _heartbeat_manager(monkeypatch)
    manager._next_heartbeat_time = 0.0  # long overdue

    manager._maybe_queue_antenna_heartbeat(now=100.0)

    assert len(manager.move_queue) == 1
    assert isinstance(manager.move_queue[0], AntennaHeartbeatMove)
    assert manager._next_heartbeat_time > 100.0  # rescheduled


def test_heartbeat_first_call_only_schedules(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first tick schedules a future flutter instead of firing immediately at startup."""
    manager = _heartbeat_manager(monkeypatch)

    manager._maybe_queue_antenna_heartbeat(now=100.0)

    assert len(manager.move_queue) == 0
    lo, hi = MovementManager.HEARTBEAT_INTERVAL_RANGE_S
    assert 100.0 + lo <= manager._next_heartbeat_time <= 100.0 + hi


def test_heartbeat_suppressed_in_control_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Control condition: no flutter, ever — not even scheduling."""
    manager = _heartbeat_manager(monkeypatch, control=True)
    manager._next_heartbeat_time = 0.0

    manager._maybe_queue_antenna_heartbeat(now=100.0)

    assert len(manager.move_queue) == 0


def test_heartbeat_postponed_while_busy(monkeypatch: pytest.MonkeyPatch) -> None:
    """A due flutter defers (short retry) while listening instead of interrupting."""
    manager = _heartbeat_manager(monkeypatch)
    manager._next_heartbeat_time = 0.0
    manager._is_listening = True

    manager._maybe_queue_antenna_heartbeat(now=100.0)

    assert len(manager.move_queue) == 0
    assert manager._next_heartbeat_time == 105.0  # 5s retry, slot not burned
