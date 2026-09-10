"""Live-demo mode (--demo / DEMO_MODE) tests.

Sensing and the intervention gates must run exactly as in the robot condition;
only the automatic send is replaced by a "DEMO: would have ..." log line that
still consumes the cooldowns. Control mode takes precedence over demo mode.
"""

from __future__ import annotations
import sys
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

import reachy_mini_conversation_app.huggingface_realtime as hf_mod
from reachy_mini_conversation_app.utils import parse_args
from reachy_mini_conversation_app.config import config, refresh_runtime_config_from_env
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.engagement_client import FRAMES_PER_SCORE
from reachy_mini_conversation_app.huggingface_realtime import HuggingFaceRealtimeHandler


def _make_handler() -> HuggingFaceRealtimeHandler:
    return HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))


def _arm_emotion_gate(handler: HuggingFaceRealtimeHandler, monkeypatch: pytest.MonkeyPatch) -> tuple[AsyncMock, MagicMock]:
    handler.deps.reachy_mini.media.get_frame.return_value = np.zeros((4, 4, 3), dtype=np.uint8)
    monkeypatch.setattr(hf_mod, "classify_dominant_emotion", lambda frame: ("sad", {"sad": 0.9, "neutral": 0.1}))
    monkeypatch.setattr(handler, "_is_connected", lambda: True)
    monkeypatch.setattr(handler._emotion_monitor, "should_intervene", MagicMock(return_value=True))
    mark = MagicMock()
    monkeypatch.setattr(handler._emotion_monitor, "mark_intervened", mark)
    send = AsyncMock()
    monkeypatch.setattr(handler, "_send_emotion_intervention", send)
    return send, mark


def test_demo_flag_parses_and_defaults_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """--demo is a store-true flag; absent means the real study behaviour."""
    monkeypatch.setattr(sys, "argv", ["prog"])
    args, _ = parse_args()
    assert args.demo is False

    monkeypatch.setattr(sys, "argv", ["prog", "--demo"])
    args, _ = parse_args()
    assert args.demo is True


def test_demo_mode_config_defaults_off() -> None:
    """DEMO_MODE must be off unless explicitly requested."""
    assert config.DEMO_MODE is False


def test_demo_mode_not_refreshed_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A UI settings save re-reads the env; the demo condition must not flip with it."""
    monkeypatch.setenv("DEMO_MODE", "1")
    monkeypatch.setattr(config, "DEMO_MODE", False)
    refresh_runtime_config_from_env()
    assert config.DEMO_MODE is False


@pytest.mark.asyncio
async def test_emotion_intervention_suppressed_but_cooldown_consumed(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """In demo mode the emotion gate logs a DEMO counterfactual: no send, cooldown marked."""
    handler = _make_handler()
    send, mark = _arm_emotion_gate(handler, monkeypatch)

    monkeypatch.setattr(config, "CONTROL_MODE", False)
    monkeypatch.setattr(config, "DEMO_MODE", True)
    with caplog.at_level("INFO"):
        await handler._poll_emotion_once()

    send.assert_not_awaited()
    mark.assert_called_once()
    assert any(r.getMessage().startswith("DEMO: would have sent emotion intervention") for r in caplog.records)
    assert not any(r.getMessage().startswith("CONTROL:") for r in caplog.records)


@pytest.mark.asyncio
async def test_engagement_intervention_suppressed_but_cooldown_consumed(monkeypatch: pytest.MonkeyPatch) -> None:
    """In demo mode the engagement gate logs a DEMO counterfactual: no send, cooldown marked."""
    handler = _make_handler()
    handler.deps.reachy_mini.media.get_frame_jpeg.return_value = b"jpeg"
    handler._engagement_http = MagicMock()
    for _ in range(FRAMES_PER_SCORE):
        handler._engagement_frames.append(b"jpeg")
    monkeypatch.setattr(hf_mod, "fetch_engagement_score", lambda http, frames: 0.10)
    monkeypatch.setattr(handler, "_is_connected", lambda: True)
    monkeypatch.setattr(handler._engagement_monitor, "should_intervene", MagicMock(return_value=True))
    mark = MagicMock()
    monkeypatch.setattr(handler._engagement_monitor, "mark_intervened", mark)
    send = AsyncMock()
    monkeypatch.setattr(handler, "_send_engagement_intervention", send)

    monkeypatch.setattr(config, "CONTROL_MODE", False)
    monkeypatch.setattr(config, "DEMO_MODE", True)
    await handler._poll_engagement_once(score_now=True)

    send.assert_not_awaited()
    mark.assert_called_once()


@pytest.mark.asyncio
async def test_control_mode_takes_precedence_over_demo(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """With both flags on, the control branch wins and the log says CONTROL, not DEMO."""
    handler = _make_handler()
    send, _ = _arm_emotion_gate(handler, monkeypatch)

    monkeypatch.setattr(config, "CONTROL_MODE", True)
    monkeypatch.setattr(config, "DEMO_MODE", True)
    with caplog.at_level("INFO"):
        await handler._poll_emotion_once()

    send.assert_not_awaited()
    assert any(r.getMessage().startswith("CONTROL: would have sent") for r in caplog.records)
    assert not any(r.getMessage().startswith("DEMO:") for r in caplog.records)


@pytest.mark.asyncio
async def test_sensing_still_records_in_demo_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """The monitors keep accumulating data in demo mode — only the send is gated."""
    handler = _make_handler()
    handler.deps.reachy_mini.media.get_frame.return_value = np.zeros((4, 4, 3), dtype=np.uint8)
    monkeypatch.setattr(hf_mod, "classify_dominant_emotion", lambda frame: ("sad", {"sad": 0.9, "neutral": 0.1}))
    monkeypatch.setattr(handler, "_is_connected", lambda: True)

    monkeypatch.setattr(config, "DEMO_MODE", True)
    await handler._poll_emotion_once()

    assert handler._emotion_monitor.negative_share() > 0.0


# ---------------------------------------------------------------- telemetry


def test_gate_conditions_agree_with_should_intervene() -> None:
    """The per-check breakdown is the same five predicates should_intervene() ANDs together."""
    from reachy_mini_conversation_app.emotion_monitor import EmotionMonitor

    monitor = EmotionMonitor(negative_threshold=0.5)
    now = 1000.0
    for i in range(3):
        monitor.record(0.9, now - 10 + i)
    for response_done, last_activity in ((True, now - 120), (False, now - 120), (True, now - 5)):
        gate = monitor.gate_conditions(now, response_done, last_activity)
        assert set(gate) == {
            "enough_samples", "signal_active", "robot_silent", "interaction_cooldown_ok", "intervention_cooldown_ok"
        }
        assert all(gate.values()) == monitor.should_intervene(now, response_done, last_activity)
    assert monitor.sample_count == 3


def _collect(handler: HuggingFaceRealtimeHandler) -> list[dict]:
    seen: list[dict] = []
    handler.set_sensing_observer(seen.append)
    return seen


@pytest.mark.asyncio
async def test_sensing_observer_receives_emotion_payload_with_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """Demo mode emits one emotion snapshot per poll, carrying a real downscaled JPEG and the gate breakdown."""
    handler = _make_handler()
    seen = _collect(handler)
    handler.deps.reachy_mini.media.get_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    monkeypatch.setattr(hf_mod, "classify_dominant_emotion", lambda frame: ("sad", {"sad": 0.9, "neutral": 0.1}))
    monkeypatch.setattr(handler, "_is_connected", lambda: False)
    monkeypatch.setattr(config, "DEMO_MODE", True)

    await handler._poll_emotion_once()

    assert len(seen) == 1
    p = seen[0]
    assert p["kind"] == "emotion" and p["emotion"] == "sad"
    assert p["scores"] == {"sad": 0.9, "neutral": 0.1}
    assert p["samples"] == 1 and p["would_fire"] is False and p["gate"]["enough_samples"] is False
    import base64

    assert base64.b64decode(p["jpeg_b64"]).startswith(b"\xff\xd8")  # JPEG SOI marker


@pytest.mark.asyncio
async def test_sensing_observer_fires_on_no_face(monkeypatch: pytest.MonkeyPatch) -> None:
    """A no-face frame still reaches the page (emotion=None) — the moment worth showing."""
    handler = _make_handler()
    seen = _collect(handler)
    handler.deps.reachy_mini.media.get_frame.return_value = np.zeros((4, 4, 3), dtype=np.uint8)
    monkeypatch.setattr(hf_mod, "classify_dominant_emotion", lambda frame: None)
    monkeypatch.setattr(config, "DEMO_MODE", True)

    await handler._poll_emotion_once()

    assert len(seen) == 1
    assert seen[0]["emotion"] is None and seen[0]["scores"] is None and seen[0]["jpeg_b64"]
    assert handler._emotion_monitor.sample_count == 0  # no-face frames are still not recorded


@pytest.mark.asyncio
async def test_no_sensing_emit_when_demo_mode_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Study runs pay nothing: the observer is never called outside demo mode."""
    handler = _make_handler()
    seen = _collect(handler)
    handler.deps.reachy_mini.media.get_frame.return_value = np.zeros((4, 4, 3), dtype=np.uint8)
    monkeypatch.setattr(hf_mod, "classify_dominant_emotion", lambda frame: ("sad", {"sad": 0.9, "neutral": 0.1}))
    monkeypatch.setattr(handler, "_is_connected", lambda: False)
    monkeypatch.setattr(config, "DEMO_MODE", False)

    await handler._poll_emotion_once()

    assert seen == []


@pytest.mark.asyncio
async def test_sensing_observer_receives_engagement_payload_and_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Engagement score ticks emit a snapshot; a dead /score service emits an explicit error."""
    handler = _make_handler()
    seen = _collect(handler)
    handler.deps.reachy_mini.media.get_frame_jpeg.return_value = b"jpeg"
    handler._engagement_http = MagicMock()
    for _ in range(FRAMES_PER_SCORE):
        handler._engagement_frames.append(b"jpeg")
    monkeypatch.setattr(handler, "_is_connected", lambda: False)
    monkeypatch.setattr(config, "DEMO_MODE", True)

    monkeypatch.setattr(hf_mod, "fetch_engagement_score", lambda http, frames: 0.55)
    await handler._poll_engagement_once(score_now=True)
    monkeypatch.setattr(hf_mod, "fetch_engagement_score", lambda http, frames: None)
    await handler._poll_engagement_once(score_now=True)

    assert [p["kind"] for p in seen] == ["engagement", "engagement"]
    assert seen[0]["score"] == 0.55 and seen[0]["error"] is None and "gate" in seen[0]
    assert seen[1]["score"] is None and seen[1]["error"]
