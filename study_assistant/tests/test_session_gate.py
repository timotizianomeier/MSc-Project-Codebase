"""Study-session gate (SESSION_GATE_ENABLED / Start-session button) tests.

Contract: gate off = stock behavior everywhere; gate on = armed-but-dormant
(no greeting, no monitor polls, no context input) until start_study_session(),
then a timer ends the session, closes the gate again, and announces the end
(announcement suppressed in the control condition).
"""

from __future__ import annotations
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.huggingface_realtime import HuggingFaceRealtimeHandler


def _make_handler() -> HuggingFaceRealtimeHandler:
    return HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))


def test_gate_open_when_feature_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gate off -> always open: dev runs behave exactly as before the feature."""
    monkeypatch.setattr(config, "SESSION_GATE_ENABLED", False)
    handler = _make_handler()

    assert handler.session_gate_open() is True


def test_gate_closed_before_start_and_open_after(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gate on -> closed at launch, open after start, closed again after end."""
    monkeypatch.setattr(config, "SESSION_GATE_ENABLED", True)
    handler = _make_handler()

    assert handler.session_gate_open() is False
    handler._study_session_started_at = 100.0
    assert handler.session_gate_open() is True
    handler._study_session_ended = True
    assert handler.session_gate_open() is False


@pytest.mark.asyncio
async def test_polls_are_noops_before_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """Armed-but-dormant: neither monitor touches the camera before the session starts."""
    monkeypatch.setattr(config, "SESSION_GATE_ENABLED", True)
    handler = _make_handler()

    await handler._poll_emotion_once()
    await handler._poll_engagement_once(score_now=True)

    handler.deps.reachy_mini.media.get_frame.assert_not_called()
    handler.deps.reachy_mini.media.get_frame_jpeg.assert_not_called()


@pytest.mark.asyncio
async def test_greeting_deferred_not_consumed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pre-start greeting attempt must neither speak nor burn the one-shot flag."""
    monkeypatch.setattr(config, "SESSION_GATE_ENABLED", True)
    monkeypatch.setattr(config, "CONTROL_MODE", False)
    handler = _make_handler()
    create = AsyncMock()
    connection = MagicMock()
    connection.conversation.item.create = create
    handler.connection = connection

    await handler._send_startup_greeting_prompt()

    create.assert_not_awaited()
    assert handler._startup_greeting_sent is False


@pytest.mark.asyncio
async def test_start_fires_greeting_and_timer_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """First start opens the gate, greets, and schedules the timer; repeats are ignored."""
    monkeypatch.setattr(config, "SESSION_GATE_ENABLED", True)
    handler = _make_handler()
    greeting = AsyncMock()
    monkeypatch.setattr(handler, "_send_startup_greeting_prompt", greeting)

    first = await handler.start_study_session()
    second = await handler.start_study_session()
    handler._session_timer_task.cancel()

    assert first is True
    assert second is False
    greeting.assert_awaited_once()
    assert handler.session_gate_open() is True


@pytest.mark.asyncio
async def test_timer_ends_session_and_announces(monkeypatch: pytest.MonkeyPatch) -> None:
    """Timer expiry closes the gate and queues the session-over announcement."""
    monkeypatch.setattr(config, "SESSION_GATE_ENABLED", True)
    monkeypatch.setattr(config, "CONTROL_MODE", False)
    monkeypatch.setattr(config, "SESSION_DURATION_MINUTES", 0.0001)
    handler = _make_handler()
    say = AsyncMock()
    monkeypatch.setattr(handler, "say", say)
    monkeypatch.setattr(handler, "_send_startup_greeting_prompt", AsyncMock())

    await handler.start_study_session()
    await asyncio.wait_for(handler._session_timer_task, timeout=2.0)

    assert handler._study_session_ended is True
    assert handler.session_gate_open() is False
    say.assert_awaited_once()


@pytest.mark.asyncio
async def test_timer_announcement_suppressed_in_control(monkeypatch: pytest.MonkeyPatch) -> None:
    """Control condition: session ends and is logged, but the robot stays silent."""
    monkeypatch.setattr(config, "SESSION_GATE_ENABLED", True)
    monkeypatch.setattr(config, "CONTROL_MODE", True)
    monkeypatch.setattr(config, "SESSION_DURATION_MINUTES", 0.0001)
    handler = _make_handler()
    say = AsyncMock()
    monkeypatch.setattr(handler, "say", say)
    monkeypatch.setattr(handler, "_send_startup_greeting_prompt", AsyncMock())

    await handler.start_study_session()
    await asyncio.wait_for(handler._session_timer_task, timeout=2.0)

    assert handler._study_session_ended is True
    say.assert_not_awaited()


@pytest.mark.asyncio
async def test_context_rejected_before_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """Typed context never reaches the model while the gate is closed."""
    monkeypatch.setattr(config, "SESSION_GATE_ENABLED", True)
    handler = _make_handler()
    say = AsyncMock()
    monkeypatch.setattr(handler, "say", say)

    await handler.send_user_text("problem sheet 3")

    say.assert_not_awaited()


@pytest.mark.asyncio
async def test_start_is_noop_when_gate_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stray session.start call with the gate off must not create a timer."""
    monkeypatch.setattr(config, "SESSION_GATE_ENABLED", False)
    handler = _make_handler()

    started = await handler.start_study_session()

    assert started is False
    assert handler._session_timer_task is None
