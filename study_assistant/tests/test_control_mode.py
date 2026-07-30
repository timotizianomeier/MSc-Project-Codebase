"""Control-condition (--control / CONTROL_MODE) gating tests.

Sensing must run identically to the treatment condition; only the four robot
output channels are gated: mic forwarding, startup greeting, intervention sends
(replaced by counterfactual logs that still consume the cooldown), and the
context-page reply.
"""

from __future__ import annotations
import sys
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

import reachy_mini_conversation_app.huggingface_realtime as hf_mod
from reachy_mini_conversation_app.utils import parse_args
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.engagement_client import FRAMES_PER_SCORE
from reachy_mini_conversation_app.huggingface_realtime import HuggingFaceRealtimeHandler


def _make_handler() -> HuggingFaceRealtimeHandler:
    return HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))


def test_control_flag_parses_and_defaults_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """--control is a store-true flag; absent means treatment condition."""
    monkeypatch.setattr(sys, "argv", ["prog"])
    args, _ = parse_args()
    assert args.control is False

    monkeypatch.setattr(sys, "argv", ["prog", "--control"])
    args, _ = parse_args()
    assert args.control is True


def test_control_mode_config_defaults_off() -> None:
    """CONTROL_MODE must be off unless explicitly requested — treatment is the default."""
    assert config.CONTROL_MODE is False


@pytest.mark.asyncio
async def test_emotion_intervention_suppressed_but_cooldown_consumed(monkeypatch: pytest.MonkeyPatch) -> None:
    """In control mode the emotion gate fires the counterfactual path: no send, cooldown marked."""
    handler = _make_handler()
    handler.deps.reachy_mini.media.get_frame.return_value = np.zeros((4, 4, 3), dtype=np.uint8)
    monkeypatch.setattr(hf_mod, "classify_dominant_emotion", lambda frame: ("sad", {"sad": 0.9, "neutral": 0.1}))
    monkeypatch.setattr(handler, "_is_connected", lambda: True)
    monkeypatch.setattr(handler._emotion_monitor, "should_intervene", MagicMock(return_value=True))
    mark = MagicMock()
    monkeypatch.setattr(handler._emotion_monitor, "mark_intervened", mark)
    send = AsyncMock()
    monkeypatch.setattr(handler, "_send_emotion_intervention", send)

    monkeypatch.setattr(config, "CONTROL_MODE", True)
    await handler._poll_emotion_once()

    send.assert_not_awaited()
    mark.assert_called_once()


@pytest.mark.asyncio
async def test_emotion_intervention_sent_in_treatment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same setup without control mode must actually send — the gate has the right polarity."""
    handler = _make_handler()
    handler.deps.reachy_mini.media.get_frame.return_value = np.zeros((4, 4, 3), dtype=np.uint8)
    monkeypatch.setattr(hf_mod, "classify_dominant_emotion", lambda frame: ("sad", {"sad": 0.9, "neutral": 0.1}))
    monkeypatch.setattr(handler, "_is_connected", lambda: True)
    monkeypatch.setattr(handler._emotion_monitor, "should_intervene", MagicMock(return_value=True))
    monkeypatch.setattr(handler._emotion_monitor, "mark_intervened", MagicMock())
    send = AsyncMock()
    monkeypatch.setattr(handler, "_send_emotion_intervention", send)

    monkeypatch.setattr(config, "CONTROL_MODE", False)
    await handler._poll_emotion_once()

    send.assert_awaited_once()


@pytest.mark.asyncio
async def test_engagement_intervention_suppressed_but_cooldown_consumed(monkeypatch: pytest.MonkeyPatch) -> None:
    """In control mode the engagement gate logs the counterfactual: no send, cooldown marked."""
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

    monkeypatch.setattr(config, "CONTROL_MODE", True)
    await handler._poll_engagement_once(score_now=True)

    send.assert_not_awaited()
    mark.assert_called_once()


@pytest.mark.asyncio
async def test_startup_greeting_suppressed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Control mode marks the greeting as sent without creating a conversation item."""
    handler = _make_handler()
    create = AsyncMock()
    connection = MagicMock()
    connection.conversation.item.create = create
    handler.connection = connection

    monkeypatch.setattr(config, "CONTROL_MODE", True)
    await handler._send_startup_greeting_prompt()

    create.assert_not_awaited()
    assert handler._startup_greeting_sent is True


@pytest.mark.asyncio
async def test_send_user_text_not_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Context-page submissions are accepted and logged but never reach the model."""
    handler = _make_handler()
    say = AsyncMock()
    monkeypatch.setattr(handler, "say", say)

    monkeypatch.setattr(config, "CONTROL_MODE", True)
    await handler.send_user_text("problem sheet 3, question 2")
    say.assert_not_awaited()

    monkeypatch.setattr(config, "CONTROL_MODE", False)
    await handler.send_user_text("problem sheet 3, question 2")
    say.assert_awaited_once()


@pytest.mark.asyncio
async def test_sensing_still_records_in_control_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """The monitors must keep accumulating data in control mode — only the send is gated."""
    handler = _make_handler()
    handler.deps.reachy_mini.media.get_frame.return_value = np.zeros((4, 4, 3), dtype=np.uint8)
    monkeypatch.setattr(hf_mod, "classify_dominant_emotion", lambda frame: ("sad", {"sad": 0.9, "neutral": 0.1}))
    monkeypatch.setattr(handler, "_is_connected", lambda: True)

    monkeypatch.setattr(config, "CONTROL_MODE", True)
    await handler._poll_emotion_once()

    assert handler._emotion_monitor.negative_share() > 0.0
