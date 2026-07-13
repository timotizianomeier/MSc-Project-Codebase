import json
from typing import Any

import httpx
import numpy as np
import pytest

from reachy_mini_conversation_app.engagement_client import FRAMES_PER_SCORE, fetch_engagement_score


def _dummy_frames(count: int = FRAMES_PER_SCORE) -> list[Any]:
    return [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(count)]


def test_returns_score_and_sends_contract_shaped_request() -> None:
    """A 200 response yields the score; the request must match the service contract."""
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"score": 0.87})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    score = fetch_engagement_score(client, _dummy_frames())

    assert score == pytest.approx(0.87)
    assert seen["path"] == "/score"
    sent_frames = seen["body"]["frames"]
    assert len(sent_frames) == FRAMES_PER_SCORE
    assert all(isinstance(frame, str) for frame in sent_frames)


def test_wrong_frame_count_returns_none_without_request() -> None:
    """A short window is rejected client-side; no request reaches the service."""

    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("no request should be sent for a wrong-sized window")

    client = httpx.Client(transport=httpx.MockTransport(handler))

    assert fetch_engagement_score(client, _dummy_frames(count=7)) is None


def test_service_down_returns_none() -> None:
    """A connection failure degrades to None instead of raising."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = httpx.Client(transport=httpx.MockTransport(handler))

    assert fetch_engagement_score(client, _dummy_frames()) is None


def test_error_status_returns_none() -> None:
    """A non-2xx response degrades to None instead of raising."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "model not loaded"})

    client = httpx.Client(transport=httpx.MockTransport(handler))

    assert fetch_engagement_score(client, _dummy_frames()) is None


def test_invalid_score_returns_none() -> None:
    """A well-formed 200 with a non-numeric score must not be trusted."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"score": "high"})

    client = httpx.Client(transport=httpx.MockTransport(handler))

    assert fetch_engagement_score(client, _dummy_frames()) is None
