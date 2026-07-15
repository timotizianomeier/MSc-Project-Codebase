"""HTTP client for the external engagement-scoring service."""

import base64
import logging

import httpx

from reachy_mini_conversation_app.config import config


logger = logging.getLogger(__name__)

FRAMES_PER_SCORE = 10
_REQUEST_TIMEOUT_S = 10.0  # ResNeXt feature extraction takes ~1s+ per window on CPU


def fetch_engagement_score(http_client: httpx.Client, frames: list[bytes]) -> float | None:
    """Score a 10-frame JPEG window via the engagement service, returning None on any failure."""
    if len(frames) != FRAMES_PER_SCORE:
        logger.warning("Engagement scoring needs exactly %d frames, got %d", FRAMES_PER_SCORE, len(frames))
        return None

    payload = {"frames": [base64.b64encode(frame).decode("ascii") for frame in frames]}
    try:
        response = http_client.post(f"{config.ENGAGEMENT_SERVICE_URL}/score", json=payload, timeout=_REQUEST_TIMEOUT_S)
        response.raise_for_status()
        score = response.json().get("score")
    except httpx.HTTPError as e:
        logger.warning("Engagement service request failed: %s", e)
        return None
    except ValueError as e:
        logger.warning("Engagement service returned invalid JSON: %s", e)
        return None

    if not isinstance(score, (int, float)):
        logger.warning("Engagement service returned invalid score: %r", score)
        return None
    return float(score)
