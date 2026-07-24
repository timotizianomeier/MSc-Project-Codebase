import sys
import types
from typing import Any

import numpy as np
import pytest

from reachy_mini_conversation_app.emotion_classifier import classify_dominant_emotion


def _install_fake_deepface(
    monkeypatch: pytest.MonkeyPatch,
    *,
    analyze_result: list[dict[str, Any]] | None = None,
    raise_not_detected: bool = False,
) -> None:
    """Shadow the uninstalled deepface package with a stand-in for the two symbols we use."""

    class FaceNotDetected(ValueError):
        pass

    def analyze(img_path: Any, actions: Any, detector_backend: Any) -> list[dict[str, Any]]:
        if raise_not_detected:
            raise FaceNotDetected("no face")
        assert analyze_result is not None
        return analyze_result

    deepface_mod = types.ModuleType("deepface")
    deepface_mod.DeepFace = types.SimpleNamespace(analyze=analyze)  # type: ignore[attr-defined]

    exceptions_mod = types.ModuleType("deepface.modules.exceptions")
    exceptions_mod.FaceNotDetected = FaceNotDetected  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "deepface", deepface_mod)
    monkeypatch.setitem(sys.modules, "deepface.modules.exceptions", exceptions_mod)


def test_returns_none_when_no_face_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A frame with no detectable face should yield None, not raise."""
    _install_fake_deepface(monkeypatch, raise_not_detected=True)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    assert classify_dominant_emotion(frame) is None


def test_picks_dominant_emotion_of_most_confident_face(monkeypatch: pytest.MonkeyPatch) -> None:
    """With multiple faces, the highest-confidence face's emotion wins; scores come back as 0-1 probabilities."""
    faces = [
        {"face_confidence": 0.30, "dominant_emotion": "happy", "emotion": {"happy": 80.0, "sad": 20.0}},
        {"face_confidence": 0.95, "dominant_emotion": "sad", "emotion": {"happy": 25.0, "sad": 75.0}},
    ]
    _install_fake_deepface(monkeypatch, analyze_result=faces)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    emotion, scores = classify_dominant_emotion(frame)

    assert emotion == "sad"
    assert scores == {"happy": 0.25, "sad": 0.75}
