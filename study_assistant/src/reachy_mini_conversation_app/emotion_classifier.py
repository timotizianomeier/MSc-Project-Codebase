"""DeepFace-backed classification of a camera frame's dominant emotion."""

import numpy as np
from numpy.typing import NDArray

from reachy_mini_conversation_app.config import config


def classify_dominant_emotion(frame: NDArray[np.uint8]) -> tuple[str, dict[str, float]] | None:
    """Return (dominant_emotion, per-emotion probabilities) for a BGR frame, or None when no face is detected.

    Probabilities are normalized to 0-1 and sum to ~1 regardless of classifier backend.
    """
    # Imported lazily: deepface pulls in TensorFlow and is an opt-in extra (pyproject `emotion`).
    from deepface import DeepFace
    from deepface.modules.exceptions import FaceNotDetected

    try:
        faces = DeepFace.analyze(img_path=frame, actions=["emotion"], detector_backend=config.EMOTION_DETECTOR_BACKEND)
    except FaceNotDetected:
        return None

    best_face = max(faces, key=lambda face: face["face_confidence"])
    # deepface reports percentages summing to 100; normalize to the 0-1 contract.
    scores = {label: float(score) / 100.0 for label, score in best_face["emotion"].items()}
    return str(best_face["dominant_emotion"]), scores
