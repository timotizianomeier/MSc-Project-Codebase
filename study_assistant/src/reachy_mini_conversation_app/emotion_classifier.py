"""DeepFace-backed classification of a camera frame's dominant emotion."""

import numpy as np
from numpy.typing import NDArray

from reachy_mini_conversation_app.config import config


def classify_dominant_emotion(frame: NDArray[np.uint8]) -> str | None:
    """Return the dominant emotion in a BGR frame, or None when no face is detected."""
    # Imported lazily: deepface pulls in TensorFlow and is an opt-in extra (pyproject `emotion`).
    from deepface import DeepFace
    from deepface.modules.exceptions import FaceNotDetected

    try:
        faces = DeepFace.analyze(img_path=frame, actions=["emotion"], detector_backend=config.EMOTION_DETECTOR_BACKEND)
    except FaceNotDetected:
        return None

    best_face = max(faces, key=lambda face: face["face_confidence"])
    return str(best_face["dominant_emotion"])
