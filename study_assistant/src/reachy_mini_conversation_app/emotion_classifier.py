"""Classification of a camera frame's dominant emotion (deepface or EmotiEffLib backend)."""

import numpy as np
from numpy.typing import NDArray

from reachy_mini_conversation_app.config import config

# EmotiEffLib is AffectNet-trained; map its labels onto deepface's vocabulary so the
# monitor's negative-emotion set works unchanged. "contempt" has no deepface equivalent
# and stays as-is (its gate treatment is a confidence-aware-gate decision).
_EMOTIEFFLIB_LABEL_MAP = {"anger": "angry", "happiness": "happy", "sadness": "sad"}
_CROP_MARGIN = 0.2  # fraction of box size added per side; the model saw tight crops
_emotiefflib_recognizer = None


def classify_dominant_emotion(frame: NDArray[np.uint8]) -> tuple[str, dict[str, float]] | None:
    """Return (dominant_emotion, per-emotion probabilities) for a BGR frame, or None when no face is detected.

    Probabilities are normalized to 0-1 and sum to ~1 regardless of classifier backend.
    """
    if config.EMOTION_CLASSIFIER_BACKEND == "emotiefflib":
        return _classify_emotiefflib(frame)
    return _classify_deepface(frame)


def _classify_deepface(frame: NDArray[np.uint8]) -> tuple[str, dict[str, float]] | None:
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


def _classify_emotiefflib(frame: NDArray[np.uint8]) -> tuple[str, dict[str, float]] | None:
    # Imported lazily: emotiefflib/onnx are opt-in extras (pyproject `emotion`).
    import cv2
    from emotiefflib.facial_analysis import EmotiEffLibRecognizer

    global _emotiefflib_recognizer
    if _emotiefflib_recognizer is None:
        _emotiefflib_recognizer = EmotiEffLibRecognizer(engine="onnx", model_name="enet_b0_8_best_vgaf")

    # Same haar detection + margin crop as the offline comparison methodology: the
    # detector is held constant so the classifier backend is the only variable.
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    mx, my = int(w * _CROP_MARGIN), int(h * _CROP_MARGIN)
    crop = frame[max(0, y - my) : min(frame.shape[0], y + h + my), max(0, x - mx) : min(frame.shape[1], x + w + mx)]

    _, raw_scores = _emotiefflib_recognizer.predict_emotions(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), logits=False)
    labels = [
        _EMOTIEFFLIB_LABEL_MAP.get(name.lower(), name.lower())
        for _, name in sorted(_emotiefflib_recognizer.idx_to_emotion_class.items())
    ]
    scores = {label: float(p) for label, p in zip(labels, np.asarray(raw_scores)[0])}
    return max(scores, key=scores.get), scores
