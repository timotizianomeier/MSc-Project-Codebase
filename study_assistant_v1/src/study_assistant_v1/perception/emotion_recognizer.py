"""EmotionRecognizer interface and Null implementation."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmotionRecognizer(Protocol):
    """Detects the student's dominant emotion from a camera frame.

    Real implementations (e.g. DeepFace) call get_emotion() periodically
    and return a label from the model's emotion vocabulary.
    The Null implementation always returns "neutral" so the control
    condition runs without any emotion signal.
    """

    def get_emotion(self) -> str | None:
        """Return the current dominant emotion label, or None if no face is detected.

        Labels follow the DeepFace vocabulary:
        'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'
        """
        ...

    def start(self) -> None:
        """Start any background threads or resources needed by this recognizer."""
        ...

    def stop(self) -> None:
        """Release resources and stop background threads."""
        ...


class NullEmotionRecognizer:
    """No-op implementation used in the control condition.

    Always reports 'neutral' so the rest of the pipeline sees a valid value
    without triggering any emotion-based logic.
    """

    def get_emotion(self) -> str | None:
        return "neutral"

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass
