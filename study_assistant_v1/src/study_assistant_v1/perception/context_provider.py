"""ContextProvider interface and Null implementation."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class ContextProvider(Protocol):
    """Supplies context about what the student is currently working on.

    Real implementations extract text from an uploaded PDF, a URL,
    or a typed description. The context string is injected into the
    LLM's system prompt so interventions can reference the actual task.
    The Null implementation returns an empty string, meaning the LLM
    has no task context and gives generic responses.
    """

    def get_context(self) -> str:
        """Return a text description of the student's current task.

        Returns an empty string if no context has been loaded.
        The returned text is passed directly to the LLM, so keep it
        under ~500 words to stay within token budgets.
        """
        ...

    def load(self, source: str) -> None:
        """Load context from a source (file path, URL, or plain text).

        Called once at session start when the student provides their assignment.
        Implementations decide how to parse the source.
        """
        ...

    def start(self) -> None:
        """Perform any initialisation (e.g. load models for PDF parsing)."""
        ...

    def stop(self) -> None:
        """Release resources."""
        ...


class NullContextProvider:
    """No-op implementation — provides no task context.

    Used in both the control condition and whenever no assignment is provided.
    The LLM still works but gives generic rather than task-specific responses.
    """

    def get_context(self) -> str:
        return ""

    def load(self, source: str) -> None:
        pass

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass
