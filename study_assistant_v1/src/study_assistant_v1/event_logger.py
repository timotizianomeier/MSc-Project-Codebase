"""Event logger for study session data collection.

This is research-critical code — the CSV it writes is your primary data source.
Understand the schema and vocabulary before running a study session.

Schema (one row per event):
    timestamp_iso      UTC time, ISO 8601 with milliseconds.
                       Always UTC so rows from different machines sort correctly.
    session_id         Unique identifier for this app run, auto-generated at startup.
                       Format: <participant_id>_<YYYYMMDD_HHMMSS>
    participant_id     From StudyConfig. Links rows to participant-level data.
    condition          From StudyConfig ('intervention' or 'control').
                       Embedded in every row so your analysis never needs a
                       separate condition lookup.
    event_type         What happened. See EVENT_TYPES below.
    module             Which component emitted this event (e.g. 'EngagementDetector').
                       'session' for lifecycle events, 'LLM' for robot speech events.
    value              The numeric or string value associated with the event.
                       Kept as a string for CSV simplicity; cast in analysis code.
    metadata           JSON blob for extensible extra data.
                       Add new fields here without changing the CSV header.

Event vocabulary (event_type values):
    session_start      Logged once when the app starts a session.
    session_end        Logged once on clean shutdown.
    pomodoro_start     A focus interval has begun.  value = interval number (1, 2, …)
    pomodoro_end       A focus interval has ended.  value = interval number
    break_start        A break has begun.            value = break number
    break_end          A break has ended.            value = break number
    emotion_detected   Periodic sample from EmotionRecognizer. value = emotion label
    engagement_sampled Periodic sample from EngagementDetector. value = score (0.0–1.0)
    inattention_detected  Score crossed threshold → intervention pipeline may fire.
                       value = engagement score at threshold crossing
    intervention_triggered  Robot decided to speak.  value = short reason tag
    student_spoke      STT transcript received.      value = transcript text
    robot_spoke        TTS output emitted.           value = robot utterance text
    context_loaded     ContextProvider delivered non-empty context.
                       value = first 80 chars of context (not the full text)

Thread safety: all writes are serialised through a threading.Lock so perception
threads and the main LLM thread can log concurrently without corrupting rows.
"""

import csv
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

# Canonical column order for the CSV.
# Do not reorder these — existing analysis code will depend on position.
COLUMNS = [
    "timestamp_iso",
    "session_id",
    "participant_id",
    "condition",
    "event_type",
    "module",
    "value",
    "metadata",
]


class EventLogger:
    """Append-only CSV event logger.

    Usage:
        log = EventLogger(study_config, log_dir=Path("logs"))
        log.start()                       # opens the file, writes session_start
        log.log("emotion_detected", "EmotionRecognizer", "happy")
        log.stop()                        # writes session_end, closes the file
    """

    def __init__(self, study_config: Any, log_dir: Path | None = None) -> None:
        """Initialise the logger.

        Args:
            study_config: A StudyConfig instance (imported lazily to avoid
                          circular imports).
            log_dir: Directory to write CSV files into.
                     Defaults to <app_root>/logs/.
        """
        self._config = study_config
        self._lock = threading.Lock()
        self._file = None
        self._writer = None

        if log_dir is None:
            log_dir = Path(__file__).parent.parent.parent / "logs"
        self._log_dir = log_dir

        # session_id is stable for the lifetime of this object.
        now = datetime.now(timezone.utc)
        self._session_id = f"{study_config.participant_id}_{now.strftime('%Y%m%d_%H%M%S')}"

    @property
    def session_id(self) -> str:
        return self._session_id

    def start(self) -> None:
        """Open the CSV file and write the session_start row."""
        self._log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self._log_dir / f"{self._session_id}.csv"

        self._file = open(log_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=COLUMNS)
        self._writer.writeheader()
        self._file.flush()

        self.log(
            event_type="session_start",
            module="session",
            value="",
            metadata={
                "condition": self._config.condition,
                "emotion_backend": self._config.emotion_backend,
                "engagement_backend": self._config.engagement_backend,
                "context_backend": self._config.context_backend,
                "intervention_enabled": self._config.intervention_enabled,
            },
        )
        logger.info("EventLogger started: %s", log_path)

    def stop(self) -> None:
        """Write the session_end row and close the file."""
        if self._writer is None:
            return
        self.log("session_end", "session", "")
        with self._lock:
            if self._file:
                self._file.flush()
                self._file.close()
                self._file = None
                self._writer = None

    def log(
        self,
        event_type: str,
        module: str,
        value: str = "",
        metadata: dict | None = None,
    ) -> None:
        """Append one event row to the CSV.

        This method is thread-safe and returns immediately.

        Args:
            event_type: One of the event_type values documented above.
            module:     Which component is logging this (e.g. 'EngagementDetector').
            value:      The event's primary value as a string.
            metadata:   Optional dict of extra fields, serialised to JSON.
        """
        if self._writer is None:
            logger.warning("EventLogger.log() called before start() — dropping event: %s", event_type)
            return

        row = {
            "timestamp_iso": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "session_id": self._session_id,
            "participant_id": self._config.participant_id,
            "condition": self._config.condition,
            "event_type": event_type,
            "module": module,
            "value": value,
            "metadata": json.dumps(metadata) if metadata else "{}",
        }

        with self._lock:
            if self._writer:
                self._writer.writerow(row)
                self._file.flush()
