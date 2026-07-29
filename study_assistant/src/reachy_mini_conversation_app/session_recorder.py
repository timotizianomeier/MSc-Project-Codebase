"""Full-session audio/video recording for study data collection.

Writes one per-run folder under ``SESSION_RECORDING_DIR`` containing two mono WAV
tracks (participant mic, robot speech) and the engagement-cadence camera frames as
JPEGs. The recorder is process-wide (``get_recorder()``) because the audio taps live
in the console's record/play loops while the video tap lives in the realtime
handler's engagement capture. Every write is failure-soft: an error disables that
stream and logs a warning — recording must never take down a study session.
"""

import time
import wave
import logging
import threading
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.streaming import AudioArray, audio_to_int16


logger = logging.getLogger(__name__)


def _as_mono_int16(samples: AudioArray) -> NDArray[np.int16]:
    """Collapse to mono and cast to int16 (mirrors the reshape in the audio loops)."""
    if samples.ndim == 2:
        # channels-last convention
        if samples.shape[1] > samples.shape[0]:
            samples = samples.T
        samples = samples[:, 0]
    return audio_to_int16(samples)


class SessionRecorder:
    """Write one session's A/V record under a timestamped folder."""

    def __init__(self, base_dir: Path) -> None:
        """Prepare a recorder rooted at ``base_dir/<run timestamp>`` (nothing touches disk yet)."""
        # Per-run subfolder so reruns never mix or overwrite earlier sessions
        # (same convention as the emotion frame dumps).
        self.session_dir = base_dir / time.strftime("%Y-%m-%d_%H%M")
        self._frames_dir = self.session_dir / "frames"
        self._user_wav: wave.Wave_write | None = None
        self._robot_wav: wave.Wave_write | None = None
        self._user_failed = False
        self._robot_failed = False
        self._video_failed = False
        self._frame_count = 0
        self._closed = False

    def _open_wav(self, filename: str, sample_rate: int) -> wave.Wave_write:
        """Open a mono 16-bit WAV writer inside the session folder."""
        self.session_dir.mkdir(parents=True, exist_ok=True)
        writer = wave.open(str(self.session_dir / filename), "wb")
        writer.setnchannels(1)
        writer.setsampwidth(2)  # int16
        writer.setframerate(sample_rate)
        logger.info("Session recording: writing %s at %d Hz", self.session_dir / filename, sample_rate)
        return writer

    def write_user_audio(self, sample_rate: int, samples: AudioArray) -> None:
        """Append one mic frame to user_audio.wav (opened lazily on the first frame)."""
        if self._closed or self._user_failed:
            return
        try:
            if self._user_wav is None:
                self._user_wav = self._open_wav("user_audio.wav", sample_rate)
            self._user_wav.writeframes(_as_mono_int16(samples).tobytes())
        except Exception as e:
            self._user_failed = True
            logger.warning("Session recording: user audio stopped after write error: %s", e)

    def write_robot_audio(self, sample_rate: int, samples: AudioArray) -> None:
        """Append one playback frame to robot_audio.wav (opened lazily on the first frame)."""
        if self._closed or self._robot_failed:
            return
        try:
            if self._robot_wav is None:
                self._robot_wav = self._open_wav("robot_audio.wav", sample_rate)
            self._robot_wav.writeframes(_as_mono_int16(samples).tobytes())
        except Exception as e:
            self._robot_failed = True
            logger.warning("Session recording: robot audio stopped after write error: %s", e)

    def write_video_frame(self, jpeg_bytes: bytes) -> None:
        """Save one camera frame (already JPEG-encoded) under frames/."""
        if self._closed or self._video_failed:
            return
        try:
            self._frames_dir.mkdir(parents=True, exist_ok=True)
            # Leading sequence number makes names collision-proof and lexically chronological;
            # the HHMMSS_mmm wall-clock part is what aligns frames with the audio tracks.
            now = time.time()
            name = (
                f"{self._frame_count:06d}_{time.strftime('%H%M%S', time.localtime(now))}_{int(now % 1 * 1000):03d}.jpg"
            )
            self._frame_count += 1
            (self._frames_dir / name).write_bytes(jpeg_bytes)
        except Exception as e:
            self._video_failed = True
            logger.warning("Session recording: video frames stopped after write error: %s", e)

    def close(self) -> None:
        """Close the WAV writers (idempotent; WAV headers are only finalized here)."""
        if self._closed:
            return
        self._closed = True
        for writer, label in ((self._user_wav, "user"), (self._robot_wav, "robot")):
            if writer is not None:
                try:
                    writer.close()
                except Exception as e:
                    logger.warning("Session recording: closing %s track failed: %s", label, e)
        self._user_wav = None
        self._robot_wav = None
        if self.session_dir.exists():
            logger.info("Session recording: closed %s", self.session_dir)


_recorder_lock = threading.Lock()
_recorder: SessionRecorder | None = None
_recorder_resolved = False


def get_recorder() -> SessionRecorder | None:
    """Return the process-wide recorder, or None when SESSION_RECORDING_DIR is unset."""
    global _recorder, _recorder_resolved
    with _recorder_lock:
        if not _recorder_resolved:
            _recorder_resolved = True
            if config.SESSION_RECORDING_DIR:
                _recorder = SessionRecorder(Path(config.SESSION_RECORDING_DIR))
        return _recorder
