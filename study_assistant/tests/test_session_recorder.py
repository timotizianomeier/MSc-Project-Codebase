from __future__ import annotations
import wave
from pathlib import Path

import numpy as np
import pytest

from reachy_mini_conversation_app import session_recorder
from reachy_mini_conversation_app.session_recorder import SessionRecorder, get_recorder


@pytest.fixture()
def recorder(tmp_path: Path) -> SessionRecorder:
    """Build a recorder rooted in a temp dir; the tests close it themselves."""
    return SessionRecorder(tmp_path)


def _read_wav(path: Path) -> tuple[int, int, int, bytes]:
    """Return (channels, sampwidth, framerate, frames) of a finalized WAV file."""
    with wave.open(str(path), "rb") as reader:
        return (
            reader.getnchannels(),
            reader.getsampwidth(),
            reader.getframerate(),
            reader.readframes(reader.getnframes()),
        )


def test_user_audio_written_as_mono_int16_wav(recorder: SessionRecorder) -> None:
    """Mic frames should land in user_audio.wav with the tap's sample rate and int16 payload."""
    samples = np.array([0, 1000, -1000, 32767], dtype=np.int16)
    recorder.write_user_audio(16000, samples)
    recorder.close()

    channels, sampwidth, framerate, frames = _read_wav(recorder.session_dir / "user_audio.wav")
    assert (channels, sampwidth, framerate) == (1, 2, 16000)
    assert frames == samples.tobytes()


def test_robot_audio_accepts_float32_and_converts(recorder: SessionRecorder) -> None:
    """Playback deltas may arrive as float32 in [-1, 1]; the track must still be int16."""
    samples = np.array([0.0, 0.5, -0.5], dtype=np.float32)
    recorder.write_robot_audio(24000, samples)
    recorder.close()

    channels, sampwidth, framerate, frames = _read_wav(recorder.session_dir / "robot_audio.wav")
    assert (channels, sampwidth, framerate) == (1, 2, 24000)
    assert np.frombuffer(frames, dtype=np.int16).tolist() == [0, 16383, -16383]


def test_stereo_channels_first_is_collapsed_to_first_channel(recorder: SessionRecorder) -> None:
    """The tap sees raw mic frames pre-reshape; the recorder mirrors the loops' mono collapse."""
    stereo = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)  # channels-first (2, 3)
    recorder.write_user_audio(16000, stereo)
    recorder.close()

    _, _, _, frames = _read_wav(recorder.session_dir / "user_audio.wav")
    assert np.frombuffer(frames, dtype=np.int16).tolist() == [1, 2, 3]


def test_video_frames_saved_without_overwrites(recorder: SessionRecorder) -> None:
    """Back-to-back frames must produce distinct files, in capture order by name."""
    recorder.write_video_frame(b"jpeg-one")
    recorder.write_video_frame(b"jpeg-two")
    recorder.close()

    frames = sorted((recorder.session_dir / "frames").glob("*.jpg"))
    assert len(frames) == 2
    assert [path.read_bytes() for path in frames] == [b"jpeg-one", b"jpeg-two"]


def test_write_error_disables_stream_without_raising(
    recorder: SessionRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing write must latch that stream off and never propagate into the audio loops."""
    monkeypatch.setattr(recorder, "_open_wav", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")))
    recorder.write_user_audio(16000, np.zeros(4, dtype=np.int16))
    recorder.write_user_audio(16000, np.zeros(4, dtype=np.int16))  # latched: no second attempt

    assert recorder._user_failed is True
    assert not (recorder.session_dir / "user_audio.wav").exists()


def test_close_is_idempotent_and_blocks_further_writes(recorder: SessionRecorder) -> None:
    """Writes after close() are silent no-ops and a second close() does not raise."""
    recorder.write_user_audio(16000, np.zeros(4, dtype=np.int16))
    recorder.close()
    recorder.close()
    recorder.write_user_audio(16000, np.zeros(4, dtype=np.int16))
    recorder.write_video_frame(b"late")

    assert not (recorder.session_dir / "frames").exists()


def test_nothing_touches_disk_until_first_write(recorder: SessionRecorder) -> None:
    """Constructing (and closing) an unused recorder must not create the session folder."""
    recorder.close()

    assert not recorder.session_dir.exists()


def test_get_recorder_disabled_when_knob_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty SESSION_RECORDING_DIR resolves to no recorder, once, process-wide."""
    monkeypatch.setattr(session_recorder, "_recorder", None)
    monkeypatch.setattr(session_recorder, "_recorder_resolved", False)
    monkeypatch.setattr(session_recorder.config, "SESSION_RECORDING_DIR", "")

    assert get_recorder() is None


def test_get_recorder_returns_shared_instance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """With the knob set, every tap site must see the same recorder (one session folder)."""
    monkeypatch.setattr(session_recorder, "_recorder", None)
    monkeypatch.setattr(session_recorder, "_recorder_resolved", False)
    monkeypatch.setattr(session_recorder.config, "SESSION_RECORDING_DIR", str(tmp_path))

    first = get_recorder()
    assert first is not None
    assert first.session_dir.parent == tmp_path
    assert get_recorder() is first
