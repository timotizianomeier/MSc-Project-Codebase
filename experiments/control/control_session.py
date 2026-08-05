"""No-robot control-condition session runner.

Runs the study's sensing/decision/logging pipeline against a plain USB webcam:
no Reachy, no daemon, no speech backend. Everything below the camera is imported
from the treatment app (reachy_mini_conversation_app) so both conditions share
one implementation of the monitors, thresholds, cadences, recorder, and
participant page — only the frame/mic source and the (counterfactual)
intervention delivery differ. Poll-loop bodies and log format strings are copied
verbatim from huggingface_realtime.py; keep them in lockstep when either side
changes.

Run from study_assistant/ so the shared .env applies (requires
`pip install sounddevice` in the study_assistant venv for Mac mic capture):

    .venv/bin/python ../experiments/control/control_session.py \
        --camera-index 0 --duration 45 --debug 2>&1 | tee -i app.log
"""

from __future__ import annotations

import time
import asyncio
import logging
import argparse
import threading
from typing import Any
from pathlib import Path
from collections import deque

import cv2
import httpx
import uvicorn
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from numpy.typing import NDArray
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from collections.abc import Awaitable, Callable

from reachy_mini.io.jsonrpc import JsonRpcError
from reachy_mini.apps.jsonrpc_server import JsonRpcServer

import reachy_mini_conversation_app
from reachy_mini_conversation_app.utils import setup_logger
from reachy_mini_conversation_app.config import config, refresh_runtime_config_from_env
from reachy_mini_conversation_app.emotion_monitor import EmotionMonitor, negative_mass
from reachy_mini_conversation_app.emotion_classifier import classify_dominant_emotion
from reachy_mini_conversation_app.engagement_client import FRAMES_PER_SCORE, fetch_engagement_score
from reachy_mini_conversation_app.engagement_monitor import EngagementMonitor
from reachy_mini_conversation_app.session_recorder import get_recorder

# Shared cadences — imported (not copied) so the two conditions cannot drift.
from reachy_mini_conversation_app.huggingface_realtime import (
    _EMOTION_POLL_INTERVAL_S,
    _ENGAGEMENT_FRAME_INTERVAL_S,
    _ENGAGEMENT_SCORE_EVERY_TICKS,
)


# Package-style logger name so control lines carry the same shape as app lines.
logger = logging.getLogger("reachy_mini_conversation_app.control_session")

_STUDY_ASSISTANT_DIR = Path(__file__).resolve().parents[2] / "study_assistant"
_AUDIO_SAMPLE_RATE = 16_000
_CAMERA_WARMUP_TIMEOUT_S = 5.0


class WebcamSource:
    """Latest-frame webcam reader, mirroring the daemon's media.get_frame()/get_frame_jpeg() semantics."""

    def __init__(self, index: int) -> None:
        self._capture = cv2.VideoCapture(index)
        if not self._capture.isOpened():
            raise RuntimeError(f"Cannot open webcam at index {index}")
        self._latest: NDArray[np.uint8] | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._read_loop, name="webcam-reader", daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._capture.read()
            if ok:
                with self._lock:
                    self._latest = frame
            else:
                time.sleep(0.1)

    def get_frame(self) -> NDArray[np.uint8] | None:
        """Return a copy of the most recent BGR frame, or None before the first capture."""
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def get_frame_jpeg(self) -> bytes | None:
        """Return the most recent frame encoded as JPEG bytes, or None if unavailable."""
        frame = self.get_frame()
        if frame is None:
            return None
        ok, buffer = cv2.imencode(".jpg", frame)
        return buffer.tobytes() if ok else None

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._capture.release()


class ControlSession:
    """Sensing + counterfactual-intervention loops for a robot-free control session."""

    def __init__(self, camera: WebcamSource, loop: asyncio.AbstractEventLoop) -> None:
        self.camera = camera
        self._loop = loop
        self.last_activity_time = time.monotonic()
        self._emotion_monitor = EmotionMonitor(config.EMOTION_NEGATIVE_THRESHOLD)
        self._engagement_monitor = EngagementMonitor()
        self._engagement_frames: deque[bytes] = deque(maxlen=FRAMES_PER_SCORE)
        self._engagement_http = httpx.Client()
        self._emotion_frame_dump_dir: Path | None = (
            Path(config.EMOTION_FRAME_DUMP_DIR) if config.EMOTION_FRAME_DUMP_DIR else None
        )
        self._study_session_started_at: float | None = None
        self._study_session_ended = False

    def session_gate_open(self) -> bool:
        """Return whether the study session is running (started and not yet timed out)."""
        return self._study_session_started_at is not None and not self._study_session_ended

    def _mark_activity(self, reason: str) -> None:
        # conversation_handler._mark_activity, minus the observer plumbing.
        self.last_activity_time = time.monotonic()
        logger.debug("last activity time updated to %s (%s)", self.last_activity_time, reason)

    def start_study_session(self) -> bool:
        """Start the session (idempotent). Called from the web server thread."""
        if self._study_session_started_at is not None:
            logger.info("session.start ignored — session already started")
            return False

        self._study_session_started_at = time.monotonic()
        logger.info("=" * 72)
        logger.info(
            "SESSION START — duration %.1f min%s",
            config.SESSION_DURATION_MINUTES,
            " (CONTROL condition)",
        )
        logger.info("=" * 72)
        # The timer must live on the session loop, not the web server's
        # (console.py's _on_app_loop lesson).
        asyncio.run_coroutine_threadsafe(self._session_timer(), self._loop)
        logger.info("CONTROL: startup greeting suppressed")
        return True

    async def _session_timer(self) -> None:
        """Sleep out the session, then log the END marker and close the gate."""
        await asyncio.sleep(config.SESSION_DURATION_MINUTES * 60.0)
        self._study_session_ended = True
        logger.info("=" * 72)
        logger.info("SESSION END — %.1f min elapsed", config.SESSION_DURATION_MINUTES)
        logger.info("=" * 72)
        logger.info("CONTROL: session-over announcement suppressed")

    def receive_task_context(self, text: str) -> None:
        """Accept a participant context submission — logged, never forwarded (no model exists here).

        Newlines escaped so multi-line pastes land on one log line (parse_app_log.py).
        """
        logger.info("CONTROL: task context received but not forwarded: %s", text.replace("\n", "\\n"))

    # ── Emotion loop — mirrors HuggingFaceRealtimeHandler._emotion_poll_loop ──

    async def emotion_poll_loop(self) -> None:
        """Sample the camera on a fixed interval, feeding the emotion monitor and its interventions."""
        while True:
            await asyncio.sleep(_EMOTION_POLL_INTERVAL_S)
            try:
                await self._poll_emotion_once()
            except Exception:
                logger.exception("Emotion poll iteration failed; continuing")

    def _dump_emotion_frame(self, frame: NDArray[np.uint8], emotion: str | None) -> None:
        """Save the analyzed frame, named by its result, for post-hoc detector/classifier analysis."""
        if self._emotion_frame_dump_dir is None:
            return
        try:
            self._emotion_frame_dump_dir.mkdir(parents=True, exist_ok=True)
            name = f"{time.strftime('%H%M%S')}_{emotion or 'noface'}.jpg"
            cv2.imwrite(str(self._emotion_frame_dump_dir / name), frame)
        except Exception as e:
            logger.warning("Emotion frame dump failed: %s", e)

    async def _poll_emotion_once(self) -> None:
        """Classify the current frame, record it, and log a counterfactual intervention if warranted."""
        if not self.session_gate_open():
            return
        frame = self.camera.get_frame()
        if frame is None:
            logger.debug("Emotion poll: no frame available")
            return

        classification = await asyncio.to_thread(classify_dominant_emotion, frame)
        emotion, emotion_scores = classification if classification is not None else (None, None)
        if self._emotion_frame_dump_dir is not None:
            await asyncio.to_thread(self._dump_emotion_frame, frame, emotion)
        if emotion is None:
            logger.debug("Emotion poll: no face detected")
            return

        now = time.monotonic()
        self._emotion_monitor.record(negative_mass(emotion_scores or {}), now)

        negative_share = self._emotion_monitor.negative_share()
        response_done = True  # nothing ever speaks in the no-robot condition
        interaction_gap = now - self.last_activity_time
        last_trigger = self._emotion_monitor.last_trigger_time
        intervention_gap = now - last_trigger if last_trigger is not None else None
        logger.debug(
            "Emotion poll: emotion=%s scores=%s negative_share=%.2f (need>%.2f) response_done=%s "
            "interaction_gap=%.1fs (need>%.0fs) intervention_gap=%s (need>%.0fs)",
            emotion,
            {label: round(score, 2) for label, score in emotion_scores.items()} if emotion_scores else None,
            negative_share,
            self._emotion_monitor.NEGATIVE_THRESHOLD,
            response_done,
            interaction_gap,
            self._emotion_monitor.INTERACTION_COOLDOWN_SECONDS,
            f"{intervention_gap:.1f}s" if intervention_gap is not None else "never",
            self._emotion_monitor.INTERVENTION_COOLDOWN_SECONDS,
        )

        if self._emotion_monitor.should_intervene(now, response_done, self.last_activity_time):
            logger.info(
                "CONTROL: would have sent emotion intervention (negative_share=%.2f); cooldowns reset as if sent",
                negative_share,
            )
            self._mark_activity("emotion_intervention_counterfactual")
            self._emotion_monitor.mark_intervened(now)

    # ── Engagement loop — mirrors HuggingFaceRealtimeHandler._engagement_poll_loop ──

    async def engagement_poll_loop(self) -> None:
        """Capture frames on a fixed cadence and periodically score the window for disengagement."""
        tick = 0
        while True:
            await asyncio.sleep(_ENGAGEMENT_FRAME_INTERVAL_S)
            tick += 1
            try:
                await self._poll_engagement_once(score_now=(tick % _ENGAGEMENT_SCORE_EVERY_TICKS == 0))
            except Exception:
                logger.exception("Engagement poll iteration failed; continuing")

    async def _poll_engagement_once(self, score_now: bool) -> None:
        """Buffer the current frame and, when due, score the window and gate a counterfactual."""
        if not self.session_gate_open():
            return
        frame = self.camera.get_frame_jpeg()
        if frame is None:
            logger.debug("Engagement poll: no frame available")
            return
        self._engagement_frames.append(frame)
        session_recorder = get_recorder()
        if session_recorder is not None:
            await asyncio.to_thread(session_recorder.write_video_frame, frame)

        if not score_now or len(self._engagement_frames) < FRAMES_PER_SCORE:
            return

        score = await asyncio.to_thread(fetch_engagement_score, self._engagement_http, list(self._engagement_frames))
        if score is None:
            return

        now = time.monotonic()
        self._engagement_monitor.record(score, now)

        average = self._engagement_monitor.average_score()
        response_done = True  # nothing ever speaks in the no-robot condition
        interaction_gap = now - self.last_activity_time
        last_trigger = self._engagement_monitor.last_trigger_time
        intervention_gap = now - last_trigger if last_trigger is not None else None
        logger.debug(
            "Engagement poll: score=%.2f average=%s (need<%.2f) response_done=%s "
            "interaction_gap=%.1fs (need>%.0fs) intervention_gap=%s (need>%.0fs)",
            score,
            f"{average:.2f}" if average is not None else "n/a",
            config.ENGAGEMENT_THRESHOLD,
            response_done,
            interaction_gap,
            self._engagement_monitor.INTERACTION_COOLDOWN_SECONDS,
            f"{intervention_gap:.1f}s" if intervention_gap is not None else "never",
            self._engagement_monitor.INTERVENTION_COOLDOWN_SECONDS,
        )

        if self._engagement_monitor.should_intervene(now, response_done, self.last_activity_time):
            logger.info(
                "CONTROL: would have sent engagement intervention (average=%.2f); cooldowns reset as if sent",
                average,
            )
            self._mark_activity("engagement_intervention_counterfactual")
            self._engagement_monitor.mark_intervened(now)

    def close(self) -> None:
        self._engagement_http.close()


def build_web_app(session: ControlSession) -> FastAPI:
    """Serve the control participant page (Start button only) plus the RPC methods it calls.

    The page lives next to this script; it borrows the app's /static assets
    (style.css, js/api.js) so styling and RPC behavior stay identical.
    """
    static_dir = Path(reachy_mini_conversation_app.__file__).parent / "static"
    control_page = Path(__file__).parent / "participant.html"
    app = FastAPI()

    @app.middleware("http")
    async def _no_cache(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        """Serve everything no-store so browsers don't keep stale UI modules."""
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/participant")
    def _participant() -> FileResponse:
        return FileResponse(str(control_page))

    @app.get("/favicon.ico")
    def _favicon() -> Response:
        return Response(status_code=204)

    rpc = JsonRpcServer()

    @rpc.method("session.status")  # type: ignore[untyped-decorator]
    def _rpc_session_status(params: dict[str, object]) -> dict[str, object]:
        return {
            "gate_enabled": True,
            "started": session._study_session_started_at is not None,
            "active": session.session_gate_open(),
            "duration_minutes": config.SESSION_DURATION_MINUTES,
        }

    @rpc.method("session.start")  # type: ignore[untyped-decorator]
    def _rpc_session_start(params: dict[str, object]) -> dict[str, object]:
        started = session.start_study_session()
        return {
            "started": started,
            "active": session.session_gate_open(),
            "duration_minutes": config.SESSION_DURATION_MINUTES,
        }

    @rpc.method("conversation.context")  # type: ignore[untyped-decorator]
    def _rpc_context(params: dict[str, object]) -> dict[str, object]:
        text = str(params.get("text", "")).strip()
        if not text:
            raise JsonRpcError("context requires 'text'", reason="invalid_params", code=-32602)
        if not session.session_gate_open():
            raise JsonRpcError("session not started", reason="session_not_started")
        session.receive_task_context(text)
        return {"ok": True}

    rpc.mount(app)
    return app


def start_audio_recording() -> sd.InputStream | None:
    """Record the room via the machine's mic into the session recorder's user_audio.wav."""
    recorder = get_recorder()
    if recorder is None:
        return None

    def _callback(indata: NDArray[np.int16], frames: int, time_info: Any, status: sd.CallbackFlags) -> None:
        if status:
            logger.debug("Mic status: %s", status)
        recorder.write_user_audio(_AUDIO_SAMPLE_RATE, indata.copy())

    stream = sd.InputStream(samplerate=_AUDIO_SAMPLE_RATE, channels=1, dtype="int16", callback=_callback)
    stream.start()
    logger.info("Mic recording started at %d Hz", _AUDIO_SAMPLE_RATE)
    return stream


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="No-robot control-condition session runner")
    parser.add_argument("--camera-index", type=int, default=0, help="cv2 camera index (USB webcam vs built-in)")
    parser.add_argument("--duration", type=float, default=None, help="Session minutes (overrides SESSION_DURATION_MINUTES)")
    parser.add_argument("--port", type=int, default=7860, help="Participant page port")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging (poll lines)")
    return parser.parse_args()


def main() -> None:
    args = parse_cli()
    setup_logger(args.debug)
    logger.info("Starting no-robot control-condition session runner")

    env_path = _STUDY_ASSISTANT_DIR / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path), override=True)
        refresh_runtime_config_from_env()
        logger.info("Loaded configuration from %s", env_path)

    config.CONTROL_MODE = True
    if args.duration is not None:
        config.SESSION_DURATION_MINUTES = args.duration

    logger.warning("=" * 72)
    logger.warning("CONTROL MODE: sensing and recording run normally, but the robot will")
    logger.warning("NOT interact — no greeting, no spoken replies, interventions are only")
    logger.warning("logged as 'CONTROL: would have ...' counterfactuals.")
    logger.warning("=" * 72)

    camera = WebcamSource(args.camera_index)
    deadline = time.monotonic() + _CAMERA_WARMUP_TIMEOUT_S
    frame = camera.get_frame()
    while frame is None and time.monotonic() < deadline:
        time.sleep(0.1)
        frame = camera.get_frame()
    if frame is None:
        camera.close()
        raise SystemExit(f"No frame from webcam index {args.camera_index} after {_CAMERA_WARMUP_TIMEOUT_S:.0f}s")
    logger.info("Webcam %d delivering %dx%d frames", args.camera_index, frame.shape[1], frame.shape[0])

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    session = ControlSession(camera, loop)

    server = uvicorn.Server(uvicorn.Config(build_web_app(session), host="0.0.0.0", port=args.port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True, name="ui-server").start()
    logger.info("Participant page at http://localhost:%d/participant", args.port)

    mic_stream = start_audio_recording()

    try:
        loop.run_until_complete(asyncio.gather(session.engagement_poll_loop(), session.emotion_poll_loop()))
    except KeyboardInterrupt:
        logger.info("Interrupt received — closing control session.")
    finally:
        server.should_exit = True
        if mic_stream is not None:
            mic_stream.stop()
            mic_stream.close()
        recorder = get_recorder()
        if recorder is not None:
            recorder.close()
        session.close()
        camera.close()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()
