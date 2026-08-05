"""Movement system driving sequential primary moves.

Design overview
- Primary moves (emotions, dances, goto, breathing) are mutually exclusive and run
  sequentially.
- There is a single control point to the robot: `ReachyMini.set_target`.
- The control loop runs near 100 Hz and is phase-aligned via a monotonic clock.
- Idle behaviour starts an infinite `BreathingMove` after a short inactivity delay
  unless listening is active.

Threading model
- A dedicated worker thread owns all real-time state and issues `set_target`
  commands.
- Other threads communicate via a command queue (enqueue moves, mark activity,
  toggle listening).

Units and frames
- Antennas and `body_yaw` are in radians.

Safety
- Listening freezes antennas, then blends them back on unfreeze.
- Interpolations and blends are used to avoid jumps at all times.
- `set_target` errors are rate-limited in logs.
"""

from __future__ import annotations
import time
import random
import logging
import threading
from queue import Empty, Queue
from typing import Any, Dict, Tuple
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
from reachy_mini.motion.move import Move
from reachy_mini.utils.interpolation import compose_world_offset, linear_pose_interpolation
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.dance_emotion_moves import EmotionQueueMove


logger = logging.getLogger(__name__)

# Configuration constants
CONTROL_LOOP_FREQUENCY_HZ = 60.0  # Hz - Target frequency for the movement control loop

# Type definitions
FullBodyPose = Tuple[NDArray[np.float32], Tuple[float, float], float]  # (head_pose_4x4, antennas, body_yaw)


class AntennaHeartbeatMove(Move):  # type: ignore
    """Brief antenna-only motion used as an idle 'alive' cue (study body-double signal).

    Head and body hold their current pose for the whole move — only the antennas
    animate, so the camera stays perfectly still. Every waveform starts and ends
    exactly at the held antenna positions (sine envelope), so there is no snap.

    The default parameters reproduce the original mirrored two-bounce flutter;
    `gains`/`bounces`/`stagger_s` parameterize the subtle idle variations
    (see make_idle_antenna_move).
    """

    def __init__(
        self,
        hold_pose: FullBodyPose,
        duration: float = 1.2,
        amplitude_rad: float = 0.25,
        gains: Tuple[float, float] = (-1.0, 1.0),
        bounces: float = 2.0,
        stagger_s: float = 0.0,
        name: str = "double_flutter",
    ):
        """Freeze the given pose and animate antennas around it.

        Args:
            hold_pose: Pose to hold; antennas animate around its antenna values.
            duration: Seconds of animation per antenna.
            amplitude_rad: Peak antenna deflection.
            gains: Per-antenna direction/scale; 0.0 keeps that antenna still.
            bounces: Full oscillations within the move; 0.5 = one smooth bump.
            stagger_s: Delay before the RIGHT antenna starts (ripple effect);
                total move duration grows by this amount.
            name: Variation label, used for logging only.
        """
        self._hold_head = hold_pose[0].astype(np.float64)
        self._hold_antennas = np.array(hold_pose[1], dtype=np.float64)
        self._hold_body_yaw = float(hold_pose[2])
        self._core_duration = duration
        self._duration = duration + stagger_s
        self._amplitude = amplitude_rad
        self._gains = gains
        self._bounces = bounces
        self._delays = (0.0, stagger_s)
        self.name = name

    @property
    def duration(self) -> float:
        """Duration property required by official Move interface."""
        return self._duration

    def evaluate(self, t: float) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        """Evaluate the waveform at time t; antennas rest at hold values at both ends."""
        antennas = self._hold_antennas.copy()
        for i in (0, 1):
            if self._gains[i] == 0.0:
                continue
            phase = min(max((t - self._delays[i]) / self._core_duration, 0.0), 1.0)
            # sin(pi*phase) envelope guarantees rest at start/end;
            # sin(2*bounces*pi*phase) sets how many oscillations fit inside it.
            sway = self._amplitude * np.sin(np.pi * phase) * np.sin(2.0 * self._bounces * np.pi * phase)
            antennas[i] += self._gains[i] * sway
        return (self._hold_head.copy(), antennas, self._hold_body_yaw)


def make_idle_antenna_move(hold_pose: FullBodyPose, rng: random.Random | None = None) -> AntennaHeartbeatMove:
    """Pick one of the subtle antenna-only idle variations (uniformly at random).

    All variations hold head/body still and rest at the held antenna pose at both
    ends. Single-antenna variants randomize the side, so the cue never favors one
    antenna. Amplitudes stay at or below the original flutter's 0.25 rad.
    """
    pick: Callable[..., Any] = rng.choice if rng is not None else random.choice
    kind = str(pick(("double_flutter", "single_flick", "single_twitch", "perk", "ripple")))
    if kind == "single_flick":
        # One antenna, one soft bump — a brief ear-flick.
        side = int(pick((0, 1)))
        gains = (-1.0, 0.0) if side == 0 else (0.0, 1.0)
        return AntennaHeartbeatMove(hold_pose, duration=1.6, amplitude_rad=0.17, gains=gains, bounces=0.5, name=kind)
    if kind == "single_twitch":
        # One antenna, two slow mini-bounces — smaller and busier than the flick.
        side = int(pick((0, 1)))
        gains = (-1.0, 0.0) if side == 0 else (0.0, 1.0)
        return AntennaHeartbeatMove(hold_pose, duration=2.4, amplitude_rad=0.14, gains=gains, bounces=2.0, name=kind)
    if kind == "perk":
        # Both antennas rise once and settle — a slow attentive arc. 2.0s, not
        # slower: below ~0.27 rad/s peak the servos visibly step (05.08 hardware test).
        return AntennaHeartbeatMove(hold_pose, duration=2.0, amplitude_rad=0.17, bounces=0.5, name=kind)
    if kind == "ripple":
        # Left bump, then right bump slightly delayed — a soft wave across the head.
        return AntennaHeartbeatMove(
            hold_pose, duration=1.4, amplitude_rad=0.17, bounces=0.5, stagger_s=0.5, name=kind
        )
    # double_flutter: the mirrored two-bounce flutter — slowed and shrunk from the
    # original 1.2s/0.25rad for the study's calmer presence cue (05.08 user test).
    return AntennaHeartbeatMove(hold_pose, duration=2.4, amplitude_rad=0.18, name=kind)


class BreathingMove(Move):  # type: ignore
    """Breathing move with interpolation to neutral and then continuous breathing patterns."""

    def __init__(
        self,
        interpolation_start_pose: NDArray[np.float32],
        interpolation_start_antennas: Tuple[float, float],
        interpolation_duration: float = 1.0,
    ):
        """Initialize breathing move.

        Args:
            interpolation_start_pose: 4x4 matrix of current head pose to interpolate from
            interpolation_start_antennas: Current antenna positions to interpolate from
            interpolation_duration: Duration of interpolation to neutral (seconds)

        """
        self.interpolation_start_pose = interpolation_start_pose
        self.interpolation_start_antennas = np.array(interpolation_start_antennas)
        self.interpolation_duration = interpolation_duration

        # Neutral positions for breathing base
        self.neutral_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
        self.neutral_antennas = np.array([-0.1745, 0.1745])  # ~10° offset to reduce shaking

        # Breathing parameters
        self.breathing_z_amplitude = 0.005  # 5mm gentle breathing
        self.breathing_frequency = 0.1  # Hz (6 breaths per minute)
        self.antenna_sway_amplitude = np.deg2rad(15)  # 15 degrees
        self.antenna_frequency = 0.5  # Hz (faster antenna sway)

    @property
    def duration(self) -> float:
        """Duration property required by official Move interface."""
        return float("inf")  # Continuous breathing (never ends naturally)

    def evaluate(self, t: float) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        """Evaluate breathing move at time t."""
        if t < self.interpolation_duration:
            # Phase 1: Interpolate to neutral base position
            interpolation_t = t / self.interpolation_duration

            # Interpolate head pose
            head_pose = linear_pose_interpolation(
                self.interpolation_start_pose,
                self.neutral_head_pose,
                interpolation_t,
            )

            # Interpolate antennas
            antennas_interp = (
                1 - interpolation_t
            ) * self.interpolation_start_antennas + interpolation_t * self.neutral_antennas
            antennas = antennas_interp.astype(np.float64)

        else:
            # Phase 2: Breathing patterns from neutral base
            breathing_time = t - self.interpolation_duration

            # Gentle z-axis breathing
            z_offset = self.breathing_z_amplitude * np.sin(2 * np.pi * self.breathing_frequency * breathing_time)
            head_pose = create_head_pose(x=0, y=0, z=z_offset, roll=0, pitch=0, yaw=0, degrees=True, mm=False)

            # Antenna sway (opposite directions)
            antenna_sway = self.antenna_sway_amplitude * np.sin(2 * np.pi * self.antenna_frequency * breathing_time)
            antennas = np.array([antenna_sway, -antenna_sway], dtype=np.float64)

        # Return in official Move interface format: (head_pose, antennas_array, body_yaw)
        return (head_pose, antennas, 0.0)


def clone_full_body_pose(pose: FullBodyPose) -> FullBodyPose:
    """Create a deep copy of a full body pose tuple."""
    head, antennas, body_yaw = pose
    return (head.copy(), (float(antennas[0]), float(antennas[1])), float(body_yaw))


@dataclass
class MovementState:
    """State tracking for the movement system."""

    # Primary move state
    current_move: Move | None = None
    move_start_time: float | None = None
    last_activity_time: float = 0.0

    # Status flags
    last_primary_pose: FullBodyPose | None = None

    def update_activity(self) -> None:
        """Update the last activity time."""
        self.last_activity_time = time.monotonic()


@dataclass
class LoopFrequencyStats:
    """Track rolling loop frequency statistics."""

    mean: float = 0.0
    m2: float = 0.0
    min_freq: float = float("inf")
    count: int = 0
    last_freq: float = 0.0
    potential_freq: float = 0.0

    def reset(self) -> None:
        """Reset accumulators while keeping the last potential frequency."""
        self.mean = 0.0
        self.m2 = 0.0
        self.min_freq = float("inf")
        self.count = 0


class MovementManager:
    """Coordinate sequential moves and robot output at 100 Hz.

    Responsibilities:
    - Own a real-time loop that samples the current primary move (if any) and calls
      `set_target` exactly once per tick.
    - Start an idle `BreathingMove` after `idle_inactivity_delay` when not
      listening and no moves are queued.
    - Expose thread-safe APIs so other threads can enqueue moves or mark activity
      without touching internal state.

    Timing:
    - All elapsed-time calculations rely on `time.monotonic()` through `self._now`
      to avoid wall-clock jumps.
    - The loop attempts 100 Hz

    Concurrency:
    - External threads communicate via `_command_queue` messages.
    """

    # Idle antenna heartbeat cadence: randomized so the cue never feels metronomic.
    # 10-30s per Nicole's presence-cue feedback (03.08); was 60-120s — revisit if
    # the user test finds it distracting.
    HEARTBEAT_INTERVAL_RANGE_S = (10.0, 30.0)

    def __init__(
        self,
        current_robot: ReachyMini,
    ):
        """Initialize movement manager."""
        self.current_robot = current_robot
        self._head_tracking = False

        # Single timing source for durations
        self._now = time.monotonic

        # Movement state
        self.state = MovementState()
        self.state.last_activity_time = self._now()
        neutral_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
        self.state.last_primary_pose = (neutral_pose, (0.0, 0.0), 0.0)

        # Move queue (primary moves)
        self.move_queue: deque[Move] = deque()

        # Configuration
        self.idle_inactivity_delay = 0.3  # seconds
        self.target_frequency = CONTROL_LOOP_FREQUENCY_HZ
        self.target_period = 1.0 / self.target_frequency

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._is_listening = False
        # Speaking pauses tracking; the captured look-at pose anchors queued moves.
        self._is_speaking = False
        self._track_anchor: NDArray[np.float64] | None = None
        self._last_commanded_pose: FullBodyPose = clone_full_body_pose(self.state.last_primary_pose)
        self._listening_antennas: Tuple[float, float] = self._last_commanded_pose[1]
        self._antenna_unfreeze_blend = 1.0
        self._antenna_blend_duration = 0.4  # seconds to blend back after listening
        self._last_listening_blend_time = self._now()
        self._breathing_active = False  # true when breathing move is running or queued
        self._listening_debounce_s = 0.15
        self._last_listening_toggle_time = self._now()
        self._last_set_target_err = 0.0
        self._set_target_err_interval = 1.0  # seconds between error logs
        self._set_target_err_suppressed = 0
        # Baseline pitch trim: composed under every outgoing pose so the whole
        # motion repertoire rides on a tilted resting posture (camera aiming knob).
        self._pitch_trim_pose: NDArray[np.float64] | None = (
            create_head_pose(0, 0, 0, 0, config.HEAD_PITCH_TRIM_DEG, 0, degrees=True)
            if config.HEAD_PITCH_TRIM_DEG
            else None
        )
        # Idle antenna heartbeat: next wall-clock firing time; None = not yet scheduled.
        self._next_heartbeat_time: float | None = None

        # Cross-thread signalling
        self._command_queue: "Queue[Tuple[str, Any]]" = Queue()

        self._shared_state_lock = threading.Lock()
        self._shared_last_activity_time = self.state.last_activity_time
        self._shared_is_listening = self._is_listening
        self._status_lock = threading.Lock()
        self._freq_stats = LoopFrequencyStats()
        self._freq_snapshot = LoopFrequencyStats()

    def queue_move(self, move: Move) -> None:
        """Queue a primary move to run after the currently executing one.

        Thread-safe: the move is enqueued via the worker command queue so the
        control loop remains the sole mutator of movement state.
        """
        self._command_queue.put(("queue_move", move))

    def clear_move_queue(self) -> None:
        """Stop the active move and discard any queued primary moves.

        Thread-safe: executed by the worker thread via the command queue.
        """
        self._command_queue.put(("clear_queue", None))

    def set_moving_state(self, duration: float) -> None:
        """Mark the robot as actively moving for the provided duration.

        Legacy hook used by goto helpers to keep inactivity and breathing logic
        aware of manual motions. Thread-safe via the command queue.
        """
        self._command_queue.put(("set_moving_state", duration))

    def is_idle(self) -> bool:
        """Return True when the robot has been inactive longer than the idle delay."""
        with self._shared_state_lock:
            last_activity = self._shared_last_activity_time
            listening = self._shared_is_listening

        if listening:
            return False

        return self._now() - last_activity >= self.idle_inactivity_delay

    def set_listening(self, listening: bool) -> None:
        """Enable or disable listening mode without touching shared state directly.

        While listening:
        - Antenna positions are frozen at the last commanded values.
        - Blending is reset so that upon unfreezing the antennas return smoothly.
        - Idle breathing is suppressed.

        Thread-safe: the change is posted to the worker command queue.
        """
        with self._shared_state_lock:
            if self._shared_is_listening == listening:
                return
        self._command_queue.put(("set_listening", listening))

    def set_head_tracking(self, enabled: bool) -> None:
        """Start or stop following the user's face; thread-safe via the command queue."""
        self._command_queue.put(("set_head_tracking", enabled))

    def set_speaking(self, speaking: bool) -> None:
        """Pause head tracking while the assistant speaks, resume it afterwards.

        On speaking, the current look-at pose is captured as an anchor so queued
        moves stay on the user; when speaking stops the head is handed back to
        daemon tracking. Thread-safe: posted to the worker command queue.
        """
        self._command_queue.put(("set_speaking", speaking))

    def _poll_signals(self, current_time: float) -> None:
        """Apply queued commands."""
        while True:
            try:
                command, payload = self._command_queue.get_nowait()
            except Empty:
                break
            self._handle_command(command, payload, current_time)

    def _handle_command(self, command: str, payload: Any, current_time: float) -> None:
        """Handle a single cross-thread command."""
        if command == "queue_move":
            if isinstance(payload, Move):
                self.move_queue.append(payload)
                self.state.update_activity()
                duration = getattr(payload, "duration", None)
                if duration is not None:
                    try:
                        duration_str = f"{float(duration):.2f}"
                    except (TypeError, ValueError):
                        duration_str = str(duration)
                else:
                    duration_str = "?"
                logger.debug(
                    "Queued move with duration %ss, queue size: %s",
                    duration_str,
                    len(self.move_queue),
                )
            else:
                logger.warning("Ignored queue_move command with invalid payload: %s", payload)
        elif command == "clear_queue":
            self.move_queue.clear()
            self.state.current_move = None
            self.state.move_start_time = None
            self._breathing_active = False
            logger.info("Cleared move queue and stopped current move")
        elif command == "set_moving_state":
            try:
                duration = float(payload)
            except (TypeError, ValueError):
                logger.warning("Invalid moving state duration: %s", payload)
                return
            self.state.update_activity()
        elif command == "mark_activity":
            self.state.update_activity()
        elif command == "set_listening":
            desired_state = bool(payload)
            now = self._now()
            if now - self._last_listening_toggle_time < self._listening_debounce_s:
                return
            self._last_listening_toggle_time = now

            if self._is_listening == desired_state:
                return

            self._is_listening = desired_state
            self._last_listening_blend_time = now
            if desired_state:
                # Freeze: snapshot current commanded antennas and reset blend
                self._listening_antennas = (
                    float(self._last_commanded_pose[1][0]),
                    float(self._last_commanded_pose[1][1]),
                )
                self._antenna_unfreeze_blend = 0.0
            else:
                # Unfreeze: restart blending from frozen pose
                self._antenna_unfreeze_blend = 0.0
            self.state.update_activity()
        elif command == "set_head_tracking":
            enabled = bool(payload)
            if self._head_tracking == enabled:
                return
            self._head_tracking = enabled
            self._track_anchor = None
            # set_speaking is gated off while disabled, so its state would go stale across a toggle.
            self._is_speaking = False
            try:
                if enabled:
                    self.current_robot.start_head_tracking(weight=1.0)
                else:
                    self.current_robot.stop_head_tracking()
            except Exception as e:
                logger.warning("Head-tracking toggle failed: %s", e)
        elif command == "set_speaking":
            if not self._head_tracking:
                return
            speaking = bool(payload)
            if self._is_speaking == speaking:
                return
            self._is_speaking = speaking
            try:
                if speaking and self.current_robot.get_tracked_face(wait=False).detected:
                    # Pause only once a face is locked, else speech blocks acquisition.
                    self._track_anchor = self.current_robot.get_current_head_pose()
                    self.current_robot.start_head_tracking(weight=0.0)
                elif not speaking:
                    self._track_anchor = None
                    self.current_robot.start_head_tracking(weight=1.0)
            except Exception as e:
                logger.warning("Head-tracking speaking handoff failed: %s", e)
        else:
            logger.warning("Unknown command received by MovementManager: %s", command)

    def _publish_shared_state(self) -> None:
        """Expose idle-related state for external threads."""
        with self._shared_state_lock:
            self._shared_last_activity_time = self.state.last_activity_time
            self._shared_is_listening = self._is_listening

    def _manage_move_queue(self, current_time: float) -> None:
        """Manage the primary move queue (sequential execution)."""
        if self.state.current_move is None or (
            self.state.move_start_time is not None
            and current_time - self.state.move_start_time >= self.state.current_move.duration
        ):
            self.state.current_move = None
            self.state.move_start_time = None

            if self.move_queue:
                self.state.current_move = self.move_queue.popleft()
                self.state.move_start_time = current_time
                # Any real move cancels breathing mode flag
                self._breathing_active = isinstance(self.state.current_move, BreathingMove)
                logger.debug(f"Starting new move, duration: {self.state.current_move.duration}s")

    def _manage_breathing(self, current_time: float) -> None:
        """Manage automatic breathing when idle."""
        if not config.BREATHING_ENABLED:
            return
        if (
            self.state.current_move is None
            and not self.move_queue
            and not self._is_listening
            and not self._breathing_active
        ):
            idle_for = current_time - self.state.last_activity_time
            if idle_for >= self.idle_inactivity_delay:
                try:
                    # These 2 functions return the latest available sensor data from the robot, but don't perform I/O synchronously.
                    # Therefore, we accept calling them inside the control loop.
                    _, current_antennas = self.current_robot.get_current_joint_positions()
                    current_head_pose = self.current_robot.get_current_head_pose()

                    self._breathing_active = True
                    self.state.update_activity()

                    breathing_move = BreathingMove(
                        interpolation_start_pose=current_head_pose,
                        interpolation_start_antennas=current_antennas,
                        interpolation_duration=1.0,
                    )
                    self.move_queue.append(breathing_move)
                    logger.debug("Started breathing after %.1fs of inactivity", idle_for)
                except Exception as e:
                    self._breathing_active = False
                    logger.error("Failed to start breathing: %s", e)

        if isinstance(self.state.current_move, BreathingMove) and self.move_queue:
            self.state.current_move = None
            self.state.move_start_time = None
            self._breathing_active = False
            logger.debug("Stopping breathing due to new move activity")

        if self.state.current_move is not None and not isinstance(self.state.current_move, BreathingMove):
            self._breathing_active = False

    def _get_primary_pose(self, current_time: float) -> FullBodyPose:
        """Get the primary full body pose from current move or neutral."""
        # When a primary move is playing, sample it and cache the resulting pose
        if self.state.current_move is not None and self.state.move_start_time is not None:
            move_time = current_time - self.state.move_start_time
            head, antennas, body_yaw = self.state.current_move.evaluate(move_time)

            if head is None:
                head = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            if antennas is None:
                antennas = np.array([-0.1745, 0.1745])  # ~10° offset
            if body_yaw is None:
                body_yaw = 0.0

            antennas_tuple = (float(antennas[0]), float(antennas[1]))
            head_copy = head.copy()
            primary_full_body_pose = (
                head_copy,
                antennas_tuple,
                float(body_yaw),
            )

            self.state.last_primary_pose = clone_full_body_pose(primary_full_body_pose)
        # Otherwise reuse the last primary pose so we avoid jumps between moves
        elif self.state.last_primary_pose is not None:
            primary_full_body_pose = clone_full_body_pose(self.state.last_primary_pose)
        else:
            neutral_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            primary_full_body_pose = (neutral_head_pose, (0.0, 0.0), 0.0)
            self.state.last_primary_pose = clone_full_body_pose(primary_full_body_pose)

        # Speaking pauses tracking: hold the look-at anchor, overlay emotions on it, dance from neutral.
        if self._track_anchor is not None:
            head_pose, antennas, body_yaw = primary_full_body_pose
            move = self.state.current_move
            if move is None:
                head_pose = self._track_anchor.copy()
            elif isinstance(move, EmotionQueueMove):
                head_pose = compose_world_offset(self._track_anchor, head_pose)
            primary_full_body_pose = (head_pose, antennas, body_yaw)

        return primary_full_body_pose

    def _update_primary_motion(self, current_time: float) -> None:
        """Advance queue state and idle behaviours for this tick."""
        self._manage_move_queue(current_time)
        self._manage_breathing(current_time)

    def _calculate_blended_antennas(self, target_antennas: Tuple[float, float]) -> Tuple[float, float]:
        """Blend target antennas with listening freeze state and update blending."""
        now = self._now()
        listening = self._is_listening
        listening_antennas = self._listening_antennas
        blend = self._antenna_unfreeze_blend
        blend_duration = self._antenna_blend_duration
        last_update = self._last_listening_blend_time
        self._last_listening_blend_time = now

        if listening:
            antennas_cmd = listening_antennas
            new_blend = 0.0
        else:
            dt = max(0.0, now - last_update)
            if blend_duration <= 0:
                new_blend = 1.0
            else:
                new_blend = min(1.0, blend + dt / blend_duration)
            antennas_cmd = (
                listening_antennas[0] * (1.0 - new_blend) + target_antennas[0] * new_blend,
                listening_antennas[1] * (1.0 - new_blend) + target_antennas[1] * new_blend,
            )

        if listening:
            self._antenna_unfreeze_blend = 0.0
        else:
            self._antenna_unfreeze_blend = new_blend
            if new_blend >= 1.0:
                self._listening_antennas = (
                    float(target_antennas[0]),
                    float(target_antennas[1]),
                )

        return antennas_cmd

    def _maybe_queue_antenna_heartbeat(self, now: float) -> None:
        """Queue the idle antenna flutter when due (study 'body double' cue).

        Worker-thread only. Fires every HEARTBEAT_MIN..MAX seconds, but only while
        the robot is otherwise fully idle: any active or queued move, listening
        mode, or breathing postpones the pulse without burning the slot. Disabled
        entirely in the control condition (no robot output there, not even this).
        """
        if not config.ANTENNA_HEARTBEAT_ENABLED or config.CONTROL_MODE:
            return
        if self._next_heartbeat_time is None:
            self._next_heartbeat_time = now + random.uniform(*self.HEARTBEAT_INTERVAL_RANGE_S)
            return
        if now < self._next_heartbeat_time:
            return
        if self.state.current_move is not None or self.move_queue or self._is_listening or self._breathing_active:
            self._next_heartbeat_time = now + 5.0  # busy: retry shortly
            return
        hold_pose = (
            clone_full_body_pose(self.state.last_primary_pose)
            if self.state.last_primary_pose is not None
            else (create_head_pose(0, 0, 0, 0, 0, 0, degrees=True), (-0.1745, 0.1745), 0.0)
        )
        move = make_idle_antenna_move(hold_pose)
        self.move_queue.append(move)
        self._next_heartbeat_time = now + random.uniform(*self.HEARTBEAT_INTERVAL_RANGE_S)
        logger.debug("Antenna heartbeat queued (%s)", move.name)

    def _issue_control_command(
        self, head: NDArray[np.float32], antennas: Tuple[float, float], body_yaw: float
    ) -> None:
        """Send the pose to the robot with throttled error logging."""
        if self._pitch_trim_pose is not None:
            head = compose_world_offset(self._pitch_trim_pose, head)
        try:
            self.current_robot.set_target(head=head, antennas=antennas, body_yaw=body_yaw)
        except Exception as e:
            now = self._now()
            if now - self._last_set_target_err >= self._set_target_err_interval:
                msg = f"Failed to set robot target: {e}"
                if self._set_target_err_suppressed:
                    msg += f" (suppressed {self._set_target_err_suppressed} repeats)"
                    self._set_target_err_suppressed = 0
                logger.error(msg)
                self._last_set_target_err = now
            else:
                self._set_target_err_suppressed += 1
        else:
            with self._status_lock:
                self._last_commanded_pose = clone_full_body_pose((head, antennas, body_yaw))

    def _update_frequency_stats(
        self,
        loop_start: float,
        prev_loop_start: float,
        stats: LoopFrequencyStats,
    ) -> LoopFrequencyStats:
        """Update frequency statistics based on the current loop start time."""
        period = loop_start - prev_loop_start
        if period > 0:
            stats.last_freq = 1.0 / period
            stats.count += 1
            delta = stats.last_freq - stats.mean
            stats.mean += delta / stats.count
            stats.m2 += delta * (stats.last_freq - stats.mean)
            stats.min_freq = min(stats.min_freq, stats.last_freq)
        return stats

    def _schedule_next_tick(self, loop_start: float, stats: LoopFrequencyStats) -> Tuple[float, LoopFrequencyStats]:
        """Compute sleep time to maintain target frequency and update potential freq."""
        computation_time = self._now() - loop_start
        stats.potential_freq = 1.0 / computation_time if computation_time > 0 else float("inf")
        sleep_time = max(0.0, self.target_period - computation_time)
        return sleep_time, stats

    def _record_frequency_snapshot(self, stats: LoopFrequencyStats) -> None:
        """Store a thread-safe snapshot of current frequency statistics."""
        with self._status_lock:
            self._freq_snapshot = LoopFrequencyStats(
                mean=stats.mean,
                m2=stats.m2,
                min_freq=stats.min_freq,
                count=stats.count,
                last_freq=stats.last_freq,
                potential_freq=stats.potential_freq,
            )

    def _maybe_log_frequency(self, loop_count: int, print_interval_loops: int, stats: LoopFrequencyStats) -> None:
        """Emit frequency telemetry when enough loops have elapsed."""
        if loop_count % print_interval_loops != 0 or stats.count == 0:
            return

        variance = stats.m2 / stats.count if stats.count > 0 else 0.0
        lowest = stats.min_freq if stats.min_freq != float("inf") else 0.0
        logger.debug(
            "Loop freq - avg: %.2fHz, variance: %.4f, min: %.2fHz, last: %.2fHz, potential: %.2fHz, target: %.1fHz",
            stats.mean,
            variance,
            lowest,
            stats.last_freq,
            stats.potential_freq,
            self.target_frequency,
        )
        stats.reset()

    def start(self) -> None:
        """Start the worker thread that drives the 100 Hz control loop."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Move worker already running; start() ignored")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.working_loop, daemon=True)
        self._thread.start()
        logger.debug("Move worker started")

    def stop(self, reset_to_neutral: bool = True) -> None:
        """Request the worker thread to stop and wait for it to exit.

        Optionally resets the robot to a neutral position after stopping.
        """
        if self._thread is None or not self._thread.is_alive():
            logger.debug("Move worker not running; stop() ignored")
            return

        if reset_to_neutral:
            logger.info("Stopping movement manager and resetting to neutral position...")
        else:
            logger.info("Stopping movement manager...")

        # Clear any queued moves and stop current move
        self.clear_move_queue()

        # Stop the worker thread first so it doesn't interfere
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        logger.debug("Move worker stopped")

        if self._head_tracking:
            try:
                self.current_robot.stop_head_tracking()
            except Exception as e:
                logger.warning("Failed to stop head tracking: %s", e)

        if not reset_to_neutral:
            return

        # Reset to neutral position using goto_target (same approach as wake_up)
        try:
            neutral_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            neutral_antennas = [-0.1745, 0.1745]  # ~10° offset to reduce shaking
            neutral_body_yaw = 0.0

            # Use goto_target directly on the robot
            self.current_robot.goto_target(
                head=neutral_head_pose,
                antennas=neutral_antennas,
                duration=2.0,
                body_yaw=neutral_body_yaw,
            )

            logger.info("Reset to neutral position completed")

        except Exception as e:
            logger.error(f"Failed to reset to neutral position: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Return a lightweight status snapshot for observability."""
        with self._status_lock:
            pose_snapshot = clone_full_body_pose(self._last_commanded_pose)
            freq_snapshot = LoopFrequencyStats(
                mean=self._freq_snapshot.mean,
                m2=self._freq_snapshot.m2,
                min_freq=self._freq_snapshot.min_freq,
                count=self._freq_snapshot.count,
                last_freq=self._freq_snapshot.last_freq,
                potential_freq=self._freq_snapshot.potential_freq,
            )

        head_matrix = pose_snapshot[0].tolist() if pose_snapshot else None
        antennas = pose_snapshot[1] if pose_snapshot else None
        body_yaw = pose_snapshot[2] if pose_snapshot else None

        return {
            "queue_size": len(self.move_queue),
            "is_listening": self._is_listening,
            "breathing_active": self._breathing_active,
            "last_commanded_pose": {
                "head": head_matrix,
                "antennas": antennas,
                "body_yaw": body_yaw,
            },
            "loop_frequency": {
                "last": freq_snapshot.last_freq,
                "mean": freq_snapshot.mean,
                "min": freq_snapshot.min_freq,
                "potential": freq_snapshot.potential_freq,
                "samples": freq_snapshot.count,
            },
        }

    def working_loop(self) -> None:
        """Run the primary-move control loop with a single set_target() call per tick."""
        logger.debug("Starting enhanced movement control loop (100Hz)")

        loop_count = 0
        prev_loop_start = self._now()
        print_interval_loops = max(1, int(self.target_frequency * 2))
        freq_stats = self._freq_stats

        while not self._stop_event.is_set():
            loop_start = self._now()
            loop_count += 1

            if loop_count > 1:
                freq_stats = self._update_frequency_stats(loop_start, prev_loop_start, freq_stats)
            prev_loop_start = loop_start

            # 1) Poll external commands
            self._poll_signals(loop_start)

            # 1b) Idle antenna heartbeat (no-op unless enabled and due)
            self._maybe_queue_antenna_heartbeat(loop_start)

            # 2) Manage the primary move queue (start new move, end finished move, breathing)
            self._update_primary_motion(loop_start)

            # 3) Build the primary full-body pose for this tick
            head, antennas, body_yaw = self._get_primary_pose(loop_start)

            # 4) Apply listening antenna freeze or blend-back
            antennas_cmd = self._calculate_blended_antennas(antennas)

            # 5) Single set_target call - the only control point
            self._issue_control_command(head, antennas_cmd, body_yaw)

            # 6) Adaptive sleep to align to next tick, then publish shared state
            sleep_time, freq_stats = self._schedule_next_tick(loop_start, freq_stats)
            self._publish_shared_state()
            self._record_frequency_snapshot(freq_stats)

            # 7) Periodic telemetry on loop frequency
            self._maybe_log_frequency(loop_count, print_interval_loops, freq_stats)

            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.debug("Movement control loop stopped")
