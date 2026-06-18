# Plan: ADHD Study Assistant — Reachy Mini App

> Per AGENTS.md conventions: write this first, ask questions, wait for answers before coding.
> This file lives in `MSc-Project-Codebase/` temporarily; it will move into the scaffolded app dir once created.

---

## My Understanding of the Goal

Build a Python app on top of the `reachy_mini_conversation_app` template that:

1. Runs Pomodoro-style study sessions on Reachy Mini (Wireless, streamed mode).
2. Monitors the student via webcam using three perception modules:
   - **EmotionRecognizer** — dominant emotion from webcam frames (prototype: DeepFace, which you've already explored)
   - **EngagementDetector** — inattention/distraction signals (gaze, head pose, or activity level)
   - **ContextProvider** — what the student is working on (e.g. text from an assignment PDF)
3. When a student is distracted or stuck, triggers context-aware LLM dialogue via the robot.
4. Logs everything to a timestamped CSV for post-study analysis.
5. Supports a feature-flag mechanism so you can run a "no-intervention" control condition by flipping flags to Null implementations — **this is load-bearing for your study design**.

Hardware: Reachy Mini Wireless, laptop compute (streamed mode). Webcam is the robot's own camera (accessed via the SDK/network stream in streamed mode).

---

## Technical Approach (5 steps)

### Step 1 — Scaffold with conversation template

```bash
reachy-mini-app-assistant create --template conversation <app_name> <path>
```

This forks `reachy_mini_conversation_app` with a locked profile. It gives us:
- A working `pyproject.toml` / entry point
- The full audio pipeline (STT/TTS via fastrtc)
- LLM loop with tool-calling
- Robot motion management
- Gradio web UI

The `--publish` flag requires `hf auth login` first. **See Q2 below.**

### Step 2 — Confirm baseline works

Install with `uv sync` (or `pip install -e .`), configure `.env`, and run the unmodified app end-to-end against the robot. Goal: hear a spoken response back from the robot. No code changes — just validate the scaffold.

### Step 3 — Trim to minimum

The conversation app ships with three AI backends, vision/head-tracking, MCP client, memory, and a full personality profile system. For our baseline we need only:

| Keep | Remove |
|---|---|
| One AI backend (HuggingFace Realtime — free, low-latency) | Gemini Live backend |
| Gradio UI (study interface will live here) | OpenAI Realtime backend |
| `moves.py` motion system | Vision / head-tracking (`vision/` subpackage) |
| Idle breathing behavior | MCP client |
| Single hard-coded ADHD-TA system prompt | Memory / `remember`/`forget` tools |
| `config.py` (simplified) | All personality profiles except one |
| `camera_worker.py` (robot camera feed, needed for perception) | Unused LLM tools (dance, external tools) |

**Important:** trimming happens by deletion + import cleanup, not by disabling with flags. The result should still pass its own test suite (where tests don't depend on removed modules).

**Boilerplate note:** `main.py`, `config.py`, `moves.py`, and `conversation_handler.py` / `base_realtime.py` are the files you'll need to read closely — they're the load-bearing core you'll build on. Everything else post-trim is scaffolding.

### Step 4 — Module interfaces + Null implementations

Create `src/<app_name>/perception/` with three files. Each module follows the same pattern:

```
perception/
├── __init__.py
├── emotion_recognizer.py    # Protocol + NullEmotionRecognizer
├── engagement_detector.py   # Protocol + NullEngagementDetector
└── context_provider.py      # Protocol + NullContextProvider
```

**Design choice: `Protocol` (structural subtyping) over `ABC`.**
Rationale: you don't need inheritance — you need duck-typing so that real implementations (DeepFace, etc.) and Null stubs are interchangeable without a shared base class. `typing.Protocol` is the right tool here; it also plays better with type checkers.

Proposed interfaces (draft — see Q5):

```python
# EmotionRecognizer
def get_emotion(self) -> str | None:
    """Return dominant emotion label ('happy', 'frustrated', ...) or None if no face."""

# EngagementDetector
def get_engagement(self) -> float | None:
    """Return engagement score in [0.0, 1.0], or None if signal unavailable."""

def is_inattentive(self) -> bool:
    """Convenience: True if score is below threshold."""

# ContextProvider
def get_context(self) -> str:
    """Return current student context (e.g. extracted PDF text snippet, or '' if none)."""
```

**Null implementations** return fixed safe values: `"neutral"`, `1.0`, `False`, `""`. They don't call any model. Swapping in a Null makes a module a no-op — exactly what the control condition needs.

**Feature-flag mechanism:** A `StudyConfig` dataclass (or simple dict loaded from `.env` / `study_config.toml`) with fields like:

```python
@dataclass
class StudyConfig:
    condition: str                  # e.g. "intervention" | "control"
    emotion_backend: str            # "null" | "deepface" | ...
    engagement_backend: str         # "null" | "mediapipe" | ...
    context_backend: str            # "null" | "pdf" | ...
    intervention_enabled: bool      # master switch for LLM interventions
```

A factory function `build_perception(config: StudyConfig) -> tuple[EmotionRecognizer, EngagementDetector, ContextProvider]` resolves the string keys to implementations. To run the control condition: set all backends to `"null"` and `intervention_enabled = False`.

**Why this design matters for your study:** the condition and backend flags will be written into every CSV log row, so your analysis code can filter by condition without needing metadata from outside the log file itself.

### Step 5 — Event logging

A single `EventLogger` class that opens a CSV on session start and appends rows synchronously (thread-safe with a lock). No external dependencies beyond stdlib.

**Proposed schema:**

| Column | Type | Example | Notes |
|---|---|---|---|
| `timestamp_iso` | str | `2026-06-18T14:03:22.412Z` | UTC, ISO 8601 |
| `session_id` | str | `s001_p03` | Set at session start |
| `condition` | str | `intervention` | From StudyConfig |
| `event_type` | str | `inattention_detected` | See vocabulary below |
| `module` | str | `EngagementDetector` | Which module emitted this |
| `value` | str | `0.23` | Free-form value (score, label, text excerpt) |
| `metadata` | str | `{}` | JSON blob — use for anything extensible |

**Proposed event_type vocabulary (initial):**

```
session_start, session_end
pomodoro_start, pomodoro_end
break_start, break_end
emotion_detected          # periodic sample from EmotionRecognizer
engagement_sampled        # periodic sample from EngagementDetector
inattention_detected      # threshold crossed → triggers intervention pipeline
intervention_triggered    # LLM decided to speak
student_spoke             # STT transcript available
robot_spoke               # TTS output emitted
context_loaded            # ContextProvider delivered non-empty context
```

The schema is simple but the `metadata` JSON column makes it extensible — if you later want to add gaze direction, utterance length, etc., you add it to `metadata` without changing the CSV header.

---

## Files You Should Read Closely (before building on them)

These are the core engine. After trimming, everything else is wiring on top of them.

| File | Why it matters |
|---|---|
| `src/<app>/conversation_handler.py` | This is where LLM responses are processed. Your inattention decision logic will hook in here. |
| `src/<app>/moves.py` | Real-time motion control (60–100 Hz worker thread). You'll call into this when the robot reacts to detected states. |
| `src/<app>/base_realtime.py` | Base class for AI backend handlers. Understand the tool-calling flow before writing intervention tools. |
| `src/<app>/config.py` | All configuration loading. You'll extend this with your StudyConfig. |

---

## Clarifying Questions

**Please fill in the `ANSWER:` fields below, then I'll start coding.**

---

### Q1: App name and location

I'll run:
```
reachy-mini-app-assistant create --template conversation <app_name> MSc-Project-Codebase/
```

This creates `MSc-Project-Codebase/<app_name>/`. Snake_case, becomes the Python package name.

What should `<app_name>` be?

Options:
- `adhd_focus_assistant`
- `focus_assistant`
- `study_assistant`
- Something else?

```
ANSWER: study_assistant_v1
```

---

### Q2: Hugging Face publish?

`--publish` creates a Git repo on HF Spaces immediately. Requires `hf auth login` first (you'd run that in your terminal).

Pros of `--publish`: free version control from day one, required if you ever want to demo on HF.
Cons: public by default (use `--private` to make it private), requires HF login.

For a university study app, I'd lean toward `--publish --private`. But your call.

```
ANSWER (publish / local-only / publish-private): local-only — keep it inside the existing MSc-Project-Codebase GitHub repo
(https://github.com/timotizianomeier/MSc-Project-Codebase.git), no HF publish. Before
running the scaffold command, confirm whether it initialises its own git repo, and if
so, run it without git init (or scaffold outside the repo and move the files in, then
commit normally) to avoid a nested-git conflict.
```

---

### Q3: Which AI backend to keep after trimming?

The conversation app supports three:

| Backend | Cost | Latency | Notes |
|---|---|---|---|
| **HuggingFace Realtime** | Free (via HF token) | ~400 ms | Best for no-API-key setup |
| **OpenAI Realtime** | ~$0.06/min audio | ~200 ms | Lowest latency, needs key |
| **Gemini Live** | Free tier available | ~300 ms | Google account needed |

For the study: you'll be running sessions in a lab, so cost and latency both matter. I'd recommend keeping HuggingFace as the primary and removing the other two — but if you already have an OpenAI key, keeping OpenAI is worth considering.

```
ANSWER (hf / openai / gemini / keep-all): hf (maybe I'll add in others later again)
```

---

### Q4: Study condition names

The feature-flag system needs to know what conditions your study has. Based on your brief I'm guessing two or three:

- `intervention` — full system: emotion + engagement detection, LLM interventions active
- `control` (or `no_intervention`) — Null modules, robot present but doesn't intervene
- (Optional) `robot_absent` — no robot at all (pure passive condition)?

What are the exact condition names for your ethics-approved study design? intervention and control. We are still figuring out what exactly will be switched on and off under the control condition, but for now, let's indeed assume null modules, robot present but doesn't intervene

```
ANSWER: 
```

---

### Q5: Event schema and analysis questions

The event log will be the primary data source for your study analysis. To make sure the schema fits, it helps to know your analysis questions. Based on your brief I'm guessing:

1. Did intervention reduce inattention frequency over the session?
2. Does emotion correlate with inattention events?
3. How long after an inattention detection does the student re-engage?

Are these right? Are there other outcomes you'll measure? (This shapes whether we need, e.g., a `re_engagement_detected` event, or whether we track pomodoro cycle number, etc.)

```
ANSWER: 1. Can a context-aware social robot help students with self-reported ADHD improve self-assessed productivity?
2. Can a context-aware social robot help students with self-reported ADHD improve
objectively measured productivity?
3. Is the potential improvement in productivity, both self-assessed and objectively measured,
larger for students with ADHD than a control group without?
```

---

### Q6: Feature-flag loading mechanism

How should `StudyConfig` be set before a session? Options:

A. **`.env` file** — edit once before the session, loaded at startup. Simple, matches what the conversation app already does.
B. **Gradio UI** — researcher selects condition in the web UI before starting. More ergonomic for lab use, shows condition clearly.
C. **CLI argument** — `reachy-mini-conversation-app --condition intervention`. Clean but requires terminal access during study.

For a lab study where a researcher switches conditions between participants, I'd lean toward **B** (Gradio UI radio button) — it's hard to forget and visible during the session.

```
ANSWER (env / gradio / cli / combination): Let's start with env and we might switch later to Gradio UI.
```

---

### Q7: Webcam access

You said "robot's own camera" for perception. In streamed mode, the robot camera arrives via WebRTC/GStreamer as a network stream. The SDK provides `camera_worker.py` which already handles this.

One potential wrinkle: the conversation app's camera worker feeds frames into the LLM's vision tool (for SmolVLM2), not into a separate perception loop. We'll need a second subscriber to the same frame stream.

Is this confirmed — you want to use **only** the robot's camera (not the student's laptop webcam) for engagement/emotion detection? (Robot camera points at the student from the desk, so this makes sense, just confirming.)

```
ANSWER (robot-cam-only / laptop-webcam / both): Initially, we will us the robot cam only and maybe add the laptop webcam later
```

---

## Status (updated 2026-06-18)

### ✅ Step 1 — Scaffold complete
App: `MSc-Project-Codebase/study_assistant_v1/`. Template: conversation (develop branch, OpenAI Realtime only — HuggingFace dropped upstream). Nested `.git` removed; files track under `MSc-Project-Codebase` git repo.

### ✅ Step 2 — Baseline confirmed
`python3.11 src/study_assistant_v1/main.py --gradio` → full spoken conversation. SSL fix applied (Install Certificates.command). reachy-mini upgraded to 1.8.3 to match daemon.

### ✅ Step 3 — Trimmed
Removed `tools/dance.py`, `tools/stop_dance.py`. Updated locked profile: `instructions.txt` (ADHD TA persona), `tools.txt` (play_emotion, stop_emotion, move_head, sweep_look). `moves.py`: control loop 100Hz → 20Hz (WiFi-safe).

### ✅ Step 4 — Perception interfaces
`src/study_assistant_v1/perception/`: EmotionRecognizer, EngagementDetector, ContextProvider Protocols + Null stubs. `StudyConfig` dataclass + `load_study_config_from_env()` + `build_perception()`. Feature flags in `.env`. Consistency check enforces all-Null backends for control condition.

### ✅ Step 5 — Event logging
`src/study_assistant_v1/event_logger.py`. CSV: `timestamp_iso, session_id, participant_id, condition, event_type, module, value, metadata`. Writes to `logs/<participant_id>_<timestamp>.csv`. Thread-safe, flushes every row.

### Remaining wiring (next session)
- Wire `EventLogger` + `StudyConfig` into `main.py` lifecycle
- Implement real perception backends (DeepFace, MediaPipe, PDF parser)
- Build Pomodoro timer and inattention → intervention decision loop
