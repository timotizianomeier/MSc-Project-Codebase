# Key design decisions

---

### Log

- [14.07.2026] InterventionMonitor as a generic ABC (`Generic[SampleValueT]`) with a `_signal_active()` hook; EmotionMonitor and EngagementMonitor are now thin subclasses — template-method refactor keeps the window/cooldown/response-done gating in one place instead of duplicating it per signal.
- [14.07.2026] Eviction rule established: requirement expressed in *time* → timestamp-filtered window (engagement scores, 30s average); requirement expressed in *count* → `deque(maxlen)` (engagement frames — the model needs exactly 10, so count is the invariant, not age).
- [14.07.2026] Two-cadence engagement poll in a single task: frame capture every 0.5s, scoring every 10th tick (~5s) — one tick counter instead of two tasks, and the 0.5s spacing keeps frame timing consistent (temporal spacing affects the model's scores).
- [14.07.2026] Engagement scoring as a localhost HTTP service (own py3.11/TF2.15 venv, port 8100) rather than in-process — the app venv is py3.12, for which tensorflow-macos has no wheels; the HTTP layer is provably transparent (score diff 0.00e+00 vs direct predict). TL;DR given that Del Duchetto's repo was for ROS, this step is needed to be able to lev erage their proven detection weights.
- [14.07.2026] Engagement threshold 0.93 adopted from Lalwani et al. as a provisional ClassVar — own-webcam attending scores hovered ~0.89, so a local calibration check is planned before the study.
- [14.07.2026] Engagement monitoring starts only after the realtime session is allocated (start_up ordering) - accepted consequence: no monitoring while the backend is down.
- [08.07.2026] Added tf-keras explicitly to the emotion extra as deepface's own packaging doesn't declare it, but its retinaface backend needs it under TensorFlow 2.16+'s Keras-3 default, and it's imported eagerly regardless of which detector backend is actually configured.
- [08.07.2026] Poll task started in start_up(), not _run_realtime_session() — survives reconnects, which are routine here (profile/voice switches), not rare.
- [08.07.2026] Sampling continues through a disconnected gap, only the send is gated on connection state.
- [07.07.2026] EmotionMonitor as a dependency-free pure-logic class vs. bundling classification in for testability (mirrors idle_policy.py's split).
- [07.07.2026] should_intervene() as a pure query + separate mark_intervened(), instead of a self-mutating predicate, avoids the "decided-but-didn't-act" bug class (precedent: ConversationHandler.last_idle_behavior_time).
- [07.07.2026] Strict > threshold semantics (the <= fix), matching the paper's literal "exceed."
- [07.07.2026] Two separate 60s cooldown constants (interaction vs. intervention), conceptually distinct even though numerically equal today.
- [07.07.2026] deepface as an opt-in pyproject.toml extra, not a core dependency or requirements.txt — keeps TensorFlow out of the base install.
- [07.07.2026] The lazy import deepface inside the function, a deliberate, justified exception to AGENTS.md's "imports at module top."
- [07.07.2026] Fake-module (sys.modules injection) test strategy for the classifier, the only approach that both protects my logic and runs in CI without the heavy dependency installed.

### Choices affecting Reachy's personality

- Persona specified in the adhd_study_assistant profile
- EMOTION_INTERVENTION_PROMPT in prompts.py
- The wrapped framing around the text context a user feeds
- ENGAGEMENT_INTERVENTION_PROMPT in prompts.py

### Local LLM setups tested

- [14.07.2026] **llama-server + gemma-4-E4B Q4 (5.0GB) via s2s responses-api**: worked, but 23 t/s and memory pressure on the 16GB Air → intermittent PortAudio -9986 crashes at audio-stream open.
- [14.07.2026] **llama-server + gemma-4-E2B QAT q4_0 (3.35GB, ctx 8192)**: LLM solved — 43-48 t/s, 0.7-1.7s/turn, no more -9986. New bottleneck: Qwen3-TTS first-audio erratic (1.5-14s), suspected llama.cpp↔MLX Metal contention; end-to-end 6-26s/turn.
- [14.07.2026] **All-MLX consolidation, first attempt (invalid test)**: `--local_mac_optimal_settings` alone doesn't forward `--model_name` to the MLX handler (parser binds it by pre-parsing `--llm_backend`, which the preset sets too late) → silently loaded default Qwen3-4B bf16 (8GB) → swap, thermal throttling, Metal OOM. Lesson: always pass `--llm_backend mlx-lm` explicitly and verify the loaded model name in the startup log.
- [15.07.2026] **All-MLX, corrected** (`--llm_backend mlx-lm --model_name mlx-community/gemma-4-e2b-it-4bit`, 3.55GB): **VERIFIED — the chosen local stack.** 1.3–3.3s speech-to-speech per turn, TTS first-audio constant ~0.3s, TTS RTF ~1.4 (faster than realtime) — confirms the Metal-contention diagnosis; llama-server retired. Prerequisite: s2s pins mlx-lm==0.31.1, which predates the gemma4 architecture → patch the uv tool venv with `uv pip install --python ~/.local/share/uv/tools/speech-to-speech/bin/python "mlx-lm==0.31.3"` (any later `uv tool install/upgrade speech-to-speech` silently reverts this).
- [15.07.2026] **Answer brevity is prompt, not model**: s2s's default `--init_chat_prompt` caps responses at "less than 20 words"; overriding it produced solid multi-sentence answers from the 2B model (incl. context-carrying follow-ups) at 1.7–2.8s/turn. In production the app's persona instructions replace this default, so only standalone testing is affected.
- [15.07.2026] **Turn-taking limits of the s2s engine (standalone `--mode local`)**: silence-based endpointing (`--live_transcription_min_silence_ms`, default 500ms; raised to 900ms) still clips hesitant/pausing speakers — relevant to the study whichever backend is used, since the hosted backend runs the same engine. Separately, observed a likely engine bug: turns reopened 13s after being answered (outside both `--speculative_reopen_ms 2500` and the 7s unanswered cap), gluing new speech onto old turns and re-answering them. Not tunable; decision: stop tuning standalone mode, evaluate turn-taking on the real path (server mode + app over realtime websocket) and report upstream if it reproduces there.
