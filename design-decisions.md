# Key design decisions

---

### Log

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
- [14.07.2026] **All-MLX, corrected** (`--llm_backend mlx-lm --model_name mlx-community/gemma-4-e2b-it-4bit`, 3.55GB): pending.
