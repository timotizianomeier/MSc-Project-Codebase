# Key design decisions

---

### Log

- [08.07.2026] Poll task started in start_up(), not _run_realtime_session() — survives reconnects, which are routine here (profile/voice switches), not rare.
- [08.07.2026] Sampling continues through a disconnected gap, only the send is gated on connection state.
- [07.07.2026] EmotionMonitor as a dependency-free pure-logic class vs. bundling classification in for testability (mirrors idle_policy.py's split).
- [07.07.2026] should_intervene() as a pure query + separate mark_intervened(), instead of a self-mutating predicate, avoids the "decided-but-didn't-act" bug class (precedent: ConversationHandler.last_idle_behavior_time).
- [07.07.2026] Strict > threshold semantics (the <= fix), matching the paper's literal "exceed."
- [07.07.2026] Two separate 60s cooldown constants (interaction vs. intervention), conceptually distinct even though numerically equal today.
- [07.07.2026] deepface as an opt-in pyproject.toml extra, not a core dependency or requirements.txt — keeps TensorFlow out of the base install.
- [07.07.2026] The lazy import deepface inside the function, a deliberate, justified exception to AGENTS.md's "imports at module top."
- [07.07.2026] Fake-module (sys.modules injection) test strategy for the classifier, the only approach that both protects my logic and runs in CI without the heavy dependency installed.