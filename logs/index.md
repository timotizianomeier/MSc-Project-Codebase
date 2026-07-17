# app.log index

---

### Log

- [15.07.2026] **three-feature-e2e-attempt1** - Engagement detection working well, might have to lower threshold. Emotion recognition never triggered. Text input working well. Think about introducing hotkey instead, assess feasability. Idle motions might have to be reduced. Log: `2026-07-15_1618_three-feature-e2e-attempt2.log`
- 2026-07-16 14:22 **wireless-robot-first-session** — setup: wireless Reachy via network mode (`reachy-mini.local`, WebRTC media), hosted backend, engagement service up, desktop app closed — outcome: first engagement intervention from the ROBOT's head camera (14:23:58) + text-context drop confirmed working; but progressive WiFi degradation killed the session: control drops from 14:22 (antenna stutter), robot-bound audio inaudible by ~14:26 (deltas streamed but never played), video gone by 14:27. Not code, not image quality — link. Next: measure ping/loss, test near router, decide wireless-vs-Lite for study room.
