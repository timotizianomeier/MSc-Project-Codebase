# app.log index

---

### Log

- [21.07.2026] **emotion-detector-arm1-opencv-webcam** - Frame-dump mechanism verified (23 polls = 23 files, 0 failures, cadence stretches 5s→~7.8s with classification time). Webcam + ring light: detection 20/20 on visible faces (the 3 "noface" were a deliberately blocked lens, not misses — checked the actual frames). Classification is the weak link: deliberate sad pout → angry, 15/20 mild expressions → neutral. Implications: sad↔angry confusion is harmless to negative_share (same sign), but mild-negative→neutral suppresses interventions — EmotiEffLib swap (step 2) is the main event; retinaface arm still worth running under WORSE lighting for robustness margin. Robot-camera arms pending. Frames: `frames/2026-07-21_opencv-webcam/`. Log: `2026-07-21_1447_emotion-detector-arm1-opencv-webcam.log`
- [16.07.2026] **wireless-robot-first-session** — setup: wireless Reachy via network mode (`reachy-mini.local`, WebRTC media), hosted backend, engagement service up, desktop app closed — outcome: first engagement intervention from the ROBOT's head camera (14:23:58) + text-context drop confirmed working; but progressive WiFi degradation killed the session: control drops from 14:22 (antenna stutter), robot-bound audio inaudible by ~14:26 (deltas streamed but never played), video gone by 14:27. Not code, not image quality — link. Next: measure ping/loss, test near router, decide wireless-vs-Lite for study room.
- [15.07.2026] **three-feature-e2e-attempt1** - Engagement detection working well, might have to lower threshold. Emotion recognition never triggered. Text input working well. Think about introducing hotkey instead, assess feasability. Idle motions might have to be reduced. Log: `2026-07-15_1618_three-feature-e2e-attempt2.log`
