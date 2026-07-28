# Fixes potentially needed later

---

### Log

- [28.07.2026] Conversational breathing: with breathing now off by default, consider re-enabling it only WHILE the student is speaking (+ ~10s tail after the interaction ends) — liveliness during interaction, stillness during focused work. Would hook the mic/VAD state into moves' breathing manager instead of the current inactivity timer.
- [28.07.2026] Lab-machine test paused (user may run study on own laptop); revisit only if the laptop route fails.
- [21.07.2026] EmotiEffLib can also do engagement detection, benchmark against Del Duchetto
- [14.07.2026] If local models are too taxing, figure out if there is a way to host them on Imperial's cluster and somehow make them communicate via server
- [10.07.2026] Maybe make user ingested text durable context leveraging remember, especially if long sessions seem to time out / run out of context after a while
- [10.07.2026] Have to put in more safeguards such that it asks the student questions rather than just giving away the correct solution. Maybe will be resolved if the ingestion layer frames this context correctly
- [08.07.2026] Duplicated audio responses by Reachy occasionally upon start up and also once negative emotions were detected, most likely deeper in the stack / not cuased by any of my added modules
- [08.07.2026] gi (PyGObject) ModuleNotFoundError blocking full pytest tests/ collection. Pre-existing environment gap in reachy_mini's GStreamer bindings, unrelated to emotion recognition, needs a system-level (Homebrew) install to fix
- [08.07.2026] intervention message text will start as a fixed string rather than profile-configurable like greeting.txt
- [08.07.2026] detector_backend="opencv" was picked for speed over accuracy, unbenchmarked against alternatives.