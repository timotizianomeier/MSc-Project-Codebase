# Fixes potentially needed later

---

### Log

- [08.07.2026] Duplicated audio responses by Reachy occasionally upon start up and also once negative emotions were detected, most likely deeper in the stack / not cuased by any of my added modules
- [08.07.2026] gi (PyGObject) ModuleNotFoundError blocking full pytest tests/ collection. Pre-existing environment gap in reachy_mini's GStreamer bindings, unrelated to emotion recognition, needs a system-level (Homebrew) install to fix
- [08.07.2026] intervention message text will start as a fixed string rather than profile-configurable like greeting.txt
- [08.07.2026] detector_backend="opencv" was picked for speed over accuracy, unbenchmarked against alternatives.