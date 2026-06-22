---
title: Conversation Test
emoji: 🤖
colorFrom: purple
colorTo: gray
sdk: static
pinned: false
tags:
  - reachy_mini
  - reachy_mini_python_app
---

# Conversation Test

Forked from the Reachy Mini conversation app.

Use the `src/study_assistant_v2/profiles/_study_assistant_v2_locked_profile` folder to customize your own app from this template:
- Edit instructions `_study_assistant_v2_locked_profile/instructions.txt`
- Edit available tools in `_study_assistant_v2_locked_profile/tools.txt`
- You can create your own tools in `_study_assistant_v2_locked_profile` by subclassing the `Tool` class.

Do not forget to customize:
- this `README.md` file
- the `index.html` file (Hugging Face Spaces landing page)
- the `src/study_assistant_v2/static/index.html` (the web app parameters page)

The original README from the conversation app is available in `README_OLD.md`.

## Running the app

Activate your virtual environment, then use the `run.sh` wrapper instead of the entry point directly:

```bash
./run.sh --no-camera --gradio
```

This is equivalent to `study-assistant-v2 --no-camera --gradio` but works around a macOS issue where the `.venv` directory is marked hidden (`UF_HIDDEN`), causing Python to silently skip `.pth` files and fail to find installed packages. The wrapper clears that flag before every launch.

The Gradio web UI will be available at http://127.0.0.1:7860 once the app has started.

For all available CLI options, see the [original README](README_OLD.md#cli-options).