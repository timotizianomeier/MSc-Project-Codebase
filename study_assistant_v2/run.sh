#!/bin/bash
# macOS marks .venv with UF_HIDDEN which causes Python to skip .pth files.
# Clear it before every launch.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
chflags -R nohidden "$SCRIPT_DIR/.venv"
"$SCRIPT_DIR/.venv/bin/study-assistant-v2" "$@"
