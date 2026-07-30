#!/bin/bash
# macOS marks .venv with UF_HIDDEN which causes Python to skip .pth files.
# Clear it before every launch.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
chflags -R nohidden "$SCRIPT_DIR/.venv"

# Auto-archive the previous session's app.log before it gets overwritten, so a
# missed manual copy can never lose a session (the named copies in logs/ stay
# the curated set; these autosaves are the safety net). Timestamp = the old
# log's mtime, i.e. when that session ended.
APP_LOG="$SCRIPT_DIR/app.log"
if [ -f "$APP_LOG" ]; then
    mv "$APP_LOG" "$SCRIPT_DIR/../logs/app_autosave_$(date -r "$(stat -f %m "$APP_LOG")" +%Y-%m-%d_%H%M).log"
fi

"$SCRIPT_DIR/.venv/bin/reachy-mini-conversation-app" "$@"
