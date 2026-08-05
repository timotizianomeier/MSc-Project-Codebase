"""Extract tidy CSVs from a study-session app.log (treatment or control).

Both conditions emit byte-identical poll/event line formats (the control runner
imports the app's monitors and copies its format strings), so this one parser
serves every session. Sessions must be run with --debug: the engagement/emotion
poll lines and speech-activity lines are DEBUG level.

Usage:
    python parse_app_log.py path/to/app.log [-o OUTDIR]

Writes into OUTDIR (default: <log_dir>/<log_stem>_csv/):
    engagement.csv  one row per engagement score (every ~5s)
    emotion.csv     one row per emotion poll, incl. no-face rows (every ~5s)
    speech.csv      user/robot speech segments (start, end, duration)
    events.csv      discrete events: session start/end, interventions,
                    counterfactuals, context submits (with text, newlines
                    escaped as \n; 'context_forwarded' = delivery confirmation,
                    the only context marker in logs from before 05.08),
                    transcripts, ws drops
    session.json    session metadata + row counts (parse sanity summary)

Times: `timestamp` is the wall-clock log stamp; `t_session_s` is seconds since
the SESSION START banner (negative = before Start, empty = no banner in log).
Robot speech segments are contiguous runs of assistant audio deltas (gap-split
at ROBOT_SPEECH_GAP_S) — they measure audible streaming time; playback lags by
the ~0.2s preroll, which is negligible for timeline plots.
"""

from __future__ import annotations

import ast
import csv
import json
import re
import argparse
from pathlib import Path
from datetime import datetime


# Split a robot speech segment when consecutive audio deltas are further apart
# than this. Observed intra-response stalls reach ~1.25s (tunnel burstiness),
# inter-sentence gaps ~3s — 2.5s keeps stalls glued while separating responses.
ROBOT_SPEECH_GAP_S = 2.5

_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) "
    r"(?P<level>[A-Z]+) (?P<logger>[\w.]+):(?P<lineno>\d+) \| (?P<msg>.*)$"
)

_ENGAGEMENT_RE = re.compile(
    r"Engagement poll: score=(?P<score>[\d.]+) average=(?P<average>[\d.]+|n/a) "
    r"\(need<(?P<threshold>[\d.]+)\) response_done=(?P<response_done>\w+) "
    r"interaction_gap=(?P<interaction_gap>[\d.]+)s \(need>(?P<interaction_need>[\d.]+)s\) "
    r"intervention_gap=(?P<intervention_gap>[\d.]+s|never) \(need>(?P<intervention_need>[\d.]+)s\)"
)

_EMOTION_RE = re.compile(
    r"Emotion poll: emotion=(?P<emotion>\w+) scores=(?P<scores>\{.*?\}|None) "
    r"negative_share=(?P<negative_share>[\d.]+) \(need>(?P<threshold>[\d.]+)\) "
    r"response_done=(?P<response_done>\w+) "
    r"interaction_gap=(?P<interaction_gap>[\d.]+)s \(need>(?P<interaction_need>[\d.]+)s\) "
    r"intervention_gap=(?P<intervention_gap>[\d.]+s|never) \(need>(?P<intervention_need>[\d.]+)s\)"
)

_SESSION_START_RE = re.compile(
    r"SESSION START — duration (?P<minutes>[\d.]+) min(?P<control> \(CONTROL condition\))?"
)

_COUNTERFACTUAL_RE = re.compile(
    r"CONTROL: would have sent (?P<kind>engagement|emotion) intervention "
    r"\((?:average|negative_share)=(?P<value>[\d.]+)\)"
)

_TRANSCRIPT_RE = re.compile(r"role=(?P<role>user|assistant) content=(?P<text>.*)$")

_ACTIVITY_RE = re.compile(r"last activity time updated to [\d.]+ \((?P<reason>\w+)\)")


def _parse_ts(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S,%f")


def parse_log(log_path: Path, out_dir: Path) -> dict[str, object]:
    """Parse one app.log into CSVs; return the session summary that was written."""
    engagement_rows: list[dict[str, object]] = []
    emotion_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    speech_rows: list[dict[str, object]] = []

    session_start: datetime | None = None
    session_end: datetime | None = None
    condition: str | None = None
    duration_minutes: float | None = None
    first_ts: datetime | None = None
    last_ts: datetime | None = None

    user_speech_open: datetime | None = None
    robot_run_start: datetime | None = None
    robot_run_last: datetime | None = None

    def t_session(ts: datetime) -> float | str:
        return round((ts - session_start).total_seconds(), 3) if session_start else ""

    def add_event(ts: datetime, event_type: str, detail: str = "", value: object = "") -> None:
        event_rows.append(
            {
                "timestamp": ts.isoformat(sep=" ", timespec="milliseconds"),
                "t_session_s": t_session(ts),
                "event_type": event_type,
                "value": value,
                "detail": detail,
            }
        )

    def close_robot_run() -> None:
        nonlocal robot_run_start, robot_run_last
        if robot_run_start is not None and robot_run_last is not None:
            speech_rows.append(
                {
                    "actor": "robot",
                    "t_start": robot_run_start,
                    "t_end": robot_run_last,
                }
            )
        robot_run_start = None
        robot_run_last = None

    with log_path.open(encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            m = _LINE_RE.match(raw_line.rstrip("\n"))
            if m is None:
                continue  # continuation lines of multi-line transcripts etc.
            ts = _parse_ts(m.group("ts"))
            msg = m.group("msg")
            first_ts = first_ts or ts
            last_ts = ts

            eng = _ENGAGEMENT_RE.search(msg)
            if eng is not None:
                gap = eng.group("intervention_gap")
                engagement_rows.append(
                    {
                        "timestamp": ts.isoformat(sep=" ", timespec="milliseconds"),
                        "t_session_s": t_session(ts),
                        "score": float(eng.group("score")),
                        "average": "" if eng.group("average") == "n/a" else float(eng.group("average")),
                        "threshold": float(eng.group("threshold")),
                        "response_done": eng.group("response_done") == "True",
                        "interaction_gap_s": float(eng.group("interaction_gap")),
                        "intervention_gap_s": "" if gap == "never" else float(gap[:-1]),
                    }
                )
                continue

            emo = _EMOTION_RE.search(msg)
            if emo is not None:
                scores_raw = emo.group("scores")
                scores = ast.literal_eval(scores_raw) if scores_raw != "None" else {}
                gap = emo.group("intervention_gap")
                emotion_rows.append(
                    {
                        "timestamp": ts.isoformat(sep=" ", timespec="milliseconds"),
                        "t_session_s": t_session(ts),
                        "dominant_emotion": emo.group("emotion"),
                        "negative_share": float(emo.group("negative_share")),
                        "threshold": float(emo.group("threshold")),
                        "response_done": emo.group("response_done") == "True",
                        "interaction_gap_s": float(emo.group("interaction_gap")),
                        "intervention_gap_s": "" if gap == "never" else float(gap[:-1]),
                        "scores": scores,
                    }
                )
                continue

            if "Emotion poll: no face detected" in msg:
                emotion_rows.append(
                    {
                        "timestamp": ts.isoformat(sep=" ", timespec="milliseconds"),
                        "t_session_s": t_session(ts),
                        "dominant_emotion": "noface",
                        "negative_share": "",
                        "threshold": "",
                        "response_done": "",
                        "interaction_gap_s": "",
                        "intervention_gap_s": "",
                        "scores": {},
                    }
                )
                continue

            start = _SESSION_START_RE.search(msg)
            if start is not None:
                session_start = ts
                duration_minutes = float(start.group("minutes"))
                condition = "control" if start.group("control") else "treatment"
                add_event(ts, "session_start", detail=condition, value=duration_minutes)
                continue

            if "SESSION END" in msg and "min elapsed" in msg:
                session_end = ts
                add_event(ts, "session_end")
                continue

            counterfactual = _COUNTERFACTUAL_RE.search(msg)
            if counterfactual is not None:
                add_event(
                    ts,
                    f"counterfactual_{counterfactual.group('kind')}",
                    value=float(counterfactual.group("value")),
                )
                continue

            if "Queued engagement intervention prompt" in msg:
                add_event(ts, "intervention_engagement")
                continue
            if "Queued emotion intervention prompt" in msg:
                add_event(ts, "intervention_emotion")
                continue

            # Context submissions. Both conditions log the content with newlines
            # escaped as \n (kept escaped here — one line per CSV cell). Treatment
            # additionally logs "Queued user text input" once actually delivered;
            # logs from before 05.08 have ONLY that line (no content).
            received = re.search(r"Task context received: (?P<text>.*)$", msg)
            if received is not None:
                add_event(ts, "context_submit", detail=received.group("text"))
                continue
            if "Queued user text input" in msg:
                add_event(ts, "context_forwarded")
                continue
            cf_context = re.search(r"CONTROL: task context received but not forwarded: (?P<text>.*)$", msg)
            if cf_context is not None:
                add_event(ts, "context_submit", detail=cf_context.group("text"))
                continue
            rejected = re.search(r"Session gate: task context rejected \(session not active\): (?P<text>.*)$", msg)
            if rejected is not None:
                add_event(ts, "context_rejected", detail=rejected.group("text"))
                continue

            transcript = _TRANSCRIPT_RE.search(msg)
            if transcript is not None:
                add_event(ts, f"transcript_{transcript.group('role')}", detail=transcript.group("text"))
                continue

            if "Realtime websocket closed unexpectedly" in msg:
                add_event(ts, "ws_drop", detail=msg)
                continue

            activity = _ACTIVITY_RE.search(msg)
            if activity is not None:
                reason = activity.group("reason")
                if reason == "user_speech_started" and user_speech_open is None:
                    user_speech_open = ts
                elif reason == "user_speech_stopped" and user_speech_open is not None:
                    speech_rows.append({"actor": "user", "t_start": user_speech_open, "t_end": ts})
                    user_speech_open = None
                elif reason == "assistant_audio_delta":
                    if (
                        robot_run_last is not None
                        and (ts - robot_run_last).total_seconds() > ROBOT_SPEECH_GAP_S
                    ):
                        close_robot_run()
                    if robot_run_start is None:
                        robot_run_start = ts
                    robot_run_last = ts
                continue

    close_robot_run()
    if user_speech_open is not None:  # log ended mid-utterance
        speech_rows.append({"actor": "user", "t_start": user_speech_open, "t_end": user_speech_open})

    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "engagement.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp", "t_session_s", "score", "average", "threshold",
                "response_done", "interaction_gap_s", "intervention_gap_s",
            ],
        )
        writer.writeheader()
        writer.writerows(engagement_rows)

    emotion_labels = sorted({label for row in emotion_rows for label in row["scores"]})  # type: ignore[union-attr]
    with (out_dir / "emotion.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp", "t_session_s", "dominant_emotion", "negative_share",
                "threshold", "response_done", "interaction_gap_s", "intervention_gap_s",
                *emotion_labels,
            ],
        )
        writer.writeheader()
        for row in emotion_rows:
            scores = row.pop("scores")
            writer.writerow({**row, **{label: scores.get(label, "") for label in emotion_labels}})  # type: ignore[union-attr]

    with (out_dir / "speech.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["actor", "timestamp_start", "timestamp_end", "t_start_s", "t_end_s", "duration_s"]
        )
        writer.writeheader()
        for row in sorted(speech_rows, key=lambda r: r["t_start"]):  # type: ignore[arg-type,return-value]
            t_start, t_end = row["t_start"], row["t_end"]
            writer.writerow(
                {
                    "actor": row["actor"],
                    "timestamp_start": t_start.isoformat(sep=" ", timespec="milliseconds"),  # type: ignore[union-attr]
                    "timestamp_end": t_end.isoformat(sep=" ", timespec="milliseconds"),  # type: ignore[union-attr]
                    "t_start_s": t_session(t_start),  # type: ignore[arg-type]
                    "t_end_s": t_session(t_end),  # type: ignore[arg-type]
                    "duration_s": round((t_end - t_start).total_seconds(), 3),  # type: ignore[operator]
                }
            )

    with (out_dir / "events.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "t_session_s", "event_type", "value", "detail"])
        writer.writeheader()
        writer.writerows(event_rows)

    summary: dict[str, object] = {
        "log_file": str(log_path),
        "condition": condition,
        "session_start": session_start.isoformat(sep=" ") if session_start else None,
        "session_end": session_end.isoformat(sep=" ") if session_end else None,
        "configured_duration_min": duration_minutes,
        "log_span": [
            first_ts.isoformat(sep=" ") if first_ts else None,
            last_ts.isoformat(sep=" ") if last_ts else None,
        ],
        "rows": {
            "engagement": len(engagement_rows),
            "emotion": len(emotion_rows),
            "speech_user": sum(1 for r in speech_rows if r["actor"] == "user"),
            "speech_robot": sum(1 for r in speech_rows if r["actor"] == "robot"),
            "events": len(event_rows),
        },
        "event_counts": {
            event_type: sum(1 for r in event_rows if r["event_type"] == event_type)
            for event_type in sorted({str(r["event_type"]) for r in event_rows})
        },
    }
    with (out_dir / "session.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", type=Path, help="app.log to parse")
    parser.add_argument("-o", "--out-dir", type=Path, default=None, help="Output directory for the CSVs")
    args = parser.parse_args()

    out_dir = args.out_dir or args.log.parent / f"{args.log.stem}_csv"
    summary = parse_log(args.log, out_dir)
    print(json.dumps(summary, indent=2))
    print(f"\nWrote CSVs to {out_dir}/")


if __name__ == "__main__":
    main()
