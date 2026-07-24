"""Gate simulation: label-counting vs probability-weighted intervention gate, both backends.

Re-classifies the archived frame-dump sessions through the APP's own classifier
(reachy_mini_conversation_app.emotion_classifier, both EMOTION_CLASSIFIER_BACKEND values),
then replays each timeline through two gates:

  old (label): share of window samples whose dominant label is negative     > 0.40
  new (mass):  mean over window samples of P(angry)+P(disgust)+P(fear)+P(sad) > 0.40

reporting episodes (below->above threshold crossings, 30s trailing window, matching the
prior awk episode-simulation methodology) plus calibration stats (mean top-1 probability)
per backend.

Classifications are cached to results_cache_*.jsonl next to this script (gitignored via
experiments/**/results_*), so gate/threshold tweaks re-run in seconds; delete the cache
to force re-classification.

Run with the app venv (the app package must be importable):

    ../../study_assistant/.venv/bin/python gate_simulation.py
"""

import json
import sys
import time
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
SESSIONS = {
    "arm1-webcam-21.07": REPO_ROOT / "logs" / "frames" / "2026-07-21_opencv-webcam",
    "leetcode-22.07": REPO_ROOT / "logs" / "frames" / "opencv" / "2026-07-22_1132",
}
BACKENDS = ("deepface", "emotiefflib")
NEGATIVE = {"angry", "disgust", "fear", "sad"}
WINDOW_SECONDS = 30.0
THRESHOLD = 0.40


def frame_timestamp_seconds(path: Path) -> int:
    """Filename convention HHMMSS_<label>.jpg -> seconds since midnight."""
    hhmmss = path.stem.split("_", 1)[0]
    return int(hhmmss[0:2]) * 3600 + int(hhmmss[2:4]) * 60 + int(hhmmss[4:6])


def classify_session(frames: list[Path], backend: str, cache_path: Path) -> list[dict]:
    """Classify every frame with the app classifier under `backend`; cache as JSON lines."""
    if cache_path.exists():
        return [json.loads(line) for line in cache_path.read_text().splitlines()]

    from reachy_mini_conversation_app.config import config
    from reachy_mini_conversation_app.emotion_classifier import classify_dominant_emotion

    config.EMOTION_CLASSIFIER_BACKEND = backend
    samples = []
    started = time.monotonic()
    for i, path in enumerate(frames, 1):
        result = classify_dominant_emotion(cv2.imread(str(path)))
        if result is None:
            sample = {"t": frame_timestamp_seconds(path), "frame": path.name, "label": None}
        else:
            label, scores = result
            sample = {
                "t": frame_timestamp_seconds(path),
                "frame": path.name,
                "label": label,
                "top1": max(scores.values()),
                "neg_mass": sum(scores.get(e, 0.0) for e in NEGATIVE),
            }
        samples.append(sample)
        if i % 50 == 0:
            print(f"    {backend}: {i}/{len(frames)} frames ({time.monotonic() - started:.0f}s)", flush=True)

    cache_path.write_text("\n".join(json.dumps(s) for s in samples))
    return samples


def count_episodes(samples: list[dict], value_of, threshold: float = THRESHOLD) -> int:
    """Below->above threshold crossings of the trailing-window mean of value_of(sample)."""
    classified = [s for s in samples if s["label"] is not None]
    episodes = 0
    above = False
    for i, sample in enumerate(classified):
        window = [value_of(s) for s in classified[: i + 1] if s["t"] > sample["t"] - WINDOW_SECONDS]
        share = sum(window) / len(window)
        if share > threshold and not above:
            episodes += 1
        above = share > threshold
    return episodes


def main() -> None:
    rows = []
    for session, frames_dir in SESSIONS.items():
        frames = sorted(frames_dir.glob("*.jpg"))
        if not frames:
            sys.exit(f"no frames in {frames_dir}")
        print(f"{session}: {len(frames)} frames")
        for backend in BACKENDS:
            cache = SCRIPT_DIR / f"results_cache_{session}_{backend}.jsonl"
            samples = classify_session(frames, backend, cache)
            classified = [s for s in samples if s["label"] is not None]
            mean_top1 = sum(s["top1"] for s in classified) / len(classified)
            rows.append(
                {
                    "session": session,
                    "backend": backend,
                    "frames": len(classified),
                    "noface": len(samples) - len(classified),
                    "mean_top1": mean_top1,
                    "old_gate": count_episodes(samples, lambda s: float(s["label"] in NEGATIVE)),
                    "new_gate": count_episodes(samples, lambda s: s["neg_mass"]),
                }
            )

    print()
    print(f"{'session':<22} {'backend':<12} {'frames':>6} {'noface':>6} {'top1':>6} {'old':>4} {'new':>4}")
    print("-" * 66)
    for r in rows:
        print(
            f"{r['session']:<22} {r['backend']:<12} {r['frames']:>6} {r['noface']:>6} "
            f"{r['mean_top1']:>6.2f} {r['old_gate']:>4} {r['new_gate']:>4}"
        )
    print(f"\ngates: trailing {WINDOW_SECONDS:.0f}s window, threshold {THRESHOLD}; episodes = below->above crossings")

    # Threshold sweep for the mass gate + the mass distribution itself: the 0.40 vote
    # threshold need not be the right mass threshold (soft backends carry a noise floor).
    print(f"\nnew-gate episodes by threshold, and neg-mass distribution (median/p90 of classified frames)")
    sweep = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    header = " ".join(f"{t:>5.2f}" for t in sweep)
    print(f"{'session':<22} {'backend':<12} {header}   med   p90")
    for session, frames_dir in SESSIONS.items():
        for backend in BACKENDS:
            cache = SCRIPT_DIR / f"results_cache_{session}_{backend}.jsonl"
            samples = [json.loads(line) for line in cache.read_text().splitlines()]
            masses = sorted(s["neg_mass"] for s in samples if s["label"] is not None)
            episodes = [count_episodes(samples, lambda s: s["neg_mass"], threshold=t) for t in sweep]
            counts = " ".join(f"{e:>5}" for e in episodes)
            med = masses[len(masses) // 2]
            p90 = masses[int(len(masses) * 0.9)]
            print(f"{session:<22} {backend:<12} {counts}  {med:.2f}  {p90:.2f}")


if __name__ == "__main__":
    main()
