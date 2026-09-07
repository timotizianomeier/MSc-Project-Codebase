"""Raw-data package for Nicole (rebuilt 07.09 — original scratchpad
script wiped; same structure as the 03.09 package): metrics_tidy.csv
(one row per participant x condition: group, coverage-gate flags, every
session metric incl. half splits), parsed_logs.zip (CSV session logs,
pseudonymous P-numbers, no recordings, no linking file), README.txt."""
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

import generate_results as gr
from generate_appendix import gated_signals, session_dirs

OUT = os.path.expanduser("~/Downloads/nicole_data_package")
os.makedirs(OUT, exist_ok=True)

pre, _ = gr.load_qualtrics(gr.newest_file(gr.FILE_PATTERNS["pre"]))
pre = gr.clean(pre, "PRE_PID", "pre")
groups = gr.assign_groups(pre)
sdirs = session_dirs()
gated = gated_signals(sdirs)

rows = {}
for pid in sorted(groups, key=int):
    for cond in ("Robot", "Control"):
        rows[(pid, cond)] = {
            "participant": f"P{pid}", "condition": cond,
            "group": groups[pid],
            "eng_signal_gated": (pid, cond, "eng") in gated,
            "emo_signal_gated": (pid, cond, "emo") in gated,
        }
for label, df, dec in gr._cross_condition_rows(groups, None):
    for pid, r in df.iterrows():
        for cond in ("Robot", "Control"):
            if (pid, cond) in rows:
                rows[(pid, cond)][label] = r[cond]
for label, vals, dec in gr._half_split_rows(groups, set(groups)):
    for (pid, cond, half), v in vals.items():
        if (pid, cond) in rows:
            rows[(pid, cond)][f"{label} [half {half}]"] = v

tidy = pd.DataFrame(list(rows.values()))
tidy.to_csv(os.path.join(OUT, "metrics_tidy.csv"), index=False)
print(f"metrics_tidy.csv: {len(tidy)} rows x {len(tidy.columns)} cols")

logs_root = os.path.expanduser(
    "~/Projects/MSc-Project-Codebase/analysis/logs")
zpath = os.path.join(OUT, "parsed_logs.zip")
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(logs_root):
        for f in sorted(files):
            if f.endswith((".csv", ".json")):
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, logs_root))
print(f"parsed_logs.zip: {os.path.getsize(zpath) / 1e6:.1f} MB")

readme = """Data package for Nicole - ADHD study-assistant robot study (N=22)
Generated 07.09.2026 from the parsed session logs.

metrics_tidy.csv
  One row per participant x condition (44 rows). Columns: group,
  per-signal coverage-gate flags (True = that signal series covered <80%
  of the session and is excluded from signal-level analyses), then every
  metric from the results tables, including first/second-half splits.
  SPEECH EXCLUSION (adopted 03.09): the five signal-level metrics - mean
  engagement score, mean neg.-emotion share, and the three time-within-
  threshold percentages (plus their half splits) - exclude every sample
  recorded while the user or the robot was speaking, and the 10 s after
  each speech segment ends (the engagement score integrates a rolling
  window of ~10 frames, so it stays contaminated for a few seconds after
  speech stops). Exclusion intervals are subtracted exactly from both the
  numerator and the denominator of the percentages. Intervention counts
  and episode durations are NOT affected - they keep wall-clock
  semantics - and neither is the 80% coverage gate, which is computed on
  the raw poll series.
  Control-condition intervention counts are COUNTERFACTUAL events (what
  the robot would have said) and are an upper bound - no conversation
  existed there to reset the interaction cooldown.

parsed_logs.zip
  Per-session CSV folders (P<id>_<condition>_..._csv): engagement.csv
  (5s polls: raw score, rolling average, threshold, gating state),
  emotion.csv (per-emotion probabilities, negative share, threshold),
  events.csv (interventions, counterfactuals, transcripts, latencies),
  speech.csv (speech segments with actor and duration).
  Episode definition used in the analyses: >=2 consecutive polls past
  threshold (engagement rolling avg < 0.80 / negative share > 0.60);
  inter-poll gaps > 30s censor an episode.

Pseudonymous P-numbers throughout; no recordings, no linking file.
"""
open(os.path.join(OUT, "README.txt"), "w").write(readme)
print("README.txt written ->", OUT)
