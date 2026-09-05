#!/usr/bin/env python3
"""
generate_appendix.py
====================
Generates LaTeX appendix fragments (tables + pgfplots charts) from the latest
Qualtrics CSV exports of the pre-session, control-session and post-session
(robot) surveys.

Usage:
    python generate_appendix.py

Output:
    One .tex file per appendix section in OUTPUT_DIR, plus a preamble snippet.
    \\input{} these from the Appendix subfile in Overleaf.

IMPORTANT: Raw CSVs and the generated .tex contain participant data
(free-text answers). Keep both OUT of the git repo — only this script is
committed. Add `analysis/appendix_gen/output/` to .gitignore.

Handles both Qualtrics export modes ("Use numeric values" and
"Use choice text") transparently.
"""

from __future__ import annotations

import csv
import glob
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats as scistats

# ============================================================================
# CONFIG — edit here
# ============================================================================

# Directory holding the raw Qualtrics CSV exports (NOT synced to GitHub).
# Lives on Box next to the session data — same ethics protocol, backed up,
# and participant_groups.txt lands there too (never in git).
DATA_DIR = os.path.expanduser(
    "~/Library/CloudStorage/Box-Box/MSc-Project-Storage-Timo/study-data/qualtrics"
)

# Where generated .tex fragments go (gitignore this).
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# The newest file matching each pattern is used (Qualtrics timestamps the
# filename on every export, and the pattern sorts chronologically by mtime).
FILE_PATTERNS = {
    "pre": "pre-session-*.csv",
    "control": "control-session-*.csv",
    "post": "post-session-*.csv",
}

# Likert layout: question rows per page. Pagination is emitted explicitly
# (\newpage) so the answer-scale labels can be shown once per page — bold,
# below the LAST chart of the page — and omitted on the rows above it.
# 8 rows of 2.2cm charts + 1em row gaps fit an a4 12pt page with 1in
# margins; lower the row count or height if a template change overflows.
LIKERT_ROWS_PER_PAGE = 8
LIKERT_CHART_HEIGHT = "2.2cm"

# Run with --sync to also copy the three fragments into the thesis repo's
# apx-subfiles/ folder, commit, and push (then in Overleaf: Menu -> GitHub ->
# "Pull GitHub changes"). The thesis repo is PRIVATE and must stay private —
# the fragments contain verbatim participant answers.
THESIS_APX_DIR = os.path.expanduser(
    "~/Projects/MSc-Project-Final-Report/apx-subfiles")
SYNC_FILES = ["apx_pre_study.tex", "apx_post_control.tex", "apx_post_robot.tex",
              "apx_session_logs.tex", "apx_session_stats.tex",
              "apx_instrument_stats.tex"]

# Short subheaders for the open-ended questions (the full question text is
# still printed above each answer table). Keys are CSV column names.
OE_HEADERS = {
    "POST_OE_01": "Task description",
    "POST_OE_02": "Overall experience",
    "POST_OE_03": "Body doubling effect",
    "POST_OE_04": "Awareness of disengagement detection",
    "POST_OE_05": "Re-engagement cues",
    "POST_OE_06": "Task-aware guidance",
    "POST_OE_07": "Comparison to a human study partner",
    "POST_OE_08": "Task initiation and persistence",
    "POST_OE_09": "Suggested changes",
    "POST_OE_10": "Unhelpful or distracting moments",
    "POST_OE_11": "Interest in future use",
    "POST_OE_12": "Recommendation to others",
}

# DISPLAY-ONLY renumbering: study PIDs started at 11, but the thesis shows
# P1, P2, ... Applied ONLY when a PID is rendered into the LaTeX output —
# filenames, raw CSVs, INCLUDE_PIDS, groups files etc. all keep true PIDs
# (the authoritative mapping lives in Participant_Linking_File.xlsx on Box;
# keep the two in sync). Default: subtract 10 (P11->P1 ... P27->P17). Edit
# individual entries here if the thesis needs different labels (e.g. to
# close the gap left by P26 until their rescheduled session happens).
PID_DISPLAY = {str(n): str(n - 10) for n in range(11, 40)}


def disp_pid(pid) -> str:
    """Display label for a participant id (falls back to the true PID)."""
    return PID_DISPLAY.get(str(pid), str(pid))


# Participants to INCLUDE (whitelist). Compared after PID normalisation
# (leading zeros stripped, so "0001" == "1"). Empty set = include everyone.
INCLUDE_PIDS: set[str] = {"11", "12", "13", "14", "15", "16", "17", "18",
                          "19", "20", "21", "22", "23", "24", "25", "26",
                          "27", "28", "29", "30", "31", "32"}

# PIDs to exclude (applied after INCLUDE_PIDS; mostly redundant with a
# whitelist, kept for when INCLUDE_PIDS is empty).
EXCLUDE_PIDS: set[str] = set()

# How to split participants into ADHD / Control:
#   "file"      -> read participant_groups.txt (written by compute_asrs.py,
#                  or hand-edited / pasted; lives in DATA_DIR, never in git)
#   "asrs"      -> compute from ASRS in the pre-survey (see knobs below)
#   "diagnosis" -> PRE_ADHD_DX == Yes
GROUPING = "file"
GROUPS_FILE = os.path.join(DATA_DIR, "participant_groups.txt")

# --- ASRS scoring (used by GROUPING="asrs" and by compute_asrs.py) --------
# Your questionnaire's ASRS items are NOT in the official checklist order.
# Verified mapping (12.08.2026, against ASRS v1.1 and Questionnaire Sheet):
# official Part A item -> (your PRE_ASRS_<n>, minimum "shaded box" response)
# Responses coded 0=Never, 1=Rarely, 2=Sometimes, 3=Often, 4=Very often.
ASRS_PART_A = {
    "A1 final details":      (4,  2),   # positive at Sometimes+
    "A2 organisation":       (5,  2),   # positive at Sometimes+
    "A3 remembering appts":  (9,  2),   # positive at Sometimes+
    "A4 delayed start":      (6,  3),   # positive at Often+
    "A5 fidgeting":          (10, 3),   # positive at Often+
    "A6 driven by motor":    (14, 3),   # positive at Often+
}
# Which ASRS metric defines the ADHD group, and the threshold (group = ADHD
# when metric > ASRS_THRESHOLD, i.e. strictly above):
#   "screener_positives" -> count of positive Part A items (0-6).
#       Threshold 3 (=> at least 4 positives) equals the standard Kessler
#       cutoff and O'Connell et al. (2024). Lalwani et al. (2025) used
#       "above 3" as their own threshold.
#   "mean_score_1to5"    -> mean over all 18 items, rescaled to 1-5.
#   "sum_score"          -> sum over all 18 items on the raw 0-4 coding.
ASRS_METRIC = "screener_positives"
ASRS_THRESHOLD = 3

GROUP_ADHD = "ADHD"
GROUP_CONTROL = "Control"

# ============================================================================
# Scale definitions
# ============================================================================

LIKERT5 = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]
ASRS_FREQ5 = ["Never", "Rarely", "Sometimes", "Often", "Very often"]
ESQR_FREQ = ["Never or rarely", "Sometimes", "Often", "Very often"]  # ESQ-R codes 0-3
YESNO = ["Yes", "No", "Prefer not to say"]
GENDER_ORDER = ["Male", "Female", "Non-binary / third gender",
                "Prefer to self-describe", "Prefer not to say"]
LEVEL_ORDER = ["Bachelor's", "Master's", "PhD", "Other"]

# Known orderings, tried in turn when answers arrive as choice text.
KNOWN_ORDERINGS = [LIKERT5, ASRS_FREQ5, ESQR_FREQ, GENDER_ORDER, LEVEL_ORDER, YESNO]

# When answers arrive as numeric recode values, these map code -> label for
# axis ticks. Adjust if your Qualtrics recode values differ (check in
# Survey > question > Recode values).
# Verified against the 12.08.2026 numeric export (cross-checked with the
# choice-text export): NARS/SUS/features 1-5, ASRS 0-4, ESQ-R 0-3,
# Gender 1=Male 2=Female..., Level 1=Bachelor's..., Yes/No 1=Yes 2=No.
NUMERIC_LABELS = {
    "LIKERT5": {i + 1: lbl for i, lbl in enumerate(LIKERT5)},
    "ASRS": {i: lbl for i, lbl in enumerate(ASRS_FREQ5)},        # 0-4
    "ESQR": {i: lbl for i, lbl in enumerate(ESQR_FREQ)},         # 0-3
    "GENDER": {i + 1: lbl for i, lbl in enumerate(GENDER_ORDER)},
    "LEVEL": {i + 1: lbl for i, lbl in enumerate(LEVEL_ORDER)},
    "YESNO": {i + 1: lbl for i, lbl in enumerate(YESNO)},
}

# ============================================================================
# Small utilities
# ============================================================================

LATEX_SPECIALS = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def esc(text) -> str:
    """Escape LaTeX special characters in free text."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = str(text)
    s = s.replace("\\", LATEX_SPECIALS["\\"])
    for ch, rep in LATEX_SPECIALS.items():
        if ch == "\\":
            continue
        s = s.replace(ch, rep)
    # collapse newlines inside answers
    s = re.sub(r"\s*\n\s*", r" \\newline ", s)
    return s


def normalise_pid(raw) -> str | None:
    """'0001' -> '1'; reject anything non-numeric (test rows, blanks)."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    s = str(raw).strip()
    if s.endswith(".0"):  # pandas float artefact
        s = s[:-2]
    if not s.isdigit():
        return None
    return s.lstrip("0") or "0"


def newest_file(pattern: str) -> str:
    matches = glob.glob(os.path.join(DATA_DIR, pattern))
    if not matches:
        sys.exit(f"ERROR: no file matching '{pattern}' in {DATA_DIR}")
    return max(matches, key=os.path.getmtime)


def load_qualtrics(path: str) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load a Qualtrics CSV. Returns (data, {column -> question text}).

    Qualtrics exports have three header rows: internal names, question text,
    and an ImportId JSON row. We keep row 0 as columns, capture row 1 as the
    question-text map, and skip row 2.
    """
    qtext_row = pd.read_csv(path, nrows=1, skiprows=[0], header=None, dtype=str)
    names = pd.read_csv(path, nrows=0).columns.tolist()
    qtext = dict(zip(names, qtext_row.iloc[0].fillna("").tolist()))
    df = pd.read_csv(path, skiprows=[1, 2], dtype=str)
    return df, qtext


def clean(df: pd.DataFrame, pid_col: str, label: str) -> pd.DataFrame:
    """Drop unfinished, invalid-PID, excluded and preview rows."""
    n0 = len(df)
    df = df.copy()
    df["PID"] = df[pid_col].map(normalise_pid)
    if "Finished" in df.columns:
        df = df[df["Finished"].astype(str).str.lower().isin(["true", "1"])]
    if "Status" in df.columns:  # 'Survey Preview' / 'Spam' etc. in text exports
        df = df[~df["Status"].astype(str).str.contains("Preview", case=False, na=False)]
    df = df[df["PID"].notna()]
    if INCLUDE_PIDS:
        skipped = sorted(set(df["PID"]) - INCLUDE_PIDS)
        if skipped:
            print(f"  {label}: not in INCLUDE_PIDS, skipped: {skipped}")
        df = df[df["PID"].isin(INCLUDE_PIDS)]
    df = df[~df["PID"].isin(EXCLUDE_PIDS)]
    dropped = n0 - len(df)
    dup = df["PID"].duplicated(keep="last")
    if dup.any():
        print(f"  WARNING [{label}]: duplicate PIDs {sorted(df.loc[dup, 'PID'].unique())} "
              f"— keeping the most recent response per PID.")
        df = df[~df["PID"].duplicated(keep="last")]
    print(f"  {label}: kept {len(df)} of {n0} rows ({dropped} filtered).")
    return df.reset_index(drop=True)


# ============================================================================
# Value handling: numeric export vs choice-text export
# ============================================================================

def series_numeric(s: pd.Series) -> pd.Series | None:
    """Return the numeric version of a series, or None if it's choice text."""
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().sum() >= s.notna().sum() and s.notna().any():
        return x
    return None


def category_order(observed: list, hint: str | None = None) -> list:
    """Ordering of answer categories for one question (labels or codes)."""
    obs = [o for o in observed if o not in (None, "", np.nan)]
    nums = pd.to_numeric(pd.Series(obs), errors="coerce")
    if len(obs) and nums.notna().all():  # numeric export
        codes = sorted({int(v) if float(v).is_integer() else float(v) for v in nums})
        labels = NUMERIC_LABELS.get(hint or "", {})
        if labels and set(codes) <= set(labels):
            return sorted(labels.items())  # full canonical scale
        if labels:
            print(f"  WARNING: codes {sorted(set(codes) - set(labels))} outside "
                  f"the verified {hint} map — check Qualtrics recode values.")
        return [(c, labels.get(c, str(c))) for c in codes]
    hint_orderings = {"LIKERT5": LIKERT5, "ASRS": ASRS_FREQ5, "ESQR": ESQR_FREQ}
    if hint in hint_orderings and set(obs) <= set(hint_orderings[hint]):
        return [(o, o) for o in hint_orderings[hint]]  # full canonical scale
    for ordering in KNOWN_ORDERINGS:  # choice-text export, no/failed hint
        if set(obs) <= set(ordering):
            return [(o, o) for o in ordering]
    print(f"  WARNING: unknown answer set {sorted(set(map(str, obs)))} — using alphabetical order.")
    return [(o, o) for o in sorted(set(obs), key=str)]


def to_rank(s: pd.Series, hint: str | None = None) -> pd.Series:
    """Map answers to their numeric codes for summary statistics.
    Numeric export passes through; choice text is mapped via the verified
    code maps (so both modes yield identical statistics)."""
    num = series_numeric(s)
    if num is not None:
        return num
    if hint in NUMERIC_LABELS:
        mapping = {lbl: code for code, lbl in NUMERIC_LABELS[hint].items()}
    else:
        order = [v for v, _ in category_order(s.dropna().unique().tolist(), hint)]
        mapping = {v: i + 1 for i, v in enumerate(order)}
    return s.map(mapping).astype(float)


# ============================================================================
# Grouping
# ============================================================================

def _asrs_code(val) -> float:
    """One ASRS answer -> 0-4 code, whichever export mode.
    Numeric export: codes are already 0=Never .. 4=Very often (verified).
    Text export: label position in ASRS_FREQ5 gives the same code."""
    num = pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
    if not pd.isna(num):
        return float(num)
    try:
        return float(ASRS_FREQ5.index(str(val).strip()))
    except ValueError:
        return float("nan")


def asrs_metrics(row: pd.Series) -> dict:
    """All ASRS metrics for one pre-survey row (both export modes)."""
    positives, part_a = 0, {}
    for name, (item, thr) in ASRS_PART_A.items():
        code = _asrs_code(row.get(f"PRE_ASRS_{item}"))
        pos = (not np.isnan(code)) and code >= thr
        positives += int(pos)
        part_a[name] = (item, code, pos)
    codes = [_asrs_code(row.get(f"PRE_ASRS_{i}")) for i in range(1, 19)]
    codes = [c for c in codes if not np.isnan(c)]
    return {
        "screener_positives": positives,
        "mean_score_1to5": (sum(codes) / len(codes) + 1) if codes else float("nan"),
        "sum_score": sum(codes) if codes else float("nan"),
        "n_items_answered": len(codes),
        "part_a": part_a,
    }


def parse_groups_file(path: str) -> dict[str, str]:
    """participant_groups.txt: 'ADHD: 11, 13' / 'CONTROL: 12, 14' lines
    (case-insensitive; '#' comments allowed)."""
    if not os.path.exists(path):
        sys.exit(f"ERROR: GROUPING='file' but {path} does not exist.\n"
                 f"Run compute_asrs.py first, or create it by hand, "
                 f"or set GROUPING to 'asrs'/'diagnosis'.")
    groups = {}
    for line in open(path):
        line = line.split("#")[0].strip()
        if not line or ":" not in line:
            continue
        key, ids = line.split(":", 1)
        gname = {"ADHD": GROUP_ADHD, "CONTROL": GROUP_CONTROL}.get(key.strip().upper())
        if gname is None:
            print(f"  WARNING: ignoring unknown group '{key.strip()}' in {path}")
            continue
        for raw in ids.replace(";", ",").split(","):
            pid = normalise_pid(raw.strip())
            if pid:
                if pid in groups and groups[pid] != gname:
                    sys.exit(f"ERROR: PID {pid} listed as both groups in {path}")
                groups[pid] = gname
    return groups


def assign_groups(pre: pd.DataFrame) -> dict[str, str]:
    if GROUPING == "file":
        groups = parse_groups_file(GROUPS_FILE)
        missing = sorted(set(pre["PID"]) - set(groups))
        if missing:
            print(f"  WARNING: PIDs in pre-survey but not in "
                  f"{os.path.basename(GROUPS_FILE)} (excluded from all "
                  f"output): {missing}")
        groups = {p: g for p, g in groups.items() if p in set(pre["PID"])}
    else:
        groups = {}
        for _, row in pre.iterrows():
            if GROUPING == "diagnosis":
                is_adhd = str(row.get("PRE_ADHD_DX", "")).strip().lower() in ("yes", "1")
            elif GROUPING == "asrs":
                is_adhd = asrs_metrics(row)[ASRS_METRIC] > ASRS_THRESHOLD
            else:
                sys.exit(f"ERROR: unknown GROUPING '{GROUPING}'")
            groups[row["PID"]] = GROUP_ADHD if is_adhd else GROUP_CONTROL
    n_a = sum(1 for g in groups.values() if g == GROUP_ADHD)
    print(f"  Groups ({GROUPING}): {n_a} {GROUP_ADHD}, {len(groups) - n_a} {GROUP_CONTROL}.")
    return groups


# ============================================================================
# Questionnaire scoring (shared with compute_stats.py)
# ============================================================================

# ESQ-R subscales: official instrument (Strait et al. 2020) has 5 subscales
# over 25 items, coded 0-3. Mapping verified 26.08.2026 against Strait et
# al. (2020) Table 2 by matching item text to our Qualtrics order. 24/25
# matched verbatim; our item 20 is assigned to Time Management by
# elimination (paper item 42, the remaining Factor-2 slot). Factor 5 had
# the weakest internal consistency in the source (alpha = .65).
ESQR_SUBSCALES: dict[str, list[int]] = {
    "Plan management": [6, 7, 12, 13, 14, 16, 17, 18, 22, 23, 24],
    "Time management": [10, 11, 15, 20],
    "Materials organization": [3, 8, 9],
    "Emotional regulation": [4, 5, 21],
    "Behavioral regulation": [1, 2, 19, 25],
}

# NARS reverse-scored items, verified 26.08.2026 by item wording: our items
# 3, 5, 6 are the positively-worded S3 trio of the official scale (Nomura
# et al. 2006). Higher total = more negative attitude.
NARS_REVERSE_ITEMS: set[int] = {3, 5, 6}

N_NARS_ITEMS = 14
N_ESQR_ITEMS = 25
TLX_DIMS = ["MENTAL", "PHYSICAL", "TEMPORAL", "PERFORMANCE", "EFFORT",
            "FRUSTRATION"]
FEATURE_BLOCKS = [
    ("Body doubling / presence", "POST_FEAT_PRESENCE_", 3),
    ("Inattention detection", "POST_FEAT_INATT_", 5),
    ("Context-aware support", "POST_FEAT_CONTEXT_", 4),
    ("Overall experience", "POST_FEAT_OVERALL_", 3),
]

# Frustration-mechanism predictors (exploratory Spearman correlations
# against the robot-minus-control TLX frustration delta).
FRUSTRATION_PREDICTORS = [
    ("n_tot", "Total interventions"),
    ("n_emo", "Emotion interventions"),
    ("n_eng", "Engagement interventions"),
    ("med_lat_s", "Median first-audio latency (s)"),
    ("user_min", "User talk-time (min)"),
    ("robot_min", "Robot talk-time (min)"),
    ("n_turns", "Spoken turns"),
]


def num_col(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df else pd.Series(dtype=float)


def mean_items(df: pd.DataFrame, prefix: str, items: list[int],
               reverse: set[int] | None = None, scale_max: int = 5) -> pd.Series:
    cols = []
    for i in items:
        s = num_col(df, f"{prefix}{i}")
        if reverse and i in reverse:
            s = (scale_max + 1) - s
        cols.append(s)
    return pd.concat(cols, axis=1).mean(axis=1)


def sus_scores(post: pd.DataFrame) -> pd.Series:
    vals = []
    for _, r in post.iterrows():
        items = [pd.to_numeric(r.get(f"POST_SUS_{i}"), errors="coerce")
                 for i in range(1, 11)]
        if any(pd.isna(v) for v in items):
            vals.append(np.nan)
            continue
        vals.append(sum((v - 1) if i % 2 == 1 else (5 - v)
                        for i, v in enumerate(items, 1)) * 2.5)
    return pd.Series(vals, index=post.index)


# ============================================================================
# LaTeX renderers
# ============================================================================

ADHD_COLOR = "ApxADHD"
CTRL_COLOR = "ApxControl"
COND_ROBOT_COLOR = "ApxCondRobot"
COND_CTRL_COLOR = "ApxCondControl"

PREAMBLE_SNIPPET = r"""% ---------------------------------------------------------------
% Requirements for the auto-generated appendix fragments.
% Add ONCE to the thesis preamble (Main.tex), not to the fragments.
% ---------------------------------------------------------------
\usepackage{amsmath}
\usepackage{pgfplots}
\pgfplotsset{compat=1.17}
\usepgfplotslibrary{statistics}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{pdflscape}
\usepackage{colortbl}
\usepackage{xcolor}
\definecolor{ApxADHD}{RGB}{68,119,170}    % Tol blue
\definecolor{ApxControl}{RGB}{204,102,17} % Tol orange (darker, prints well)
\definecolor{ApxUserSpeech}{RGB}{102,153,204}  % session timelines: user
\definecolor{ApxRobotSpeech}{RGB}{238,153,68}  % session timelines: robot
\definecolor{ApxSilence}{RGB}{235,235,235}     % session timelines: no speech
\definecolor{ApxTrigEng}{RGB}{187,34,34}       % engagement trigger marks
\definecolor{ApxTrigEmo}{RGB}{34,136,85}       % emotion trigger marks
\definecolor{ApxCondRobot}{RGB}{170,68,153}    % condition charts: robot session
\definecolor{ApxCondControl}{RGB}{119,119,119} % condition charts: control session
% ---------------------------------------------------------------
"""


def header_comment(sources: list[str]) -> str:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    src = ", ".join(os.path.basename(s) for s in sources)
    return (f"% AUTO-GENERATED by generate_appendix.py on {stamp}\n"
            f"% Sources: {src}\n"
            f"% Do not edit by hand — rerun the script instead.\n\n")


def _ymax(peak: int, ceiling: int | None = None) -> float:
    """Axis top for a histogram. `ceiling` = tallest bar across the whole
    instrument (NARS, SUS, ...) so every chart WITHIN one instrument shares
    the same y-scale and bar heights are directly comparable; None -> the
    chart autoscales to its own tallest bar."""
    top = ceiling if ceiling else max(peak, 1)
    return top * 1.3 + 0.3  # headroom for the count labels above the bars


def stacked_chart(counts_by_group: dict[str, dict], cats: list[tuple], *,
                  width: str = "0.55\\textwidth", height: str = "3.4cm",
                  rotate_labels: bool = True, single_series: bool = False,
                  label_mode: str = "default", show_legend: bool = True,
                  y_ceiling: int | None = None) -> str:
    """One pgfplots ybar chart: x = answer categories, ADHD and Control as
    side-by-side (grouped) bars with the count above each nonzero bar.
    `cats` is [(value, display_label), ...].

    label_mode: "default" — tick labels below the axis (rotated per
    rotate_labels); "footer" — bold labels below the axis, wrapped one/two
    words per line (LAST chart on a page of a likert batch); "none" — no
    tick labels (charts above the footer chart on the same page)."""
    labels = [esc(lbl) for _, lbl in cats]
    sym = ",".join("{" + l + "}" for l in labels)
    a = [int(counts_by_group.get(GROUP_ADHD, {}).get(v, 0)) for v, _ in cats]
    c = [int(counts_by_group.get(GROUP_CONTROL, {}).get(v, 0)) for v, _ in cats]
    peak = max(a + c) if single_series is False else max(a, default=0)
    ymax = _ymax(peak, y_ceiling)

    coords_a = " ".join(f"({{{l}}},{v})" for l, v in zip(labels, a))
    coords_c = " ".join(f"({{{l}}},{v})" for l, v in zip(labels, c))

    if label_mode == "footer":
        # Wrapped (word-per-line) bold labels below the axis.
        xtick_style = ("x tick label style="
                       "{font=\\scriptsize\\bfseries, align=center, text width=1.55cm},")
    elif label_mode == "none":
        xtick_style = "xticklabels={},"
    elif rotate_labels:
        xtick_style = "x tick label style={rotate=28, anchor=east, font=\\scriptsize},"
    else:
        xtick_style = "x tick label style={font=\\scriptsize},"

    # Grouped bars: 8pt wide, 2pt apart -> each plot's bars sit +-5pt from
    # the category centre; the count nodes use the same shift to land above
    # their own bar.
    nodes = []
    if single_series:
        for l, av in zip(labels, a):
            if av > 0:
                nodes.append(f"\\node[font=\\footnotesize, above] at (axis cs:{{{l}}},{av}) {{\\textbf{{{av}}}}};")
    else:
        for l, av, cv in zip(labels, a, c):
            if av > 0:
                nodes.append(f"\\node[font=\\scriptsize, above, xshift=-5pt] at (axis cs:{{{l}}},{av}) {{{av}}};")
            if cv > 0:
                nodes.append(f"\\node[font=\\scriptsize, above, xshift=5pt] at (axis cs:{{{l}}},{cv}) {{{cv}}};")
    nodes_tex = "\n    ".join(nodes)

    if single_series:
        bar_opts = "ybar, bar width=11pt"
        plots = (f"\\addplot[ybar, fill={ADHD_COLOR}!45, draw={ADHD_COLOR}] "
                 f"coordinates {{{coords_a}}};")
    else:
        bar_opts = "ybar=2pt, bar width=8pt"
        legend_cmd = "\n    \\legend{ADHD, No-ADHD}" if show_legend else ""
        plots = (f"\\addplot[ybar, fill={ADHD_COLOR}, draw={ADHD_COLOR}!70!black] "
                 f"coordinates {{{coords_a}}};\n"
                 f"    \\addplot[ybar, fill={CTRL_COLOR}, draw={CTRL_COLOR}!70!black] "
                 f"coordinates {{{coords_c}}};{legend_cmd}")

    return f"""\\begin{{tikzpicture}}[trim axis left, trim axis right]
  \\begin{{axis}}[
    {bar_opts}, scale only axis, width={width}, height={height},
    symbolic x coords={{{sym}}}, xtick=data,
    {xtick_style}
    ymin=0, ymax={ymax:.1f}, ytick=\\empty, axis y line=none,
    axis x line*=bottom,
    enlarge x limits={{abs=0.6cm}},
    legend style={{font=\\scriptsize, at={{(1,1.04)}}, anchor=south east, draw=none, fill=none}},
    legend columns=2,
    legend image code/.code={{\\draw[#1] (0cm,-0.06cm) rectangle (0.18cm,0.12cm);}},
  ]
    {plots}
    {nodes_tex}
  \\end{{axis}}
\\end{{tikzpicture}}"""


def slider_chart(counts_by_group: dict[str, dict], *, width: str, height: str,
                 label_mode: str, show_legend: bool,
                 y_ceiling: int | None = None) -> str:
    """TLX histogram on a true NUMERIC 0-100 axis: grouped bars at the bin
    centres (5, 15, ..., 95), ticks at the bin edges 0,10,...,100 — so the
    100 label exists and the plain number labels hug the axis (no wrapped
    text-width box like the likert footer labels need)."""
    bins = [(f"{lo}--{lo + 9}", lo + 5) for lo in range(0, 90, 10)] + \
           [("90--100", 95)]
    a = [int(counts_by_group.get(GROUP_ADHD, {}).get(k, 0)) for k, _ in bins]
    c = [int(counts_by_group.get(GROUP_CONTROL, {}).get(k, 0)) for k, _ in bins]
    ymax = _ymax(max(a + c), y_ceiling)
    coords_a = " ".join(f"({x},{v})" for (_, x), v in zip(bins, a))
    coords_c = " ".join(f"({x},{v})" for (_, x), v in zip(bins, c))

    if label_mode == "footer":
        xtick_style = "x tick label style={font=\\scriptsize\\bfseries},"
    else:
        xtick_style = "xticklabels={},"

    nodes = []
    for (_, x), av, cv in zip(bins, a, c):
        if av > 0:
            nodes.append(f"\\node[font=\\scriptsize, above, xshift=-5pt] at (axis cs:{x},{av}) {{{av}}};")
        if cv > 0:
            nodes.append(f"\\node[font=\\scriptsize, above, xshift=5pt] at (axis cs:{x},{cv}) {{{cv}}};")
    nodes_tex = "\n    ".join(nodes)
    legend_cmd = "\n    \\legend{ADHD, No-ADHD}" if show_legend else ""

    return f"""\\begin{{tikzpicture}}[trim axis left, trim axis right]
  \\begin{{axis}}[
    ybar=2pt, bar width=8pt, scale only axis, width={width}, height={height},
    xmin=-2, xmax=102, xtick={{0,10,...,100}},
    {xtick_style}
    ymin=0, ymax={ymax:.1f}, ytick=\\empty, axis y line=none,
    axis x line*=bottom,
    legend style={{font=\\scriptsize, at={{(1,1.04)}}, anchor=south east, draw=none, fill=none}},
    legend columns=2,
    legend image code/.code={{\\draw[#1] (0cm,-0.06cm) rectangle (0.18cm,0.12cm);}},
  ]
    \\addplot[ybar, fill={ADHD_COLOR}, draw={ADHD_COLOR}!70!black] coordinates {{{coords_a}}};
    \\addplot[ybar, fill={CTRL_COLOR}, draw={CTRL_COLOR}!70!black] coordinates {{{coords_c}}};{legend_cmd}
    {nodes_tex}
  \\end{{axis}}
\\end{{tikzpicture}}"""


def question_block(qnum_label: str, qtext: str, chart: str) -> str:
    """Question text (left) next to its chart (right)."""
    return f"""\\noindent
\\begin{{minipage}}[c]{{0.40\\textwidth}}
  \\small \\textbf{{{esc(qnum_label)}}}\\quad {esc(qtext)}
\\end{{minipage}}\\hfill
\\begin{{minipage}}[c]{{0.58\\textwidth}}
  {chart}
\\end{{minipage}}
\\par\\vspace{{1em}}
"""


def strip_stem(text: str) -> str:
    """Qualtrics matrix items export as 'Common stem - Statement'; keep the
    statement. TLX exports as 'Name - Description'; keep the name."""
    if " - " in text:
        parts = text.split(" - ")
        # For matrix items the statement is the LAST part; guard against
        # hyphens inside the statement by joining everything after the stem.
        return " - ".join(parts[1:]).strip()
    return text.strip()


def summary_stats_table(values_by_group: dict[str, pd.Series], caption: str) -> str:
    rows = []
    order = [GROUP_ADHD, GROUP_CONTROL, "Overall"]
    combined = pd.concat([v for v in values_by_group.values()], ignore_index=True) \
        if values_by_group else pd.Series(dtype=float)
    for name in order:
        s = combined if name == "Overall" else values_by_group.get(name, pd.Series(dtype=float))
        s = pd.to_numeric(s, errors="coerce").dropna()
        if len(s) == 0:
            rows.append(f"{name} & 0 & -- & -- & -- & -- & -- & -- & -- \\\\")
            continue
        sd = s.std(ddof=1) if len(s) > 1 else float("nan")
        sd_s = f"{sd:.2f}" if not np.isnan(sd) else "--"
        rows.append(
            f"{name} & {len(s)} & {s.min():.0f} & {s.quantile(.25):.1f} & "
            f"{s.median():.1f} & {s.mean():.2f} & {s.quantile(.75):.1f} & "
            f"{s.max():.0f} & {sd_s} \\\\")
    body = "\n".join(rows)
    return f"""\\begin{{center}}\\small
\\begin{{tabular}}{{lrrrrrrrr}}
\\toprule
 & $n$ & Min & $Q_1$ & Median & Mean & $Q_3$ & Max & SD \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}\\\\[2pt]
{{\\footnotesize {esc(caption)}}}
\\end{{center}}
"""


def category_table(df, col, groups, cats) -> str:
    """Simple categorical count table (rows = answer categories, columns =
    ADHD / Control / Overall) — for demographics where a chart is overkill."""
    counts = counts_for(df, col, groups)
    rows = []
    for v, lbl in cats:
        a_n = int(counts[GROUP_ADHD].get(v, 0))
        c_n = int(counts[GROUP_CONTROL].get(v, 0))
        rows.append(f"{esc(lbl)} & {a_n} & {c_n} & {a_n + c_n} \\\\")
    body = "\n".join(rows)
    return f"""\\begin{{center}}\\small
\\begin{{tabular}}{{lrrr}}
\\toprule
 & ADHD & No-ADHD & Overall \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{center}}
"""


def text_answers_table(answers_by_group: dict[str, list[tuple[str, str]]],
                       title: str, header: str | None = None) -> str:
    """Stacked full-width layout (decided 30.08): the ADHD block first,
    then the no-ADHD block below it, each a plain flowing stream of
    answers with uniform 5pt spacing. Ordinary page breaking applies
    everywhere, so this needs no minipages, no paracol and no height
    estimation (all earlier two-column variants retired). Group identity
    comes from the chart colour chips; blocks never waste space on
    cross-column height matching."""
    def block(items: list[tuple[str, str]], color: str, gname: str) -> str:
        body = "\n\\par\\vspace{5pt}\\noindent\n".join(
            f"\\textbf{{P{esc(disp_pid(pid))}:}} {esc(text)}"
            for pid, text in items) or "---"
        return (f"\\noindent\\textcolor{{{color}}}{{\\rule{{1.2ex}}{{1.2ex}}}}"
                f"~\\textbf{{{gname}}}\\par\\nopagebreak\\vspace{{4pt}}"
                f"\\noindent\n{body}")

    head = (f"\\textbf{{{esc(header)}}}; \\textit{{{esc(title)}}}" if header
            else f"\\textit{{{esc(title)}}}")
    a = block(answers_by_group.get(GROUP_ADHD, []), ADHD_COLOR, "ADHD")
    c = block(answers_by_group.get(GROUP_CONTROL, []), CTRL_COLOR, "No-ADHD")
    return f"""\\noindent{head}\\par\\nopagebreak\\vspace{{0.3em}}
{{\\small
\\noindent\\rule{{\\linewidth}}{{0.6pt}}\\par\\nopagebreak\\vspace{{6pt}}
{a}
\\par\\vspace{{10pt}}
{c}
\\par\\nopagebreak\\vspace{{5pt}}
\\noindent\\rule{{\\linewidth}}{{0.4pt}}}}
\\par\\vspace{{1.0em}}
"""


def counts_for(df: pd.DataFrame, col: str, groups: dict[str, str]) -> dict:
    out = {GROUP_ADHD: {}, GROUP_CONTROL: {}}
    for _, row in df.iterrows():
        g = groups.get(row["PID"])
        v = row.get(col)
        if g is None or v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "":
            continue
        v = _norm_value(v)
        out[g][v] = out[g].get(v, 0) + 1
    return out


def _norm_value(v):
    """Normalise a cell to either an int code or a stripped label."""
    s = str(v).strip()
    try:
        f = float(s)
        return int(f) if f.is_integer() else f
    except ValueError:
        return s


def observed_values(df: pd.DataFrame, col: str) -> list:
    vals = df[col].dropna().map(_norm_value)
    return [v for v in vals.unique().tolist() if str(v).strip() != ""]


def bin_slider(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Bin 0-100 slider values into 10-point bins, stored in a helper column."""
    df = df.copy()
    num = pd.to_numeric(df[col], errors="coerce")
    edges = list(range(0, 101, 10))
    labels = [f"{lo}--{lo + 9}" for lo in range(0, 90, 10)] + ["90--100"]
    df[col + "_BIN"] = pd.cut(num, bins=[-0.5] + [e - 0.5 for e in edges[1:-1]] + [100.5],
                              labels=labels)
    df[col + "_BIN"] = df[col + "_BIN"].astype(str).replace("nan", np.nan)
    return df


# ============================================================================
# Instrument-level builders
# ============================================================================

class Paginator:
    """Tracks the vertical slot position on the current page so consecutive
    chart instruments FLOW into one another (no forced page break between
    them) while the answer-scale labels still land where they are needed:
    below the last chart on every page AND below the last chart of every
    instrument (scales differ between instruments). Page breaks are emitted
    explicitly every LIKERT_ROWS_PER_PAGE slots; an instrument heading
    consumes one slot (conservative — keeps pages from overflowing)."""

    def __init__(self, rows_per_page: int = LIKERT_ROWS_PER_PAGE):
        self.rows = rows_per_page
        self.pos = 0  # slots used on the current page

    def heading(self) -> str:
        """Prefix to emit before an instrument heading: a page break if
        the heading would be orphaned at the very bottom of the page."""
        if self.pos >= self.rows - 1:
            self.pos = 0
            return "\\newpage\n"
        self.pos += 1
        return ""

    def row(self, first_in_instrument: bool, last_in_instrument: bool):
        """Returns (prefix, footer_labels, show_legend) for the next chart."""
        prefix = ""
        if self.pos >= self.rows:
            prefix = "\\newpage\n"
            self.pos = 0
        first_on_page = (self.pos == 0)
        self.pos += 1
        last_on_page = (self.pos == self.rows)
        return (prefix,
                last_on_page or last_in_instrument,
                first_on_page or first_in_instrument)

    def force_break(self):
        """Call after non-chart material (e.g. stats tables) so the next
        chart starts on a fresh page with a known slot position."""
        self.pos = self.rows


def _fmt_pair(a, c, dec: int = 2) -> str:
    """'ADHD | Control' cell: two values joined by a vertical bar."""
    def one(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "--"
        return f"{v:.{dec}f}" if dec else f"{v:g}"
    return f"{one(a)}\\,$|$\\,{one(c)}"


def likert_summary_table(df, qtext, groups, prefix, n_items, title,
                         hint=None) -> str:
    """One row per item: per-value response counts, mean, median and SD,
    each cell as 'ADHD | Control', plus a per-item Mann-Whitney U.
    Replaces the per-question histogram charts (decided 29.08)."""
    pids_a = [p for p, g in groups.items() if g == GROUP_ADHD]
    n_a = len(pids_a)
    n_c = len(groups) - n_a
    items = [(i, f"{prefix}{i}") for i in range(1, n_items + 1)
             if f"{prefix}{i}" in df.columns
             or print(f"  WARNING: column {prefix}{i} missing — skipped.")]
    all_obs = sorted({v for _, col in items for v in observed_values(df, col)},
                     key=str)
    cats = category_order(all_obs, hint)
    codes = [v for v, _ in cats if isinstance(v, (int, float))]
    if not codes:  # choice-text export: ranks 1..k in category order
        codes = list(range(1, len(cats) + 1))
    labels = [str(lbl) for _, lbl in cats]
    scale_note = "; ".join(f"{c:g} = {esc(l)}" for c, l in zip(codes, labels)
                           if str(c) != str(l))
    rows = []
    for pos, (i, col) in enumerate(items):
        ranks = to_rank(df[col], hint)
        a = ranks[df["PID"].isin(pids_a)].dropna()
        c = ranks[~df["PID"].isin(pids_a)].dropna()
        if len(a) >= 2 and len(c) >= 2:
            u = scistats.mannwhitneyu(a, c, alternative="two-sided")
            p_cell = _p_val(_fmt_p(u.pvalue))
        else:
            p_cell = "--"
        label = strip_stem(qtext.get(col, col))
        if len(label) > 78:
            label = label[:77].rstrip() + "…"
        counts = " & ".join(
            f"{int((a == v).sum())}\\,$|$\\,{int((c == v).sum())}"
            for v in codes)
        shade = "\\rowcolor{gray!8} " if pos % 2 == 0 else ""
        # Fixed two-line, vertically centred item box: every row gets the
        # same height and the value cells sit at the row's vertical middle.
        # NB height must be em-based — \baselineskip is 0 inside table cells.
        item_box = (f"\\parbox[c][3.1em][c]{{6.3cm}}"
                    f"{{Q{i}: {esc(label)}}}")
        rows.append(
            f"{shade}{item_box} & {counts} & "
            f"{_fmt_pair(a.mean(), c.mean())} & "
            f"{_fmt_pair(a.median(), c.median(), dec=1)} & "
            f"{_fmt_pair(a.std(ddof=1), c.std(ddof=1))} & "
            f"{p_cell} \\\\")
    k = len(codes)
    code_heads = " & ".join(f"{v:g}" for v in codes)
    # legend reduced to the coding only (decided 05.09); the n's live in
    # the header line and the cell semantics in the authored caption.
    note = (f"\\noindent{{\\small Coding: {scale_note}.}}"
            f"\\par\\vspace{{0.4em}}\n" if scale_note else "")
    body = "\n".join(rows)
    return f"""\\subsection*{{{esc(title)}}}
{note}{{\\small
\\setlength{{\\tabcolsep}}{{3.5pt}}
\\setlength{{\\LTleft}}{{0pt}}\\setlength{{\\LTright}}{{0pt}}
\\begin{{longtable}}{{@{{}}l@{{\\hspace{{6pt}}\\extracolsep{{\\fill}}}}{'c' * k}rrrr@{{}}}}
\\toprule
Item & {code_heads} & Mean & Md & SD & $p_U$ \\\\
 & \\multicolumn{{{k + 3}}}{{c}}{{ADHD ($n = {n_a}$)\\,$|$\\,no-ADHD ($n = {n_c}$)}} & \\\\
\\midrule
\\endhead
{body}
\\bottomrule
\\end{{longtable}}}}
"""


def slider_summary_table(df, qtext, groups, cols, title) -> str:
    """0-100 slider dimensions (NASA-TLX): one row per dimension with
    five-number summaries as 'ADHD | Control' pairs plus Mann-Whitney U."""
    pids_a = [p for p, g in groups.items() if g == GROUP_ADHD]
    n_a = len(pids_a)
    n_c = len(groups) - n_a
    rows = []
    present = [c for c in cols if c in df.columns
               or print(f"  WARNING: column {c} missing — skipped.")]
    for pos, col in enumerate(present):
        v = pd.to_numeric(df[col], errors="coerce")
        a = v[df["PID"].isin(pids_a)].dropna()
        c = v[~df["PID"].isin(pids_a)].dropna()
        if len(a) >= 2 and len(c) >= 2:
            u = scistats.mannwhitneyu(a, c, alternative="two-sided")
            p_cell = _p_val(_fmt_p(u.pvalue))
        else:
            p_cell = "--"
        dim = esc(qtext.get(col, col).split(" - ")[0])
        shade = "\\rowcolor{gray!8} " if pos % 2 == 0 else ""
        cells = " & ".join(
            _fmt_pair(fa, fc, dec=0) for fa, fc in (
                (a.min(), c.min()), (a.quantile(.25), c.quantile(.25)),
                (a.median(), c.median()), (a.quantile(.75), c.quantile(.75)),
                (a.max(), c.max())))
        rows.append(f"{shade}{dim} & {cells} & {p_cell} \\\\")
    note = "0--100 scale."
    body = "\n".join(rows)
    return f"""\\subsection*{{{esc(title)}}}
\\noindent{{\\small {note}}}\\par\\vspace{{0.4em}}
\\noindent{{\\small
\\setlength{{\\tabcolsep}}{{4pt}}%
\\begin{{tabular*}}{{\\textwidth}}{{@{{}}l@{{\\extracolsep{{\\fill}}}} rrrrr r@{{}}}}
\\toprule
Dimension & Min & $Q_1$ & Median & $Q_3$ & Max & $p_U$ \\\\
 & \\multicolumn{{5}}{{c}}{{ADHD ($n = {n_a}$)\\,$|$\\,no-ADHD ($n = {n_c}$)}} & \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular*}}}}
\\par\\vspace{{0.6em}}
"""


def likert_instrument(df, qtext, groups, prefix, n_items, title, hint=None,
                      with_stats=False, pag: Paginator | None = None):
    """A block of question rows (text + grouped chart), optionally followed
    by per-question summary-statistics tables (ranks 1..k). Pagination and
    label placement are delegated to `pag` (see Paginator)."""
    pag = pag or Paginator()
    parts = [pag.heading() + f"\\subsection*{{{esc(title)}}}\n"]
    stats_parts = []
    items = []
    for i in range(1, n_items + 1):
        col = f"{prefix}{i}"
        if col not in df.columns:
            print(f"  WARNING: column {col} missing — skipped.")
            continue
        items.append((i, col))
    # One y-scale for the whole instrument: tallest bar over all its charts.
    per_item = [(i, col, category_order(observed_values(df, col), hint),
                 counts_for(df, col, groups)) for i, col in items]
    ceiling = max((n for _, _, _, cnt in per_item
                   for grp in cnt.values() for n in grp.values()), default=1)
    for pos, (i, col, cats, counts) in enumerate(per_item):
        pre, footer, legend = pag.row(pos == 0, pos == len(per_item) - 1)
        if pre:
            parts.append(pre)
        chart = stacked_chart(
            counts, cats, width="\\linewidth", height=LIKERT_CHART_HEIGHT,
            label_mode="footer" if footer else "none", show_legend=legend,
            y_ceiling=ceiling)
        parts.append(question_block(f"Q{i}", strip_stem(qtext.get(col, col)), chart))
        if with_stats:
            vals = {}
            for g in (GROUP_ADHD, GROUP_CONTROL):
                pids = [p for p, gg in groups.items() if gg == g]
                vals[g] = to_rank(df[df["PID"].isin(pids)][col], hint)
            codes = [v for v, _ in cats if isinstance(v, (int, float))]
            rng = (f"{min(codes):g}--{max(codes):g}" if codes
                   else f"1--{len(cats)}")
            stats_parts.append(summary_stats_table(
                vals, f"Q{i} — {strip_stem(qtext.get(col, col))} "
                      f"(responses coded {rng})"))
    if stats_parts:
        parts.append("\\subsubsection*{Summary statistics}\n")
        parts.extend(stats_parts)
        pag.force_break()
    return "\n".join(parts)


def slider_instrument(df, qtext, groups, cols, title, with_stats=False,
                      pag: Paginator | None = None):
    """TLX-style 0-100 sliders: binned grouped histogram per dimension, with
    the same Paginator-driven flow and label placement as the likert
    instruments (labels = bin edges 0..100 in steps of ten)."""
    pag = pag or Paginator()
    parts = [pag.heading() + f"\\subsection*{{{esc(title)}}}\n"]
    stats_parts = []
    present = [c for c in cols if c in df.columns]
    for c in cols:
        if c not in present:
            print(f"  WARNING: column {c} missing — skipped.")
    per_col = [(col, counts_for(bin_slider(df, col), col + "_BIN", groups))
               for col in present]
    ceiling = max((n for _, cnt in per_col
                   for grp in cnt.values() for n in grp.values()), default=1)
    for pos, (col, counts) in enumerate(per_col):
        pre, footer, legend = pag.row(pos == 0, pos == len(per_col) - 1)
        if pre:
            parts.append(pre)
        chart = slider_chart(
            counts, width="\\linewidth", height=LIKERT_CHART_HEIGHT,
            label_mode="footer" if footer else "none", show_legend=legend,
            y_ceiling=ceiling)
        dim = qtext.get(col, col).split(" - ")[0]
        parts.append(question_block(dim, "", chart))
        if with_stats:
            vals = {}
            for g in (GROUP_ADHD, GROUP_CONTROL):
                pids = [p for p, gg in groups.items() if gg == g]
                vals[g] = pd.to_numeric(df[df["PID"].isin(pids)][col], errors="coerce")
            stats_parts.append(summary_stats_table(vals, f"{dim} (0--100)"))
    if stats_parts:
        parts.append("\\subsubsection*{Summary statistics}\n")
        parts.extend(stats_parts)
        pag.force_break()
    return "\n".join(parts)


def open_ended_tables(df, qtext, groups, cols, title):
    """Open-ended sections in ordinary PORTRAIT flow (landscape retired
    30.08 with the stacked full-width layout — see text_answers_table).
    Each question gets a short bold subheader from OE_HEADERS (skipped if
    it would just repeat the section title), with the full question text
    below it."""
    parts = [f"\\subsection*{{{esc(title)}}}\n"]
    wrote_any = False
    for col in cols:
        if col not in df.columns:
            continue
        answers = {GROUP_ADHD: [], GROUP_CONTROL: []}
        for _, row in df.iterrows():
            g = groups.get(row["PID"])
            v = row.get(col)
            if g and isinstance(v, str) and v.strip():
                answers[g].append((row["PID"], v.strip()))
        if not (answers[GROUP_ADHD] or answers[GROUP_CONTROL]):
            continue
        header = OE_HEADERS.get(col)
        if header and header.lower() == title.lower():
            header = None  # would just repeat the section title
        parts.append(text_answers_table(answers, qtext.get(col, col),
                                        header=header))
        wrote_any = True
    if not wrote_any:
        return ""
    return "\n".join(parts)


# ============================================================================
# Section builders
# ============================================================================

def build_pre_study(pre, qtext, groups, src):
    out = [header_comment([src])]
    pag = Paginator()
    n_a = sum(1 for g in groups.values() if g == GROUP_ADHD)
    n_c = len(groups) - n_a
    method = {
        "diagnosis": "formal ADHD diagnosis (self-reported)",
        "asrs": f"ASRS ({ASRS_METRIC.replace('_', ' ')} $>$ {ASRS_THRESHOLD})",
        "file": f"ASRS screening (threshold {ASRS_THRESHOLD}; see Methods)",
    }[GROUPING]
    out.append(f"\\noindent Group assignment based on {method}: "
               f"$n_{{\\text{{ADHD}}}} = {n_a}$, $n_{{\\text{{No-ADHD}}}} = {n_c}$, "
               f"$N = {len(groups)}$.\\par\\vspace{{1em}}\n")

    # AGE
    out.append("\\subsection*{Age}\n")
    vals = {}
    for g in (GROUP_ADHD, GROUP_CONTROL):
        pids = [p for p, gg in groups.items() if gg == g]
        vals[g] = pd.to_numeric(pre[pre["PID"].isin(pids)]["PRE_AGE"], errors="coerce")
    out.append(summary_stats_table(vals, "Age in years."))

    # GENDER
    out.append("\\subsection*{Gender}\n")
    cats = category_order(observed_values(pre, "PRE_GENDER"), "GENDER")
    out.append(category_table(pre, "PRE_GENDER", groups, cats))
    selfdesc = pre["PRE_GENDER_4_TEXT"].dropna() if "PRE_GENDER_4_TEXT" in pre else []
    if len(selfdesc):
        out.append("\\noindent\\footnotesize Self-described: " +
                   "; ".join(esc(v) for v in selfdesc) + "\\normalsize\\par\n")

    # DEGREE
    deg = {GROUP_ADHD: [], GROUP_CONTROL: []}
    for _, r in pre.iterrows():
        g = groups.get(r["PID"])
        if g and isinstance(r.get("PRE_DEGREE"), str) and r["PRE_DEGREE"].strip():
            deg[g].append((r["PID"], r["PRE_DEGREE"].strip()))
    out.append("\\subsection*{Degree programme}\n")
    out.append(text_answers_table(deg, "Degree programme (verbatim)."))

    # LEVEL
    out.append("\\subsection*{Level of study}\n")
    cats = category_order(observed_values(pre, "PRE_LEVEL"), "LEVEL")
    out.append(category_table(pre, "PRE_LEVEL", groups, cats))

    # FIELD
    fld = {GROUP_ADHD: [], GROUP_CONTROL: []}
    for _, r in pre.iterrows():
        g = groups.get(r["PID"])
        if g and isinstance(r.get("PRE_FIELD"), str) and r["PRE_FIELD"].strip():
            fld[g].append((r["PID"], r["PRE_FIELD"].strip()))
    out.append("\\subsection*{Field of study}\n")
    out.append(text_answers_table(fld, "Field of study (verbatim)."))

    # DIAGNOSIS + SUPPORT (simple count tables)
    for col, ttl in [("PRE_ADHD_DX", "Formal ADHD diagnosis"),
                     ("PRE_ADHD_SUPPORT", "Support / accommodations")]:
        out.append(f"\\subsection*{{{ttl}}}\n")
        cats = category_order(observed_values(pre, col), "YESNO")
        out.append(category_table(pre, col, groups, cats))

    # Demographics above fill part of the page -> chart instruments start fresh.
    out.append("\\newpage\n")
    out.append(likert_summary_table(pre, qtext, groups, "PRE_NARS_", 14,
                                    "Negative Attitudes towards Robots Scale (NARS)",
                                    hint="LIKERT5"))
    out.append(likert_summary_table(pre, qtext, groups, "PRE_ASRS_", 18,
                                    "Adult ADHD Self-Report Scale (ASRS)",
                                    hint="ASRS"))
    out.append(likert_summary_table(pre, qtext, groups, "PRE_ESQR_", 25,
                                    "Executive Skills Questionnaire Revised (ESQ-R)",
                                    hint="ESQR"))
    return "\n".join(out)


def build_post_control(ctrl, qtext, groups, src):
    out = [header_comment([src])]
    pag = Paginator()
    out.append(open_ended_tables(ctrl, qtext, groups, ["POST_OE_01"],
                                 "Task description"))
    tlx_cols = ["POST_TLX_MENTAL_1", "POST_TLX_PHYSICAL_1", "POST_TLX_TEMPORAL_1",
                "POST_TLX_PERFORMANCE_1", "POST_TLX_EFFORT_1", "POST_TLX_FRUSTRATION_1"]
    out.append(slider_summary_table(ctrl, qtext, groups, tlx_cols,
                                     "NASA Task Load Index (TLX) — no-robot session"))
    return "\n".join(out)


def build_post_robot(post, qtext, groups, src):
    out = [header_comment([src])]
    pag = Paginator()
    out.append(likert_summary_table(post, qtext, groups, "POST_NARS_", 14,
                                    "Negative Attitudes towards Robots Scale (NARS)",
                                    hint="LIKERT5"))
    out.append(likert_summary_table(post, qtext, groups, "POST_SUS_", 10,
                                    "System Usability Scale (SUS)", hint="LIKERT5"))
    tlx_cols = ["POST_TLX_MENTAL_1", "POST_TLX_PHYSICAL_1", "POST_TLX_TEMPORAL_1",
                "POST_TLX_PERFORMANCE_1", "POST_TLX_EFFORT_1", "POST_TLX_FRUSTRATION_1"]
    out.append(slider_summary_table(post, qtext, groups, tlx_cols,
                                     "NASA Task Load Index (TLX) — robot session"))
    out.append(likert_summary_table(post, qtext, groups, "POST_FEAT_PRESENCE_", 3,
                                    "Body doubling", hint="LIKERT5"))
    out.append(likert_summary_table(post, qtext, groups, "POST_FEAT_INATT_", 5,
                                    "Inattention detection", hint="LIKERT5"))
    out.append(likert_summary_table(post, qtext, groups, "POST_FEAT_CONTEXT_", 4,
                                    "Task-aware support", hint="LIKERT5"))
    out.append(likert_summary_table(post, qtext, groups, "POST_FEAT_OVERALL_", 3,
                                    "Overall experience", hint="LIKERT5"))
    oe_cols = [f"POST_OE_{i:02d}" for i in range(1, 13)]
    out.append(open_ended_tables(post, qtext, groups, oe_cols,
                                 "Open-ended questions"))
    return "\n".join(out)


# ============================================================================
# Main
# ============================================================================


# ============================================================================
# Session-log timelines (appendix section "Session log results")
# ============================================================================

SESSION_LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
TIMELINE_ROW_H_CM = 0.85
SESSION_MAX_MIN = 45.0
_BAND_LO, _BAND_HI = 0.06, 0.94   # score band inside each participant row
_NEG_CLASSES = ("angry", "disgust", "fear", "sad")


def _session_csv_dir(pid: int, cond: str) -> str | None:
    hits = sorted(glob.glob(os.path.join(SESSION_LOGS_DIR, f"P{pid}_{cond}_*_csv")))
    return hits[-1] if hits else None


def _read_rows(dirpath: str, name: str) -> list[dict]:
    path = os.path.join(dirpath, name)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _t_min(row: dict, field: str = "t_session_s") -> float | None:
    raw = row.get(field) or ""
    if not raw:
        return None
    t = float(raw)
    if t < 0 or t > SESSION_MAX_MIN * 60:
        return None
    return t / 60.0


def _score_series(rows: list[dict], value) -> list[list[tuple[float, float]]]:
    """(t_min, value) points split into segments at >30s gaps, so sensing
    outages render as gaps instead of interpolated bridges. `value` maps a
    row to a float or None."""
    pts = []
    for r in rows:
        t = _t_min(r)
        if t is None:
            continue
        v = value(r)
        if v is None:
            continue
        pts.append((t, v))
    segs, cur = [], []
    for t, v in pts:
        if cur and t - cur[-1][0] > 0.5:
            segs.append(cur)
            cur = []
        cur.append((t, v))
    if cur:
        segs.append(cur)
    return segs


def _event_times(dirpath: str, kinds: set[str]) -> list[float]:
    out = []
    for r in _read_rows(dirpath, "events.csv"):
        if r.get("event_type") in kinds:
            t = _t_min(r)
            if t is not None:
                out.append(t)
    return out


def _float_or_none(row: dict, field: str) -> float | None:
    raw = row.get(field) or ""
    return float(raw) if raw else None


def _neg_mass(row: dict) -> float | None:
    if not (row.get("angry") or ""):
        return None
    return sum(float(row.get(c) or 0.0) for c in _NEG_CLASSES)


def _timeline_axis(rows_labeled: list[tuple[float, str]], ymax: float,
                   headers: list[tuple[str, float]], body: str) -> str:
    """Shared axis frame: one row per participant plus bold group headers
    in the label gutter."""
    yticks = ",".join(f"{b + 0.5:.2f}" for b, _ in rows_labeled)
    ylab = ",".join("{" + l + "}" for _, l in rows_labeled)
    header_nodes = "\n".join(
        f"    \\node[anchor=south east, font=\\small\\bfseries] "
        f"at (axis cs:0,{y:.2f}) {{{lbl}}};" for lbl, y in headers)
    body = body + "\n" + header_nodes
    h = TIMELINE_ROW_H_CM * ymax
    return f"""\\begin{{center}}\\begin{{tikzpicture}}
  \\begin{{axis}}[
    width=0.86\\textwidth, height={h:.1f}cm, scale only axis,
    xmin=0, xmax={SESSION_MAX_MIN:.0f}, ymin=0, ymax={ymax:.2f},
    xtick={{0,5,...,45}}, xlabel={{Session time (minutes)}},
    x tick label style={{font=\\small}}, xlabel style={{font=\\small}},
    ytick={{{yticks}}}, yticklabels={{{ylab}}},
    y tick label style={{font=\\small}}, ytick style={{draw=none}},
    axis x line*=bottom, axis y line*=left, clip=false,
  ]
{body}
  \\end{{axis}}
\\end{{tikzpicture}}\\end{{center}}
"""


_GROUP_GAP = 0.7  # spacer rows between the ADHD and control blocks


def _layout(sessions: list[tuple[int, str]], groups: dict[str, str]):
    """Row bases (top-down) with a gap between the ADHD and control blocks,
    plus (label, y) positions for the bold group headers that sit in the
    label gutter above each block's first row."""
    n_adhd = sum(1 for pid, _ in sessions
                 if groups.get(str(pid)) == GROUP_ADHD)
    split = 0 < n_adhd < len(sessions)
    gap = _GROUP_GAP if split else 0.0
    ymax = len(sessions) + gap
    rows = []
    for i, (pid, d) in enumerate(sessions):
        base = ymax - (i + 1) - (gap if i >= n_adhd else 0.0)
        rows.append((pid, d, base))
    headers = [("ADHD", ymax + 0.05)]
    if split:
        # top edge of the first control row is (ymax - n_adhd - gap):
        # base = ymax - (n_adhd+1) - gap, +1 for the row height.
        headers.append(("Control", ymax - n_adhd - gap + 0.05))
    return rows, ymax, headers


def _trigger_lines(times: list[float], base: float, color: str) -> list[str]:
    return [f"    \\draw[{color}, line width=0.7pt] "
            f"(axis cs:{t:.2f},{base + _BAND_LO:.2f}) -- "
            f"(axis cs:{t:.2f},{base + _BAND_HI:.2f});"
            for t in times]


def _score_row_body(base: float, raw_segs, avg_segs, threshold: float,
                    trig_times: list[float], trig_color: str,
                    annotate01: bool) -> list[str]:
    lo, hi = base + _BAND_LO, base + _BAND_HI
    span = _BAND_HI - _BAND_LO

    def y(v: float) -> float:
        return base + _BAND_LO + max(0.0, min(1.0, v)) * span

    out = [f"    \\draw[densely dashed, gray!70, line width=0.4pt] "
           f"(axis cs:0,{y(threshold):.3f}) -- (axis cs:45,{y(threshold):.3f});"]
    for segs, style in ((raw_segs, "gray!55, line width=0.3pt"),
                        (avg_segs, "black, line width=0.6pt")):
        for seg in segs:
            if len(seg) < 2:
                continue
            coords = " ".join(f"({t:.2f},{y(v):.3f})" for t, v in seg)
            out.append(f"    \\addplot[no marks, {style}] coordinates {{{coords}}};")
    out.extend(_trigger_lines(trig_times, base, trig_color))
    if annotate01:
        out.append(f"    \\node[font=\\tiny, anchor=west, gray] at (axis cs:45.2,{lo:.2f}) {{0}};")
        out.append(f"    \\node[font=\\tiny, anchor=west, gray] at (axis cs:45.2,{hi:.2f}) {{1}};")
    return out


def _sessions(cond: str, groups: dict[str, str]) -> list[tuple[int, str]]:
    """Rows ordered ADHD group first, then control, ascending PID within."""
    ordered = sorted((int(p) for p in INCLUDE_PIDS),
                     key=lambda n: (groups.get(str(n)) != GROUP_ADHD, n))
    out = []
    for pid in ordered:
        d = _session_csv_dir(pid, cond)
        if d:
            out.append((pid, d))
        else:
            print(f"  WARNING: no {cond} session CSVs for P{pid} — row skipped.")
    return out


def _interaction_chart(groups: dict[str, str]) -> str:
    sessions = _sessions("Robot", groups)
    rows, ymax, headers = _layout(sessions, groups)
    body = []
    for pid, d, base in rows:
        lo, hi = base + _BAND_LO, base + _BAND_HI
        body.append(f"    \\fill[ApxSilence] (axis cs:0,{lo:.2f}) "
                    f"rectangle (axis cs:45,{hi:.2f});")
        speech = _read_rows(d, "speech.csv")
        # robot first, user on top (user has priority where they overlap)
        for actor, color in (("robot", "ApxRobotSpeech"), ("user", "ApxUserSpeech")):
            for r in speech:
                if r.get("actor") != actor:
                    continue
                t0, t1 = _float_or_none(r, "t_start_s"), _float_or_none(r, "t_end_s")
                if t0 is None or t1 is None:
                    continue
                t0 = max(0.0, t0) / 60.0
                t1 = min(SESSION_MAX_MIN * 60, t1) / 60.0
                if t1 <= t0:
                    continue
                body.append(f"    \\fill[{color}] (axis cs:{t0:.2f},{lo:.2f}) "
                            f"rectangle (axis cs:{t1:.2f},{hi:.2f});")
    labeled = [(base, f"P{disp_pid(pid)}") for pid, _, base in rows]
    return _timeline_axis(labeled, ymax, headers, "\n".join(body))


def _score_chart(groups: dict[str, str], cond: str, csv_name: str, raw_value,
                 avg_value, threshold: float, trig_kinds: set[str],
                 trig_color: str) -> str:
    sessions = _sessions(cond, groups)
    rows, ymax, headers = _layout(sessions, groups)
    body = []
    for i, (pid, d, base) in enumerate(rows):
        data = _read_rows(d, csv_name)
        body.extend(_score_row_body(
            base,
            _score_series(data, raw_value),
            _score_series(data, avg_value),
            threshold,
            _event_times(d, trig_kinds), trig_color,
            annotate01=(i == 0)))
    labeled = [(base, f"P{disp_pid(pid)}") for pid, _, base in rows]
    return _timeline_axis(labeled, ymax, headers, "\n".join(body))


# ============================================================================
# Session-log analysis (shared with compute_stats.py) + stats fragment
# ============================================================================

EPISODE_CENSOR_GAP_S = 30.0  # inter-poll gap that censors an open episode
# A raw engagement score at time t is inferred from ~10 frames ending at t
# (~7-9 s of video at the effective recording rate), so a sample is treated
# as interaction-contaminated if any speech ended less than this many
# seconds before it (also absorbs post-speech head settling).
QUIET_BUFFER_S = 15.0
INTERACTION_COOLDOWN_S = 60.0  # mirror intervention_monitor.py
INTERVENTION_COOLDOWN_S = 60.0
REPLAY_MIN_SAMPLES = 3


def session_dirs() -> dict[tuple[str, str], str]:
    """(pid, 'Robot'|'Control') -> parsed-csv dir (newest if duplicated)."""
    out: dict[tuple[str, str], str] = {}
    for d in sorted(glob.glob(os.path.join(SESSION_LOGS_DIR, "P*_csv"))):
        parts = os.path.basename(d).split("_")
        out[(parts[0][1:], parts[1])] = d
    return out


def session_metrics(dirpath: str, cond: str) -> dict:
    ev = _read_rows(dirpath, "events.csv")

    def n(kind: str) -> int:
        return sum(1 for r in ev if r["event_type"] == kind)

    # User turns = maximal runs of consecutive transcript_user events. The
    # handler re-logs the growing user transcript on every ASR partial
    # update (one utterance -> several transcript_user events), so raw
    # event counts overcount by 1.2-12x per participant; run counts match
    # the independent turn_latency event counts within +-1 in every
    # session (verified 01.09.2026). A robot response landing mid-speech
    # splits the run — by definition the interruption starts a new turn.
    user_turns, prev = 0, None
    for r in ev:
        et = r["event_type"]
        if et in ("transcript_user", "transcript_assistant"):
            if et == "transcript_user" and prev != "transcript_user":
                user_turns += 1
            prev = et

    user_s = robot_s = 0.0
    for r in _read_rows(dirpath, "speech.csv"):
        if r["actor"] == "user":
            user_s += float(r["duration_s"] or 0)
        else:
            robot_s += float(r["duration_s"] or 0)
    eng = n("intervention_engagement_sent" if cond == "Robot"
            else "counterfactual_engagement")
    emo = n("intervention_emotion_sent" if cond == "Robot"
            else "counterfactual_emotion")
    return {
        # One robot_response event per completed assistant response
        # (conversational replies, interventions, context acks alike).
        "robot_turns": n("robot_response"),
        "user_turns": user_turns,
        "context": n("context_submit"),
        "robot_min": robot_s / 60, "user_min": user_s / 60,
        "int_eng": eng, "int_emo": emo, "int_tot": eng + emo,
    }


# Speech exclusion for the signal-level metrics (decided 05.09): samples
# during user/robot speech, plus this many seconds after each segment, are
# excluded from the mean-score and %-within-threshold metrics — a score
# integrates the preceding ~10 frames (measured 7.6s median, 10.9s in the
# slowest session), so post-speech samples still contain conversation
# footage. NB the slowest session's window slightly exceeds 10s; 15 would
# cover it fully (results are insensitive across 0-15, checked 05.09).
# The coverage gate stays on RAW polls (sensor health, not quietness);
# event-timing analyses (episodes' recovery, landmark, re-engagement)
# keep wall-clock semantics.
SPEECH_EXCLUSION_BUFFER_S = 10.0


def speech_exclusions(dirpath: str) -> list[tuple[float, float]]:
    """Merged [(start, end + buffer)] intervals of any speech, either
    actor. Empty for sessions without speech (all no-robot sessions)."""
    iv = sorted((float(r["t_start_s"]),
                 float(r["t_end_s"]) + SPEECH_EXCLUSION_BUFFER_S)
                for r in _read_rows(dirpath, "speech.csv")
                if r["t_start_s"])
    out: list[tuple[float, float]] = []
    for s0, s1 in iv:
        if out and s0 <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], s1))
        else:
            out.append((s0, s1))
    return out


def _excl_overlap(excl: list, lo: float, hi: float) -> float:
    return sum(max(0.0, min(e, hi) - max(s, lo)) for s, e in excl)


def _in_excl(excl: list, t: float) -> bool:
    return any(s0 <= t <= s1 for s0, s1 in excl)


def quiet_signal_mean(dirpath: str, sig: str,
                      excl: list | None = None) -> float | None:
    """Session mean of the raw signal value over speech-excluded samples."""
    if excl is None:
        excl = speech_exclusions(dirpath)
    csv, col = (("engagement.csv", "score") if sig == "eng"
                else ("emotion.csv", "negative_share"))
    vals = [float(r[col]) for r in _read_rows(dirpath, csv)
            if r.get(col) and r.get("t_session_s")
            and float(r["t_session_s"]) >= 0
            and not _in_excl(excl, float(r["t_session_s"]))]
    return sum(vals) / len(vals) if vals else None


def quiet_within_pct(dirpath: str, sigs: tuple, excl: list | None = None,
                     lo: float | None = None,
                     hi: float | None = None) -> float | None:
    """% of speech-excluded observed time within threshold(s), exact
    subtraction: episodes are extracted on the speech-excluded poll
    timeline, and both the observed-time denominator (inter-poll gaps
    clamped at the censor gap) and the below-threshold numerator subtract
    their overlap with the exclusion intervals. Multiple sigs = 'within
    both' (union of both signals' episodes over the merged timeline).
    lo/hi additionally clip to a window (for the session-half splits)."""
    if excl is None:
        excl = speech_exclusions(dirpath)
    if lo is not None:
        excl = excl + [(-1e9, lo), (hi, 1e9)]
    polls_by_sig = {}
    for sig in sigs:
        polls_by_sig[sig] = [(t, a) for t, a in signal_polls(dirpath, sig)
                             if not _in_excl(excl, t)]
    times = sorted(set(t for ps in polls_by_sig.values() for t, _ in ps))
    if len(times) < 2:
        return None
    span = sum((min(t2, t1 + EPISODE_CENSOR_GAP_S) - t1)
               - _excl_overlap(excl, t1, min(t2, t1 + EPISODE_CENSOR_GAP_S))
               for t1, t2 in zip(times, times[1:]))
    if span <= 0:
        return None
    ints = []
    for ps in polls_by_sig.values():
        for e in extract_episodes(ps):
            ints.append((e["t0"],
                         e["t1"] if e["t1"] is not None else e["t_last"]))
    merged: list[tuple[float, float]] = []
    for s0, s1 in sorted(ints):
        if merged and s0 <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], s1))
        else:
            merged.append((s0, s1))
    below = sum((s1 - s0) - _excl_overlap(excl, s0, s1)
                for s0, s1 in merged)
    return 100.0 * (1 - below / span)


def signal_polls(dirpath: str, which: str) -> list[tuple[float, bool]]:
    """(t_session_s, signal_active) for value-bearing polls, in time order.
    'eng': rolling average < threshold; 'emo': windowed negative share >
    threshold — i.e. exactly the signal each monitor gates on."""
    rows = _read_rows(dirpath, "engagement.csv" if which == "eng" else "emotion.csv")
    out = []
    for r in rows:
        t_raw = r.get("t_session_s") or ""
        val = r.get("average" if which == "eng" else "negative_share") or ""
        thr = r.get("threshold") or ""
        if not t_raw or not val or not thr:
            continue
        t = float(t_raw)
        if t < 0:
            continue
        active = (float(val) < float(thr)) if which == "eng" else (float(val) > float(thr))
        out.append((t, active))
    return out


SIGNAL_COVERAGE_MIN = 0.80  # per-(session, signal) validity gate


def signal_coverage(dirpath: str, which: str) -> float:
    """Fraction of the nominal session length covered by value-bearing
    polls of one signal, with inter-poll gaps clamped at
    EPISODE_CENSOR_GAP_S so sensing outages count as uncovered. Emotion
    polls without a detected face carry no value and add no coverage, so
    a mostly-noface session fails the gate too."""
    polls = signal_polls(dirpath, which)
    if len(polls) < 2:
        return 0.0
    cov = sum(min(t2 - t1, EPISODE_CENSOR_GAP_S)
              for (t1, _), (t2, _) in zip(polls, polls[1:]))
    return cov / (SESSION_MAX_MIN * 60.0)


def gated_signals(sdirs: dict) -> dict[tuple[str, str, str], float]:
    """(pid, cond, sig) -> coverage for every signal series FAILING the
    SIGNAL_COVERAGE_MIN gate. These series are excluded from all
    signal-level analyses (episodes, %-time, camera check); descriptive
    intervention/counterfactual counts stay reported but carry a note."""
    out = {}
    for (pid, cond), d in sdirs.items():
        for sig in ("eng", "emo"):
            c = signal_coverage(d, sig)
            if c < SIGNAL_COVERAGE_MIN:
                out[(pid, cond, sig)] = c
    return out


def gate_note(groups: dict, sdirs: dict) -> str:
    """Human-readable summary of the coverage gate for fragment notes
    (display PIDs). Empty string if nothing is excluded."""
    gated = gated_signals(sdirs)
    items = [f"P{disp_pid(pid)} "
             f"{'robot' if cond == 'Robot' else 'no-robot'} session "
             f"{'engagement' if sig == 'eng' else 'negative emotion'} "
             f"({100 * cov:.0f}\\%)"
             for (pid, cond, sig), cov in sorted(
                 gated.items(), key=lambda kv: (int(kv[0][0]), kv[0][1]))
             if pid in groups]
    if not items:
        return ""
    return ("Signal-coverage gate: a session's signal series enters the "
            "signal-level analyses only if its value-bearing polls cover "
            f"at least {SIGNAL_COVERAGE_MIN * 100:.0f}\\% of the session "
            "(inter-poll gaps above 30\\,s count as uncovered; emotion "
            "polls without a detected face carry no value). Excluded by "
            "this rule: " + "; ".join(items) + ".")


def extract_episodes(polls: list[tuple[float, bool]], min_polls: int = 2) -> list[dict]:
    """Group consecutive signal-active polls into episodes. t1 = first poll
    back at threshold (None = censored by a sensing gap or series end).
    Episodes with fewer than min_polls active polls are flicker, dropped."""
    eps: list[dict] = []
    cur: dict | None = None
    prev_t: float | None = None
    for t, active in polls:
        if cur is not None and prev_t is not None and t - prev_t > EPISODE_CENSOR_GAP_S:
            eps.append({**cur, "t1": None})
            cur = None
        if active:
            if cur is None:
                cur = {"t0": t, "n": 1, "t_last": t}
            else:
                cur["n"] += 1
                cur["t_last"] = t
        elif cur is not None:
            eps.append({**cur, "t1": t})
            cur = None
        prev_t = t
    if cur is not None:
        eps.append({**cur, "t1": None})
    return [e for e in eps if e["n"] >= min_polls]


def intervention_utterance_durations(sdirs: dict) -> list[float]:
    """Duration of the robot utterance each sent intervention produced:
    first robot speech segment starting within 20 s of the send event."""
    durs = []
    for (pid, cond), d in sdirs.items():
        if cond != "Robot":
            continue
        speech = [(float(r["t_start_s"]), float(r["duration_s"]))
                  for r in _read_rows(d, "speech.csv")
                  if r["actor"] == "robot" and r["t_start_s"]]
        for r in _read_rows(d, "events.csv"):
            if r["event_type"] in ("intervention_engagement_sent",
                                   "intervention_emotion_sent") and r["t_session_s"]:
                t = float(r["t_session_s"])
                cand = [dur for ss, dur in speech if t <= ss <= t + 20]
                if cand:
                    durs.append(cand[0])
    return durs


def replay_counterfactuals(dirpath: str, speech_dur: float) -> int:
    """Re-run the intervention gate logic over a control session's poll
    series, as if each fire produced a robot utterance of speech_dur
    seconds: the speaking gate is closed during it and the interaction
    cooldown restarts at its END (matching the treatment app, where
    assistant audio keeps resetting the activity clock until playback
    ends). speech_dur=0 reproduces the deployed control behaviour
    ('cooldowns reset as if sent' at fire time)."""
    stream: list[tuple[float, str, bool]] = []
    for which in ("eng", "emo"):
        polls = signal_polls(dirpath, which)
        for i, (t, active) in enumerate(polls):
            enough = sum(1 for tt, _ in polls[max(0, i - 10):i + 1]
                         if t - 30.0 < tt <= t) >= REPLAY_MIN_SAMPLES
            stream.append((t, which, active and enough))
    last_fire = {"eng": -1e9, "emo": -1e9}
    busy_until = -1e9  # end of the (synthetic) utterance = activity-clock reset
    count = 0
    for t, which, fireable in sorted(stream):
        if not fireable:
            continue
        if t <= busy_until:  # robot would still be speaking
            continue
        if t - busy_until <= INTERACTION_COOLDOWN_S:
            continue
        if t - last_fire[which] <= INTERVENTION_COOLDOWN_S:
            continue
        count += 1
        last_fire[which] = t
        busy_until = t + speech_dur
    return count


def episode_records(groups: dict, sdirs: dict) -> tuple[pd.DataFrame, dict]:
    """One row per below-threshold episode across all sessions, plus the
    observed poll span (s) per (pid, cond, signal) for %-time metrics."""
    recs, spans = [], {}
    gated = gated_signals(sdirs)
    for (pid, cond), d in sdirs.items():
        if pid not in groups:
            continue
        events = _read_rows(d, "events.csv")
        speech = [(float(r["t_start_s"]), float(r["t_end_s"]))
                  for r in _read_rows(d, "speech.csv")
                  if r["actor"] == "robot" and r["t_start_s"]]
        for which in ("eng", "emo"):
            if (pid, cond, which) in gated:  # coverage below the gate
                continue
            polls = signal_polls(d, which)
            if len(polls) > 1:
                # Observed time = inter-poll gaps clamped at the censor
                # gap, so sensing outages do not inflate the denominator
                # of the %-time-past-threshold metric.
                spans[(pid, cond, which)] = sum(
                    min(t2 - t1, EPISODE_CENSOR_GAP_S)
                    for (t1, _), (t2, _) in zip(polls, polls[1:]))
            base = "engagement" if which == "eng" else "emotion"
            kind = (f"intervention_{base}" if cond == "Robot"
                    else f"counterfactual_{base}")
            cues = [float(r["t_session_s"]) for r in events
                    if r["event_type"] == kind and r["t_session_s"]]
            for e in extract_episodes(polls):
                end = e["t1"] if e["t1"] is not None else e["t_last"]
                cue_ts = [t for t in cues if e["t0"] - 5 <= t <= end]
                rec_cue_end = np.nan
                if cue_ts and cond == "Robot" and e["t1"] is not None:
                    segs = [se for ss, se in speech
                            if cue_ts[0] <= ss <= cue_ts[0] + 20]
                    if segs:
                        rec_cue_end = max(e["t1"] - segs[0], 0.0)
                recs.append({
                    "pid": pid, "cond": cond, "sig": which,
                    "dur": (e["t1"] - e["t0"]) if e["t1"] is not None else np.nan,
                    "low_bound": (e["t_last"] - e["t0"]) if e["t1"] is None else np.nan,
                    "censored": e["t1"] is None,
                    "cued": bool(cue_ts), "rec_cue_end": rec_cue_end,
                    "elapsed_at_cue": (max(cue_ts[0] - e["t0"], 0.0)
                                       if cue_ts else np.nan),
                    "remaining_after_cue": (max(e["t1"] - cue_ts[0], 0.0)
                                            if cue_ts and e["t1"] is not None
                                            else np.nan),
                })
    return pd.DataFrame(recs), spans


def landmark_pairs(sub: pd.DataFrame, landmark: str) -> tuple[pd.Series, pd.Series]:
    """Risk-set matched landmark comparison for one signal's robot episodes.
    For each cue-delivered episode: remaining time from the landmark ('start'
    of the cue, or 'end' of its utterance) paired with the median remaining
    time of the gate-suppressed episodes still ongoing at the same elapsed
    time (>=3 at risk required). Corrects the immortal-time bias of the
    naive delivered-vs-suppressed comparison."""
    sup = sub[~sub.cued].dur.dropna()
    deliv = sub[sub.cued].dropna(subset=["remaining_after_cue"])
    d, m = [], []
    for _, row in deliv.iterrows():
        if landmark == "start":
            elapsed, remaining = row.elapsed_at_cue, row.remaining_after_cue
        else:
            if pd.isna(row.rec_cue_end):
                continue
            elapsed = row.elapsed_at_cue + row.remaining_after_cue - row.rec_cue_end
            remaining = row.rec_cue_end
        at_risk = sup[sup > elapsed] - elapsed
        if len(at_risk) >= 3:
            d.append(remaining)
            m.append(at_risk.median())
    return pd.Series(d, dtype=float), pd.Series(m, dtype=float)


def quiet_engagement_medians(groups: dict, sdirs: dict) -> pd.DataFrame:
    """Per participant: median raw engagement score in the robot session
    (all samples, and interaction-quiet samples only) and in the control
    session. Quiet = no speech by either actor within QUIET_BUFFER_S
    before the sample."""
    rows = {}
    gated = gated_signals(sdirs)
    for (pid, cond), d in sdirs.items():
        if pid not in groups:
            continue
        if (pid, cond, "eng") in gated:  # coverage below the gate
            continue
        speech = [(float(r["t_start_s"]), float(r["t_end_s"]))
                  for r in _read_rows(d, "speech.csv") if r["t_start_s"]]
        allv, quiet = [], []
        for r in _read_rows(d, "engagement.csv"):
            if not (r.get("t_session_s") and r.get("score")):
                continue
            t, v = float(r["t_session_s"]), float(r["score"])
            allv.append(v)
            if not any(ss <= t and se >= t - QUIET_BUFFER_S for ss, se in speech):
                quiet.append(v)
        rec = rows.setdefault(pid, {})
        if cond == "Robot":
            rec["robot_all"] = np.median(allv) if allv else np.nan
            rec["robot_quiet"] = np.median(quiet) if quiet else np.nan
            rec["n_quiet"] = len(quiet)
        else:
            rec["control"] = np.median(allv) if allv else np.nan
    return pd.DataFrame.from_dict(rows, orient="index")


def frustration_mechanism(post: pd.DataFrame, ctrl: pd.DataFrame) -> pd.DataFrame:
    """Per participant: robot-session behaviour metrics from the parsed
    logs plus the robot-minus-control TLX frustration delta."""
    fr_r = post.set_index("PID")["POST_TLX_FRUSTRATION_1"].pipe(
        pd.to_numeric, errors="coerce")
    fr_c = ctrl.set_index("PID")["POST_TLX_FRUSTRATION_1"].pipe(
        pd.to_numeric, errors="coerce")
    rows = []
    for (pid, cond), d in sorted(session_dirs().items()):
        if cond != "Robot" or pid not in fr_r.index or pid not in fr_c.index:
            continue
        events = _read_rows(d, "events.csv")
        lats = [float(r["value"]) / 1000 for r in events
                if r["event_type"] == "turn_latency_first_audio" and r["value"]]
        user_s = robot_s = 0.0
        for r in _read_rows(d, "speech.csv"):
            dur = float(r["duration_s"]) if r["duration_s"] else 0.0
            if r["actor"] == "user":
                user_s += dur
            else:
                robot_s += dur
        n_eng = sum(1 for r in events
                    if r["event_type"] == "intervention_engagement_sent")
        n_emo = sum(1 for r in events
                    if r["event_type"] == "intervention_emotion_sent")
        rows.append({
            "pid": pid, "n_eng": n_eng, "n_emo": n_emo, "n_tot": n_eng + n_emo,
            "med_lat_s": pd.Series(lats).median() if lats else np.nan,
            "n_turns": len(lats), "user_min": user_s / 60,
            "robot_min": robot_s / 60, "fr_delta": fr_r[pid] - fr_c[pid],
        })
    return pd.DataFrame(rows)


# --- renderers for the session-stats fragment --------------------------------

# (key, table label, boxplot label, decimals)
_ROBOT_METRICS = [
    ("robot_turns", "Robot turns", "Robot\\\\turns", 0),
    ("user_turns", "User turns", "User\\\\turns", 0),
    ("robot_min", "Robot talk-time (min)", "Robot\\\\talk (min)", 1),
    ("user_min", "User talk-time (min)", "User\\\\talk (min)", 1),
    ("context", "Context submissions", "Context\\\\submits", 0),
    ("int_eng", "Engagement interventions", "Engage-\\\\ment", 0),
    ("int_emo", "Emotion interventions", "Emotion", 0),
    ("int_tot", "All interventions", "All", 0),
]
_CTRL_METRICS = [
    ("int_eng", "Engagement counterfactuals", "Engage-\\\\ment", 0),
    ("int_emo", "Emotion counterfactuals", "Emotion", 0),
    ("int_tot", "All counterfactuals", "All", 0),
]


def _metric_series(met: dict, groups: dict, cond: str, key: str,
                   group: str | None = None) -> pd.Series:
    pids = [p for (p, c) in met if c == cond
            and (group is None or groups.get(p) == group)]
    return pd.Series({p: met[(p, cond)][key] for p in pids}, dtype=float)


def _fmt_mq(s: pd.Series, dec: int) -> str:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if not len(s):
        return "--"
    f = f"{{:.{dec}f}}"
    return (f"{f.format(s.median())} [{f.format(s.quantile(.25))}; "
            f"{f.format(s.quantile(.75))}]")


def _session_metrics_table(met: dict, groups: dict, cond: str,
                           metric_defs: list) -> str:
    rows = []
    for key, label, _, dec in metric_defs:
        s_all = _metric_series(met, groups, cond, key)
        s_a = _metric_series(met, groups, cond, key, GROUP_ADHD)
        s_c = _metric_series(met, groups, cond, key, GROUP_CONTROL)
        f = f"{{:.{dec}f}}"
        rows.append(
            f"{esc(label)} & {_fmt_mq(s_a, dec)} & {_fmt_mq(s_c, dec)} & "
            f"{_fmt_mq(s_all, dec)} & {s_all.std(ddof=1):.1f} & "
            f"{f.format(s_all.min())}--{f.format(s_all.max())} \\\\")
    body = "\n".join(rows)
    return f"""\\begin{{center}}\\small
\\begin{{tabular}}{{lrrrrr}}
\\toprule
 & ADHD & No-ADHD & Overall & SD & Range \\\\
 & \\multicolumn{{3}}{{c}}{{median [$Q_1$; $Q_3$]}} & & \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{center}}
"""


def _box_stats(s: pd.Series) -> dict | None:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 3:
        return None
    q1, med, q3 = s.quantile(.25), s.median(), s.quantile(.75)
    lo_f, hi_f = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    inside = s[(s >= lo_f) & (s <= hi_f)]
    return {"lw": inside.min(), "lq": q1, "med": med, "uq": q3,
            "uw": inside.max(),
            "outliers": sorted(s[(s < lo_f) | (s > hi_f)])}


def _fmt_p(p: float) -> str:
    if p < 0.001:
        return "$p < .001$"
    return f"$p = {p:.3f}$".replace("0.", ".", 1)


def _p_sub(p_str: str, sub: str) -> str:
    """Subscript a formatted p ('$p = .123$' -> '$p_U = .123$'); the
    test statistic itself is no longer reported (decided 05.09)."""
    return p_str.replace("$p ", f"$p_{sub} ")


def _p_val(p_str: str) -> str:
    """Bare value for a table cell under a $p_U$/$p_W$ header
    ('$p = .123$' -> '$.123$', '$p < .001$' -> '$< .001$')."""
    return p_str.replace("$p = ", "$").replace("$p < ", "$< ")


def _mwu_cells(a: pd.Series, c: pd.Series) -> tuple[str, str]:
    """(U, p) table cells for an ADHD-vs-Control Mann-Whitney U test."""
    a, c = a.dropna(), c.dropna()
    if len(a) < 2 or len(c) < 2:
        return "--", "--"
    u = scistats.mannwhitneyu(a, c, alternative="two-sided")
    return f"{u.statistic:.1f}", _fmt_p(u.pvalue)


def _wilcoxon_cells(a: pd.Series, b: pd.Series) -> tuple[str, str, int]:
    """(W, p, n) for a paired Wilcoxon signed-rank test (aligned on index)."""
    both = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    if len(both) < 3:
        return "--", "--", len(both)
    try:
        w = scistats.wilcoxon(both["a"], both["b"])
        return f"{w.statistic:.1f}", _fmt_p(w.pvalue), len(both)
    except ValueError:  # all-zero differences
        return "--", "(all diffs 0)", len(both)


def _box_chart(cols: list, *, ylabel: str, width: str,
               height: str = "5.2cm", ymin: float = 0,
               ymax: float | None = None, reflines: list | None = None) -> str:
    """Generic grouped box chart. cols = [(xlabel, [(color, series), ...])];
    one or two boxes per column. reflines = [(y, label)] dashed guides."""
    plots, ticks, labels = [], [], []
    for i, (short, boxes) in enumerate(cols):
        base = 2.0 * i
        ticks.append(f"{base + 1.0:.1f}")
        labels.append("{" + short + "}")
        positions = ([base + 1.0] if len(boxes) == 1
                     else [base + 0.62, base + 1.38])
        for pos, (color, series) in zip(positions, boxes):
            st = _box_stats(series)
            if st is None:
                continue
            outs = " ".join(f"(0,{v:g})" for v in st["outliers"])
            plots.append(
                f"    \\addplot[boxplot prepared={{draw position={pos:.2f}, "
                f"box extend=0.58, whisker extend=0.32, "
                f"lower whisker={st['lw']:g}, lower quartile={st['lq']:g}, "
                f"median={st['med']:g}, upper quartile={st['uq']:g}, "
                f"upper whisker={st['uw']:g}}}, color={color}, "
                f"fill={color}!25, mark=*, mark size=1pt, "
                f"mark options={{color={color}, fill={color}}}] "
                f"coordinates {{{outs}}};")
    xmax = 2.0 * len(cols)
    for y, lbl in (reflines or []):
        plots.append(
            f"    \\draw[dashed, black!55] (axis cs:0,{y:g}) -- "
            f"(axis cs:{xmax:.1f},{y:g}) node[right, font=\\tiny, "
            f"black!70, align=left] {{{lbl}}};")
    body = "\n".join(plots)
    ymax_opt = f" ymax={ymax:g}," if ymax is not None else ""
    return f"""\\begin{{tikzpicture}}
  \\begin{{axis}}[
    width={width}, height={height},
    boxplot/draw direction=y,
    xmin=0, xmax={xmax:.1f},
    xtick={{{",".join(ticks)}}}, xticklabels={{{",".join(labels)}}},
    x tick label style={{font=\\scriptsize, align=center}},
    ymin={ymin:g},{ymax_opt} ylabel={{{ylabel}}},
    ylabel style={{font=\\small}}, y tick label style={{font=\\scriptsize}},
    ymajorgrids, major grid style={{line width=0.3pt, draw=black!12}},
    axis x line*=bottom, axis y line*=left, clip=false,
  ]
{body}
  \\end{{axis}}
\\end{{tikzpicture}}"""


def _boxplot_panel(met: dict, groups: dict, cond: str, metric_defs: list, *,
                   ylabel: str, width: str = "0.42\\textwidth") -> str:
    cols = []
    for key, _, short, _dec in metric_defs:
        cols.append((short, [
            (ADHD_COLOR, _metric_series(met, groups, cond, key, GROUP_ADHD)),
            (CTRL_COLOR, _metric_series(met, groups, cond, key, GROUP_CONTROL)),
        ]))
    return _box_chart(cols, ylabel=ylabel, width=width)


def _paired_lines_chart(both: pd.DataFrame, groups: dict, *, ylabel: str,
                        xlabels: tuple[str, str] = ("Pre", "Post"),
                        width: str = "0.5\\textwidth",
                        ymin: float | None = None,
                        ymax: float | None = None) -> str:
    """One line per participant from column 0 to column 1 (index = PID),
    group-coloured, with bold group-median overlays."""
    a_col, b_col = both.columns[:2]
    lines = []
    for pid, row in both.dropna().iterrows():
        color = ADHD_COLOR if groups.get(pid) == GROUP_ADHD else CTRL_COLOR
        lines.append(
            f"    \\addplot[color={color}!55, line width=0.5pt, mark=*, "
            f"mark size=1pt, mark options={{fill={color}!55, "
            f"draw={color}!55}}] coordinates "
            f"{{(0,{row[a_col]:g}) (1,{row[b_col]:g})}};")
    for gname, color in ((GROUP_ADHD, ADHD_COLOR), (GROUP_CONTROL, CTRL_COLOR)):
        pids = [p for p in both.index if groups.get(p) == gname]
        sub = both.loc[pids].dropna()
        if len(sub) < 2:
            continue
        lines.append(
            f"    \\addplot[color={color}, line width=1.4pt, mark=*, "
            f"mark size=1.6pt] coordinates "
            f"{{(0,{sub[a_col].median():g}) (1,{sub[b_col].median():g})}};")
    rng = ""
    if ymin is not None:
        rng += f" ymin={ymin:g},"
    if ymax is not None:
        rng += f" ymax={ymax:g},"
    body = "\n".join(lines)
    return f"""\\begin{{tikzpicture}}
  \\begin{{axis}}[
    width={width}, height=5.8cm,
    xmin=-0.35, xmax=1.35,
    xtick={{0,1}}, xticklabels={{{xlabels[0]},{xlabels[1]}}},
    x tick label style={{font=\\small}},{rng} ylabel={{{ylabel}}},
    ylabel style={{font=\\small}}, y tick label style={{font=\\scriptsize}},
    ymajorgrids, major grid style={{line width=0.3pt, draw=black!12}},
    axis x line*=bottom, axis y line*=left,
  ]
{body}
  \\end{{axis}}
\\end{{tikzpicture}}"""


def _scatter_chart(df: pd.DataFrame, groups: dict, *, xlabel: str,
                   ylabel: str, width: str = "0.55\\textwidth",
                   hline: float | None = None) -> str:
    """Scatter of df['x'] vs df['y'] (index = PID), group-coloured."""
    plots = []
    for gname, color in ((GROUP_ADHD, ADHD_COLOR), (GROUP_CONTROL, CTRL_COLOR)):
        pts = " ".join(f"({r.x:g},{r.y:g})"
                       for pid, r in df.dropna().iterrows()
                       if groups.get(pid) == gname)
        if pts:
            plots.append(
                f"    \\addplot[only marks, mark=*, mark size=1.8pt, "
                f"color={color}, fill opacity=0.85] coordinates {{{pts}}};")
    body = "\n".join(plots)
    extra_hline = ""
    if hline is not None:
        extra_hline = f"    extra y ticks={{{hline:g}}}, extra y tick style={{grid=major, grid style={{dashed, black!45}}}},\n"
    return f"""\\begin{{tikzpicture}}
  \\begin{{axis}}[
    width={width}, height=6cm,
    xlabel={{{xlabel}}}, ylabel={{{ylabel}}},
    xlabel style={{font=\\small}}, ylabel style={{font=\\small}},
    x tick label style={{font=\\scriptsize}},
    y tick label style={{font=\\scriptsize}},
{extra_hline}    ymajorgrids, major grid style={{line width=0.3pt, draw=black!12}},
    axis x line*=bottom, axis y line*=left,
  ]
{body}
  \\end{{axis}}
\\end{{tikzpicture}}"""


def _episode_table(ep: pd.DataFrame) -> str:
    rows = []
    for sig, signame in (("eng", "Engagement"), ("emo", "Negative emotion")):
        for cond in ("Robot", "Control"):
            sub = ep[(ep.sig == sig) & (ep.cond == cond)]
            s = sub.dur.dropna()
            med = (f"{s.median():.0f} [{s.quantile(.25):.0f}; "
                   f"{s.quantile(.75):.0f}]") if len(s) else "--"
            rows.append(f"{signame} & "
                        f"{'Robot' if cond == 'Robot' else 'No-Robot'} & "
                        f"{len(sub)} & "
                        f"{int(sub.cued.sum())} & {int(sub.censored.sum())} & "
                        f"{med} \\\\")
    body = "\n".join(rows)
    return f"""\\begin{{center}}\\small
\\begin{{tabular}}{{llrrrr}}
\\toprule
Signal & Session & Episodes & Cued & Censored & Recovery (s), median [$Q_1$; $Q_3$] \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{center}}
"""


def _replay_table(groups: dict, sdirs: dict, met: dict) -> tuple[str, float]:
    idurs = pd.Series(intervention_utterance_durations(sdirs))
    dur = float(idurs.mean())
    rows, totals = [], [0, 0, 0]
    ordered = [p for g in (GROUP_ADHD, GROUP_CONTROL)
               for p in sorted((p for p, gg in groups.items() if gg == g), key=int)
               if (p, "Control") in met]
    for pid in ordered:
        d = sdirs[(pid, "Control")]
        logged = met[(pid, "Control")]["int_tot"]
        r0 = replay_counterfactuals(d, 0.0)
        rd = replay_counterfactuals(d, dur)
        totals = [totals[0] + logged, totals[1] + r0, totals[2] + rd]
        rows.append(f"{disp_pid(pid)} & {groups[pid]} & {logged} & {r0} & {rd} \\\\")
    body = "\n".join(rows)
    table = f"""\\begin{{center}}\\small
\\begin{{tabular}}{{llrrr}}
\\toprule
PID & Group & Logged & Replay (no utterance) & Replay ({dur:.0f}\\,s utterance) \\\\
\\midrule
{body}
\\midrule
Total & & {totals[0]} & {totals[1]} & {totals[2]} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{center}}
"""
    return table, dur


def build_instrument_stats(pre: pd.DataFrame, ctrl: pd.DataFrame,
                           post: pd.DataFrame, q_post: dict,
                           groups: dict[str, str]) -> str:
    """Tables + charts mirroring every questionnaire analysis in
    compute_stats.py, test statistics included."""
    out = [header_comment(["Qualtrics exports",
                           "analysis/logs/P*_csv session logs"])]
    n_a = sum(1 for g in groups.values() if g == GROUP_ADHD)
    legend_box = (f"\\textcolor{{ApxADHD}}{{\\rule{{2ex}}{{1.2ex}}}} ADHD "
                  f"($n = {n_a}$), "
                  f"\\textcolor{{ApxControl}}{{\\rule{{2ex}}{{1.2ex}}}} "
                  f"no-ADHD ($n = {len(groups) - n_a}$); boxes span the "
                  "quartiles, whiskers extend to the furthest value within "
                  "1.5 IQR, dots are outliers.")
    legend_cond = ("\\textcolor{ApxCondRobot}{\\rule{2ex}{1.2ex}} robot "
                   "session, \\textcolor{ApxCondControl}{\\rule{2ex}{1.2ex}} "
                   "no-robot session; boxes span the quartiles, whiskers "
                   "extend to the furthest value within 1.5 IQR, dots are "
                   "outliers.")
    pids_a = [p for p, g in groups.items() if g == GROUP_ADHD]
    pre_a = pre[pre["PID"].isin(pids_a)]
    pre_c = pre[~pre["PID"].isin(pids_a)]

    def by_group(df: pd.DataFrame, v: pd.Series) -> tuple[pd.Series, pd.Series]:
        return v[df["PID"].isin(pids_a)], v[~df["PID"].isin(pids_a)]

    # ------------------------------------------------------------ ESQ-R
    scales = ([("Overall (25 items)", "Overall",
                list(range(1, N_ESQR_ITEMS + 1)))]
              + [(name, short, items) for (name, items), short in zip(
                  ESQR_SUBSCALES.items(),
                  ["Plan\\\\mgmt.", "Time\\\\mgmt.", "Materials\\\\org.",
                   "Emotional\\\\reg.", "Behavioral\\\\reg."])])
    esq_rows = []
    for name, short, items in scales:
        ma = mean_items(pre_a, "PRE_ESQR_", items, scale_max=3)
        mc = mean_items(pre_c, "PRE_ESQR_", items, scale_max=3)
        _, p = _mwu_cells(ma, mc)
        esq_rows.append(f"{esc(name)} & {ma.mean():.2f} ({ma.std(ddof=1):.2f})"
                        f" & {mc.mean():.2f} ({mc.std(ddof=1):.2f}) & "
                        f"{_p_val(p)} \\\\")
    out.append("\\subsection*{Executive Skills Questionnaire Revised "
               "(ESQ-R) — pre-study}\n"
               "\\noindent{\\small Coded 0--3 (official 4-point response "
               "scale).}\\par\\vspace{0.4em}\n")
    out.append("""\\begin{center}\\small
\\begin{tabular}{lrrr}
\\toprule
Scale & ADHD & No-ADHD & $p_U$ \\\\
 & \\multicolumn{2}{c}{mean (SD)} & \\\\
\\midrule
""" + "\n".join(esq_rows) + """
\\bottomrule
\\end{tabular}
\\end{center}
""")

    # ------------------------------------------------------------ NARS
    items = list(range(1, N_NARS_ITEMS + 1))
    pre_by = pd.Series(mean_items(pre, "PRE_NARS_", items,
                                  NARS_REVERSE_ITEMS).values,
                       index=pre["PID"])
    post_by = pd.Series(mean_items(post, "POST_NARS_", items,
                                   NARS_REVERSE_ITEMS).values,
                        index=post["PID"])
    both = pd.concat([pre_by, post_by], axis=1,
                     keys=["pre", "post"]).dropna()
    _, wp, wn = _wilcoxon_cells(both["pre"], both["post"])
    change = both["post"] - both["pre"]
    nars_rows = []
    for label, sel in (("Pre", both["pre"]), ("Post", both["post"]),
                       ("Change (post $-$ pre)", change)):
        sa = sel[sel.index.isin(pids_a)]
        sc = sel[~sel.index.isin(pids_a)]
        _, p = _mwu_cells(sa, sc)
        nars_rows.append(f"{label} & {sa.mean():.2f} ({sa.std(ddof=1):.2f}) & "
                         f"{sc.mean():.2f} ({sc.std(ddof=1):.2f}) & "
                         f"{_p_val(p)} \\\\")
    out.append("\\subsection*{Negative Attitudes towards Robots Scale "
               "(NARS) — pre vs post}\n"
               "\\noindent{\\small Mean item score over the 14 items "
               "(1--5, items 3, 5, 6 reverse-coded; higher = more negative "
               "attitude).}\\par\\vspace{0.4em}\n")
    out.append("""\\begin{center}\\small
\\begin{tabular}{lrrr}
\\toprule
 & ADHD & No-ADHD & $p_U$ \\\\
 & \\multicolumn{2}{c}{mean (SD)} & \\\\
\\midrule
""" + "\n".join(nars_rows) + f"""
\\bottomrule
\\end{{tabular}}\\\\[2pt]
{{\\footnotesize Paired Wilcoxon pre vs post (all participants): """
               f"{_p_sub(wp, 'W')} ($n = {wn}$).}}\n"
               "\\end{center}\n")

    # ------------------------------------------------------------ TLX
    tlx_rows = []
    for dim in TLX_DIMS:
        col = f"POST_TLX_{dim}_1"
        r = post.set_index("PID")[col].pipe(pd.to_numeric, errors="coerce")
        c = ctrl.set_index("PID")[col].pipe(pd.to_numeric, errors="coerce")
        pair = pd.concat([r, c], axis=1, keys=["robot", "control"]).dropna()
        _, p, n = _wilcoxon_cells(pair["robot"], pair["control"])
        tlx_rows.append(
            f"{dim.capitalize()} & "
            f"{pair['robot'].mean():.1f} ({pair['robot'].std(ddof=1):.1f}) & "
            f"{pair['control'].mean():.1f} ({pair['control'].std(ddof=1):.1f})"
            f" & {n} & {_p_val(p)} \\\\")
    out.append("\\subsection*{NASA Task Load Index (TLX) — workload "
               "by condition}\n"
               "\\noindent{\\small 0--100 sliders, paired within-subject "
               "(Wilcoxon signed-rank); six uncorrected comparisons — "
               "apply multiple-testing caution when reporting."
               "}\\par\\vspace{0.4em}\n")
    out.append("""\\begin{center}\\small
\\begin{tabular}{lrrrr}
\\toprule
Dimension & Robot & No-Robot & $n$ & $p_W$ \\\\
 & \\multicolumn{2}{c}{mean (SD)} & & \\\\
\\midrule
""" + "\n".join(tlx_rows) + """
\\bottomrule
\\end{tabular}
\\end{center}
""")

    # ------------------------------------------------------------ SUS
    sus = sus_scores(post)
    sus_a, sus_c = by_group(post, sus)
    _, sp = _mwu_cells(sus_a, sus_c)
    sa, sc = sus_a.dropna(), sus_c.dropna()
    out.append("\\subsection*{System Usability Scale (SUS) — "
               "robot condition}\n")
    out.append(f"""\\begin{{center}}\\small
\\begin{{tabular}}{{lrrr}}
\\toprule
 & ADHD & No-ADHD & $p_U$ \\\\
 & \\multicolumn{{2}}{{c}}{{mean (SD)}} & \\\\
\\midrule
SUS (0--100) & {sa.mean():.1f} ({sa.std(ddof=1):.1f}) & """
               f"{sc.mean():.1f} ({sc.std(ddof=1):.1f}) & "
               f"{_p_val(sp)} \\\\\n"
               f"""\\bottomrule
\\end{{tabular}}\\\\[2pt]
{{\\footnotesize Medians [$Q_1$; $Q_3$]: ADHD {sa.median():.1f} """
               f"[{sa.quantile(.25):.1f}; {sa.quantile(.75):.1f}] "
               f"(range {sa.min():.1f}--{sa.max():.1f}, $n = {len(sa)}$); "
               f"no-ADHD {sc.median():.1f} [{sc.quantile(.25):.1f}; "
               f"{sc.quantile(.75):.1f}] (range {sc.min():.1f}--"
               f"{sc.max():.1f}, $n = {len(sc)}$).}}\n\\end{{center}}\n")

    # ------------------------------------------------- feature ratings
    # The polished table (rotated block labels, siunitx columns) is
    # generated by generate_results.py as results-charts/feature_stats.tex
    # and synced to the thesis repo; reference it instead of maintaining a
    # duplicate here (decided 05.09).
    out.append("\\subsection*{Feature ratings (robot condition, 1--5)}\n"
               "\\noindent{\\small Group means with Mann-Whitney U per item "
               "(15 uncorrected exploratory tests)."
               "}\\par\\vspace{0.4em}\n")
    out.append("\\input{results-charts/feature_stats}\n")
    return "\n".join(out)


def build_session_stats(groups: dict[str, str]) -> str:
    """Descriptive tables + boxplots from the parsed session logs."""
    out = [header_comment(["analysis/logs/P*_csv session logs"])]
    sdirs = session_dirs()
    gn = gate_note(groups, sdirs)
    if gn:
        out.append("\\noindent{\\small\\itshape " + gn +
                   "}\\par\\vspace{0.6em}\n")
    # Camera comparability check (simplified 05.09): the sections that
    # duplicated the main-report analyses (session metrics, episodes,
    # landmark recovery, cross-condition, counterfactual replay) were
    # removed; only this sensor-validity check remains. Speech-excluded
    # robot-session means vs no-robot means of both signals, per group
    # and combined, matching the mean-score definition in the results
    # tables (quiet_signal_mean over speech_exclusions).
    gated = gated_signals(sdirs)
    vals = {}
    for (pid, cond), d in sdirs.items():
        if pid not in groups:
            continue
        excl = speech_exclusions(d)
        for sig in ("eng", "emo"):
            if (pid, cond, sig) in gated:
                continue
            v = quiet_signal_mean(d, sig, excl)
            if v is not None:
                vals[(sig, cond, pid)] = v
    ser = pd.Series(vals) if vals else pd.Series(dtype=float)
    signame = {"eng": "Mean engagement score",
               "emo": "Mean neg.-emotion share"}
    cam_rows, foot_bits = [], []
    for sig in ("eng", "emo"):
        cam_rows.append(f"\\multicolumn{{4}}{{l}}{{\\itshape "
                        f"{signame[sig]}}} \\\\")
        r = ser.get((sig, "Robot"), pd.Series(dtype=float))
        c = ser.get((sig, "Control"), pd.Series(dtype=float))
        for cname, s_all in (("Robot (quiet)", r), ("No-Robot", c)):
            cells = []
            for scope in (GROUP_ADHD, GROUP_CONTROL, None):
                s = (s_all if scope is None else
                     s_all[s_all.index.map(groups.get) == scope])
                cells.append(f"{s.mean():.3f} ({s.std(ddof=1):.3f})"
                             if len(s) > 1 else "--")
            cam_rows.append(f"\\;{cname} & " + " & ".join(cells)
                            + " \\\\")
        parts = []
        for slabel, scope in (("ADHD", GROUP_ADHD),
                              ("no-ADHD", GROUP_CONTROL), ("all", None)):
            rs = (r if scope is None else
                  r[r.index.map(groups.get) == scope])
            cs = (c if scope is None else
                  c[c.index.map(groups.get) == scope])
            _, p, n = _wilcoxon_cells(rs, cs)
            parts.append(f"{slabel} {_p_sub(p, 'W')} ($n = {n}$)")
        foot_bits.append(("engagement" if sig == "eng" else
                          "neg.-emotion") + ": " + ", ".join(parts))
    out.append("\\subsection*{Camera comparability check}\n"
               "\\noindent{\\small The two conditions sense the signals "
               "through different cameras (robot head camera vs webcam at "
               "the same position). Session means over the speech-excluded "
               "timeline (samples during user/robot speech and the "
               f"{SPEECH_EXCLUSION_BUFFER_S:.0f}\\,s after each segment "
               "excluded, exact interval subtraction): a robot-vs-no-robot "
               "offset that persists outside interaction windows would "
               "indicate an optics/geometry artefact rather than head "
               "motion during speech. Signal series failing the coverage "
               "gate (note at the top of this section) are excluded."
               "}\\par\\vspace{0.4em}\n")
    out.append("""\\begin{center}\\small
\\begin{tabular}{lrrr}
\\toprule
 & ADHD & No-ADHD & All \\\\
 & \\multicolumn{3}{c}{mean (SD)} \\\\
\\midrule
""" + "\n".join(cam_rows) + f"""
\\bottomrule
\\end{{tabular}}\\\\[2pt]
{{\\footnotesize Paired Wilcoxon, Robot (quiet) vs No-Robot --- """
               + "; ".join(foot_bits) + ".}\n\\end{center}\n")
    return "\n".join(out)


def build_session_logs(groups: dict[str, str]) -> str:
    """Five full-page timeline charts from the parsed session logs."""
    out = [header_comment(["analysis/logs/P*_csv session logs"])]
    gn = gate_note(groups, session_dirs())
    if gn:
        out.append("\\noindent{\\small\\itshape " + gn + " The timeline "
                   "rows below still show all logged raw data (outages "
                   "appear as line gaps); the exclusion applies to the "
                   "statistical analyses.}\\par\\vspace{0.6em}\n")

    def sub(title: str, note: str, chart: str, last: bool = False) -> None:
        out.append(f"\\subsection*{{{esc(title)}}}\n"
                   f"\\noindent{{\\small {note}}}\\par\\vspace{{0.4em}}\n"
                   + chart + ("" if last else "\n\\newpage\n"))

    # legends trimmed to the essentials 05.09 (row order and the gating
    # caveat moved to authored prose where needed)
    legend_int = ("Speech segments per participant over the 45-minute robot "
                  "session: \\textcolor{ApxUserSpeech}{\\rule{2ex}{1.2ex}} user, "
                  "\\textcolor{ApxRobotSpeech}{\\rule{2ex}{1.2ex}} robot, grey = no "
                  "speech.")
    legend_score = ("Thin grey line: per-poll value; black line: the 30-second "
                    "rolling value used by the intervention policy; dashed "
                    "line: the trigger threshold ({thr}); vertical "
                    "{color} marks: {what}. Line gaps are sensing gaps.")

    sub("Interaction timelines (robot sessions)", legend_int,
        _interaction_chart(groups))
    sub("Engagement scores (robot sessions)",
        legend_score.format(thr="0.80",
                            color="\\textcolor{ApxTrigEng}{red}",
                            what="delivered engagement interventions"),
        _score_chart(groups, "Robot", "engagement.csv",
                     lambda r: _float_or_none(r, "score"),
                     lambda r: _float_or_none(r, "average"),
                     0.80, {"intervention_engagement_sent"}, "ApxTrigEng"))
    sub("Emotion scores (robot sessions)",
        legend_score.format(thr="0.60",
                            color="\\textcolor{ApxTrigEmo}{green}",
                            what="delivered emotion interventions"),
        _score_chart(groups, "Robot", "emotion.csv",
                     _neg_mass,
                     lambda r: _float_or_none(r, "negative_share"),
                     0.60, {"intervention_emotion_sent"}, "ApxTrigEmo"))
    sub("Engagement scores (no-robot sessions)",
        legend_score.format(thr="0.80",
                            color="\\textcolor{ApxTrigEng}{red}",
                            what="counterfactual engagement interventions "
                                 "(logged, never delivered)"),
        _score_chart(groups, "Control", "engagement.csv",
                     lambda r: _float_or_none(r, "score"),
                     lambda r: _float_or_none(r, "average"),
                     0.80, {"counterfactual_engagement"}, "ApxTrigEng"))
    sub("Emotion scores (no-robot sessions)",
        legend_score.format(thr="0.60",
                            color="\\textcolor{ApxTrigEmo}{green}",
                            what="counterfactual emotion interventions "
                                 "(logged, never delivered)"),
        _score_chart(groups, "Control", "emotion.csv",
                     _neg_mass,
                     lambda r: _float_or_none(r, "negative_share"),
                     0.60, {"counterfactual_emotion"}, "ApxTrigEmo"),
        last=True)
    return "\n".join(out)


def main():
    print(f"Data dir:   {DATA_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    paths = {k: newest_file(p) for k, p in FILE_PATTERNS.items()}
    for k, p in paths.items():
        print(f"  using [{k}]: {os.path.basename(p)}")

    pre, q_pre = load_qualtrics(paths["pre"])
    ctrl, q_ctrl = load_qualtrics(paths["control"])
    post, q_post = load_qualtrics(paths["post"])

    pre = clean(pre, "PRE_PID", "pre")
    ctrl = clean(ctrl, "POST_PID", "control")
    post = clean(post, "POST_PID", "post")

    groups = assign_groups(pre)

    for name, df in (("control", ctrl), ("post", post)):
        unknown = sorted(set(df["PID"]) - set(groups))
        if unknown:
            print(f"  WARNING: PIDs in {name} survey with no pre-survey entry "
                  f"(excluded from charts): {unknown}")

    outputs = {
        "apx_pre_study.tex": build_pre_study(pre, q_pre, groups, paths["pre"]),
        "apx_post_control.tex": build_post_control(ctrl, q_ctrl, groups, paths["control"]),
        "apx_post_robot.tex": build_post_robot(post, q_post, groups, paths["post"]),
        "apx_session_logs.tex": build_session_logs(groups),
        "apx_session_stats.tex": build_session_stats(groups),
        "apx_instrument_stats.tex": build_instrument_stats(pre, ctrl, post,
                                                           q_post, groups),
        "apx_preamble_snippet.tex": PREAMBLE_SNIPPET,
    }
    for fname, content in outputs.items():
        fpath = os.path.join(OUTPUT_DIR, fname)
        with open(fpath, "w") as f:
            f.write(content)
        print(f"  wrote {fname} ({len(content) // 1024} KB)")
    print("Done. \\input{} the fragments from your Appendix subfile "
          "(after adding the preamble snippet to Main.tex once).")

    if "--sync" in sys.argv:
        sync_to_thesis_repo()


def sync_to_thesis_repo():
    """Copy the three appendix fragments into the thesis repo and push, so
    Overleaf picks them up via Menu -> GitHub -> 'Pull GitHub changes'.
    The thesis repo is PRIVATE and must stay private: the fragments contain
    verbatim participant answers."""
    if not os.path.isdir(THESIS_APX_DIR):
        print(f"--sync: {THESIS_APX_DIR} not found — clone the thesis repo "
              "there first; skipping.")
        return
    repo = os.path.dirname(THESIS_APX_DIR)
    for fname in SYNC_FILES:
        shutil.copy2(os.path.join(OUTPUT_DIR, fname),
                     os.path.join(THESIS_APX_DIR, fname))
    changed = subprocess.run(
        ["git", "status", "--porcelain", "--", os.path.basename(THESIS_APX_DIR)],
        cwd=repo, capture_output=True, text=True).stdout.strip()
    if not changed:
        print("--sync: fragments unchanged — nothing to push.")
        return
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    subprocess.run(["git", "add", os.path.basename(THESIS_APX_DIR)],
                   cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-m",
         f"chore(appendix): regenerate questionnaire fragments ({stamp})\n\n"],
        cwd=repo, check=True)
    subprocess.run(["git", "push"], cwd=repo, check=True)
    print("--sync: pushed to thesis repo — in Overleaf: Menu -> GitHub -> "
          "Pull GitHub changes.")


if __name__ == "__main__":
    main()