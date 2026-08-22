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

import glob
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime

import numpy as np
import pandas as pd

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
SYNC_FILES = ["apx_pre_study.tex", "apx_post_control.tex", "apx_post_robot.tex"]

# Short subheaders for the open-ended questions (the full question text is
# still printed above each answer table). Keys are CSV column names.
OE_HEADERS = {
    "POST_OE_01": "Task description",
    "POST_OE_02": "Overall experience",
    "POST_OE_03": "Body doubling / presence effect",
    "POST_OE_04": "Awareness of disengagement detection",
    "POST_OE_05": "Re-engagement cues",
    "POST_OE_06": "Context-aware task guidance",
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
INCLUDE_PIDS: set[str] = {"11", "12", "13", "14", "15", "16", "17"}

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
# LaTeX renderers
# ============================================================================

ADHD_COLOR = "ApxADHD"
CTRL_COLOR = "ApxControl"

PREAMBLE_SNIPPET = r"""% ---------------------------------------------------------------
% Requirements for the auto-generated appendix fragments.
% Add ONCE to the thesis preamble (Main.tex), not to the fragments.
% ---------------------------------------------------------------
\usepackage{amsmath}
\usepackage{pgfplots}
\pgfplotsset{compat=1.17}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{pdflscape}
\usepackage{xcolor}
\definecolor{ApxADHD}{RGB}{68,119,170}    % Tol blue
\definecolor{ApxControl}{RGB}{204,102,17} % Tol orange (darker, prints well)
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
        legend_cmd = "\n    \\legend{ADHD, Control}" if show_legend else ""
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
    legend_cmd = "\n    \\legend{ADHD, Control}" if show_legend else ""

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
 & ADHD & Control & Overall \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{center}}
"""


def text_answers_table(answers_by_group: dict[str, list[tuple[str, str]]],
                       title: str, header: str | None = None) -> str:
    """Two-column longtable, ADHD | Control, one answer per row with PID.
    Column widths use \\linewidth so the table stretches on landscape pages
    (pdflscape updates \\linewidth, not \\textwidth). Rows are paired, so a
    long answer on one side leaves matching blank space on the other —
    accepted trade-off after paracol proved fragile (removed 17.08)."""
    a = answers_by_group.get(GROUP_ADHD, [])
    c = answers_by_group.get(GROUP_CONTROL, [])
    n = max(len(a), len(c))
    rows = []
    for i in range(n):
        left = f"\\textbf{{P{esc(disp_pid(a[i][0]))}:}} {esc(a[i][1])}" if i < len(a) else ""
        right = f"\\textbf{{P{esc(disp_pid(c[i][0]))}:}} {esc(c[i][1])}" if i < len(c) else ""
        rows.append(f"{left} & {right} \\\\[6pt]")
    body = "\n".join(rows)
    # Inline heading: bold short header; italic question on the SAME line.
    head = (f"\\textbf{{{esc(header)}}}; \\textit{{{esc(title)}}}" if header
            else f"\\textit{{{esc(title)}}}")
    return f"""\\noindent{head}\\par\\nopagebreak\\vspace{{0.3em}}
{{\\small
\\begin{{longtable}}{{p{{0.47\\linewidth}} p{{0.47\\linewidth}}}}
\\toprule
\\textbf{{ADHD}} & \\textbf{{Control}} \\\\
\\midrule
\\endhead
{body}
\\bottomrule
\\end{{longtable}}}}
\\vspace{{0.3em}}
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
    """Open-ended sections render in LANDSCAPE (pdflscape): the two-column
    verbatim-answer tables use the page's long edge, which long prose needs.
    Inside the landscape environment \\linewidth is the landscape width
    (\\textwidth is NOT updated by pdflscape), so the answer tables size
    their columns with \\linewidth to stretch automatically. Each question gets a
    short bold subheader from OE_HEADERS (skipped if it would just repeat
    the section title), with the full question text below it."""
    parts = ["\\begin{landscape}\n"
             f"\\subsection*{{{esc(title)}}}\n"]
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
    parts.append("\\end{landscape}\n")
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
               f"$n_{{\\text{{ADHD}}}} = {n_a}$, $n_{{\\text{{Control}}}} = {n_c}$, "
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
    out.append(likert_instrument(pre, qtext, groups, "PRE_NARS_", 14,
                                 "NARS — Negative Attitudes towards Robots Scale",
                                 hint="LIKERT5", pag=pag))
    out.append(likert_instrument(pre, qtext, groups, "PRE_ASRS_", 18,
                                 "ASRS v1.1 — Adult ADHD Self-Report Scale",
                                 hint="ASRS", pag=pag))
    out.append(likert_instrument(pre, qtext, groups, "PRE_ESQR_", 25,
                                 "ESQ-R — Executive Skills Questionnaire (Revised)",
                                 hint="ESQR", pag=pag))
    return "\n".join(out)


def build_post_control(ctrl, qtext, groups, src):
    out = [header_comment([src])]
    pag = Paginator()
    out.append(open_ended_tables(ctrl, qtext, groups, ["POST_OE_01"],
                                 "Task description"))
    tlx_cols = ["POST_TLX_MENTAL_1", "POST_TLX_PHYSICAL_1", "POST_TLX_TEMPORAL_1",
                "POST_TLX_PERFORMANCE_1", "POST_TLX_EFFORT_1", "POST_TLX_FRUSTRATION_1"]
    out.append(slider_instrument(ctrl, qtext, groups, tlx_cols,
                                 "NASA-TLX (control session)", pag=pag))
    return "\n".join(out)


def build_post_robot(post, qtext, groups, src):
    out = [header_comment([src])]
    pag = Paginator()
    out.append(likert_instrument(post, qtext, groups, "POST_NARS_", 14,
                                 "NARS — Negative Attitudes towards Robots Scale",
                                 hint="LIKERT5", pag=pag))
    out.append(likert_instrument(post, qtext, groups, "POST_SUS_", 10,
                                 "SUS — System Usability Scale", hint="LIKERT5", pag=pag))
    tlx_cols = ["POST_TLX_MENTAL_1", "POST_TLX_PHYSICAL_1", "POST_TLX_TEMPORAL_1",
                "POST_TLX_PERFORMANCE_1", "POST_TLX_EFFORT_1", "POST_TLX_FRUSTRATION_1"]
    out.append(slider_instrument(post, qtext, groups, tlx_cols,
                                 "NASA-TLX (robot session)", pag=pag))
    out.append(likert_instrument(post, qtext, groups, "POST_FEAT_PRESENCE_", 3,
                                 "Body doubling / presence", hint="LIKERT5", pag=pag))
    out.append(likert_instrument(post, qtext, groups, "POST_FEAT_INATT_", 5,
                                 "Inattention detection", hint="LIKERT5", pag=pag))
    out.append(likert_instrument(post, qtext, groups, "POST_FEAT_CONTEXT_", 4,
                                 "Context-aware / task-aware support",
                                 hint="LIKERT5", pag=pag))
    out.append(likert_instrument(post, qtext, groups, "POST_FEAT_OVERALL_", 3,
                                 "Overall experience", hint="LIKERT5", pag=pag))
    oe_cols = [f"POST_OE_{i:02d}" for i in range(1, 13)]
    out.append(open_ended_tables(post, qtext, groups, oe_cols,
                                 "Open-ended questions"))
    return "\n".join(out)


# ============================================================================
# Main
# ============================================================================

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