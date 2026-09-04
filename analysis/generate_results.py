#!/usr/bin/env python3
"""
generate_results.py
===================
Generates the RESULTS-section chart fragments (pgfplots) for the thesis.

Contract with the thesis (decided 25.08): each chart is emitted as a bare
tikzpicture fragment in OUTPUT_DIR/results-charts/<name>.tex — no figure
environment, no caption, no label. The thesis wraps each fragment in its own
\\begin{figure} ... \\caption{} ... \\label{} block and \\input{}s the
fragment, so regenerations update every figure in place without touching
authored captions. A results_preview.tex gallery (all charts under their
filenames) is emitted alongside for browsing; it is not meant to be \\input.

Run:  .venv/bin/python generate_results.py [--sync]
--sync copies the fragments into the thesis repo's results-charts/ folder,
commits, and pushes (same mechanics as generate_appendix.py; the thesis repo
is PRIVATE and must stay private).

Shares loaders, grouping, scale maps and the participant whitelist with
generate_appendix.py — the two scripts can never disagree about the data.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from datetime import datetime

import pandas as pd
from scipy import stats as sps

from generate_appendix import (
    FILE_PATTERNS, GROUP_ADHD, GROUP_CONTROL, GROUPS_FILE,  # noqa: F401
    assign_groups, clean, esc, load_qualtrics, newest_file, strip_stem,
    to_rank,
    _ROBOT_METRICS, _fmt_p, _metric_series, _mwu_cells, _wilcoxon_cells,
    _read_rows as _log_rows,
    session_dirs, session_metrics,
    EPISODE_CENSOR_GAP_S, SESSION_MAX_MIN, episode_records,
    extract_episodes, gated_signals, signal_polls,
    quiet_signal_mean, quiet_within_pct, speech_exclusions,
)

# ============================================================================
# CONFIG
# ============================================================================

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "output", "results-charts")
# --sync pushes the fragments into each of these repos (all PRIVATE — the
# charts render participant-derived data). Same \input contract everywhere.
SYNC_TARGETS = [
    os.path.expanduser("~/Projects/MSc-Project-Final-Report/results-charts"),
    os.path.expanduser("~/Projects/HRI-submission-Timo/results-charts"),
]

ADHD_COLOR = "ApxADHD"        # colors already defined in the thesis preamble
CTRL_COLOR = "ApxControl"

# The post-robot feature instruments, in questionnaire order (titles match
# the appendix section headings).
FEATURE_BLOCKS = [
    ("Body doubling", "POST_FEAT_PRESENCE_", 3),
    ("Inattention detection", "POST_FEAT_INATT_", 5),
    ("Task-aware support", "POST_FEAT_CONTEXT_", 4),
    ("Overall experience", "POST_FEAT_OVERALL_", 3),
]

# Verbal anchors only under 1/3/5 — all five collide at this axis width.
LIKERT_TICKS = [
    (1, "Strongly disagree"), (2, None), (3, "Neutral"),
    (4, None), (5, "Strongly agree"),
]

# One-line display labels for the results figure (full question text lives in
# the appendix). Keys are CSV columns; fallback is the full question.
SHORT_LABELS = {
    "POST_FEAT_PRESENCE_1": "Easier to start working",
    "POST_FEAT_PRESENCE_2": "Helped me stay on task",
    "POST_FEAT_PRESENCE_3": "Felt like working alongside someone",
    "POST_FEAT_INATT_1": "Disengagement accurately detected",
    "POST_FEAT_INATT_2": "Re-engagement prompts well-timed",
    "POST_FEAT_INATT_3": "Re-engagement prompts helped re-focus",
    "POST_FEAT_INATT_4": "Negative emotions accurately detected",
    "POST_FEAT_INATT_5": "Emotional responses appropriate",
    "POST_FEAT_CONTEXT_1": "Task guidance relevant",
    "POST_FEAT_CONTEXT_2": "Useful hints when stuck",
    "POST_FEAT_CONTEXT_3": "Similar to a teaching assistant",
    "POST_FEAT_CONTEXT_4": "Comfortable asking for help",
    "POST_FEAT_OVERALL_1": "Satisfied with experience",
    "POST_FEAT_OVERALL_2": "Would use again",
    "POST_FEAT_OVERALL_3": "Would recommend to others",
}


# ============================================================================
# Chart builders — each returns (filename_stem, tex_fragment)
# ============================================================================

def _feature_data(post, qtext, groups):
    """Shared data pass for the feature charts: rows, means, block spans."""
    all_keys, all_texts, data_rows, block_spans = [], [], [], []
    for title, prefix, n_items in FEATURE_BLOCKS:
        first_key = last_key = None
        for i in range(1, n_items + 1):
            col = f"{prefix}{i}"
            if col not in post.columns:
                print(f"  WARNING: column {col} missing — skipped.")
                continue
            means = {}
            for g in (GROUP_ADHD, GROUP_CONTROL):
                pids = [p for p, gg in groups.items() if gg == g]
                vals = to_rank(post[post["PID"].isin(pids)][col],
                               "LIKERT5").dropna()
                means[g] = vals.mean() if len(vals) else float("nan")
            key = f"r{len(all_keys)}"
            all_keys.append(key)
            all_texts.append(esc(SHORT_LABELS.get(
                col, strip_stem(qtext.get(col, col)))))
            data_rows.append((key, means[GROUP_ADHD], means[GROUP_CONTROL]))
            first_key = first_key or key
            last_key = key
        if first_key:
            block_spans.append((title, first_key, last_key))
    return all_keys, all_texts, data_rows, block_spans


def _render_feature_chart(data, *, axis_w, label_w, pitch, bar_pt,
                          label_font, title_font, tick_anchors, value_labels,
                          value_font="\\scriptsize", xmax=5.6,
                          span_ext="0.3cm", value_extra_pt=0.0,
                          adhd_only=False):
    """Render the grouped-bar feature chart at a given size. See
    chart_feature_means / chart_feature_means_col for the two variants."""
    all_keys, all_texts, data_rows, block_spans = data
    sym = ",".join(all_keys)
    yticklabels = ",".join("{" + t + "}" for t in all_texts)
    coords_a = " ".join(f"({a:.2f},{k})" for k, a, c in data_rows)
    coords_c = " ".join(f"({c:.2f},{k})" for k, a, c in data_rows)
    off = bar_pt / 2

    # nodes near coords ignores the bar shift on a reversed symbolic xbar
    # axis -> explicit value nodes at each bar's own offset (ADHD on top:
    # Control plots first / lower).
    value_nodes = ""
    if value_labels and adhd_only:
        # single-bar variant (restored 04.09): labels sit on the bar end,
        # no pair offset needed
        value_nodes = "\n    ".join(
            f"\\node[font={value_font}, inner sep=1pt, anchor=west,"
            f" xshift=2pt] "
            f"at (axis cs:{a:.2f},{k}) {{{a:.2f}}};"
            for k, a, c in data_rows)
    elif value_labels:
        # value_extra_pt pushes the labels slightly beyond the bar centres —
        # needed in the compact variant where half a bar width is less than
        # the label text's half-height.
        v_off = off + value_extra_pt
        value_nodes = "\n    ".join(
            f"\\node[font={value_font}, inner sep=1pt, anchor=west,"
            f" xshift=2pt, yshift={v_off:.2f}pt] "
            f"at (axis cs:{a:.2f},{k}) {{{a:.2f}}}; "
            f"\\node[font={value_font}, inner sep=1pt, anchor=west,"
            f" xshift=2pt, yshift=-{v_off:.2f}pt] "
            f"at (axis cs:{c:.2f},{k}) {{{c:.2f}}};"
            for k, a, c in data_rows)

    if tick_anchors:
        xticklabels = ",".join(
            f"{{{v}\\\\{{\\scriptsize ({lbl})}}}}" if lbl else f"{{{v}}}"
            for v, lbl in LIKERT_TICKS)
        xtick_line = f"xticklabels={{{xticklabels}}},\n    x tick label style={{font={label_font}, align=center}},"
    else:
        xtick_line = f"x tick label style={{font={label_font}}},"

    span_coords = "\n    ".join(
        f"\\coordinate (b{i}t) at (axis cs:0,{first});"
        f" \\coordinate (b{i}b) at (axis cs:0,{last});"
        for i, (_, first, last) in enumerate(block_spans))
    bracket_x = f"-\\dimexpr{label_w}+8pt\\relax"
    span_draws = "\n  ".join(
        f"\\draw ([xshift={bracket_x}, yshift={span_ext}]b{i}t)"
        f" -- ([xshift={bracket_x}, yshift=-{span_ext}]b{i}b)"
        f" node[midway, rotate=90, anchor=south, align=center,"
        f" font={title_font}]"
        f" {{{esc(title).replace(' ', chr(92) * 2)}}};"
        for i, (title, _, _) in enumerate(block_spans))

    if adhd_only:
        plots = (f"\\addplot[xbar, fill={ADHD_COLOR}, "
                 f"draw={ADHD_COLOR}!70!black] coordinates {{{coords_a}}};")
    else:
        plots = (
            f"\\addplot[xbar, fill={CTRL_COLOR}, "
            f"draw={CTRL_COLOR}!70!black] coordinates {{{coords_c}}};\n"
            f"    \\addplot[xbar, fill={ADHD_COLOR}, "
            f"draw={ADHD_COLOR}!70!black] coordinates {{{coords_a}}};\n"
            f"    \\legend{{No-ADHD, ADHD}}")
    # NO trim axis (see thesis variant note): the full bounding box must
    # include labels so \centering centers the ensemble.
    return f"""\\begin{{tikzpicture}}
  \\begin{{axis}}[
    xbar, bar width={bar_pt}pt, y={pitch},
    scale only axis, width={axis_w},
    symbolic y coords={{{sym}}}, ytick={{{sym}}}, y dir=reverse,
    yticklabels={{{yticklabels}}},
    yticklabel style={{font={label_font}, align=right, text width={label_w}}},
    xmin=0, xmax={xmax},
    xtick={{1,2,3,4,5}},
    {xtick_line}
    axis x line*=bottom, axis y line*=left,
    legend style={{font={label_font}, at={{(0.5,1.01)}}, anchor=south,
                   draw=none, fill=none}},
    legend columns=2,
    legend image code/.code={{\\draw[#1] (0cm,-0.06cm) rectangle (0.18cm,0.12cm);}},
    reverse legend,
  ]
    {plots}
    {value_nodes}
    {span_coords}
  \\end{{axis}}
  {span_draws}
\\end{{tikzpicture}}"""


def chart_feature_means(post, qtext, groups):
    """Thesis version: fills the text width, verbal tick anchors, per-bar
    value labels."""
    data = _feature_data(post, qtext, groups)
    return "feature_means", _render_feature_chart(
        data, axis_w="0.48\\textwidth", label_w="0.42\\textwidth",
        pitch="0.75cm", bar_pt=5.5, label_font="\\small",
        title_font="\\small\\bfseries", tick_anchors=True,
        value_labels=True)


def chart_feature_means_col(post, qtext, groups):
    """ACM column-width version (HRI): fits \\columnwidth in a two-column
    layout. Numeric ticks only (anchors in the caption); per-bar value
    labels in \\tiny, nudged slightly past the bar centres so the pair
    clears each other at the tight pitch."""
    data = _feature_data(post, qtext, groups)
    return "feature_means_col", _render_feature_chart(
        data, axis_w="0.40\\columnwidth", label_w="0.46\\columnwidth",
        pitch="0.48cm", bar_pt=4.0, label_font="\\scriptsize",
        title_font="\\scriptsize\\bfseries", tick_anchors=False,
        value_labels=True, value_font="\\tiny", value_extra_pt=1.6,
        xmax=5.6, span_ext="0.18cm")


# Compact row labels for the results-section table (the appendix keeps the
# long forms; the "(incl. interventions)" qualifier moves to the caption).
_SHORT_METRIC_LABELS = {
    "Robot turns (spoken)": "Robot turns",
    "User turns (spoken)": "User turns",
    "Engagement interventions": "Engagement interv.",
}


def _fmt_v(v, dec: int) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "--"
    return f"{v:.{dec}f}" if dec else f"{v:g}"


# --- siunitx S-column machinery (decided 02.09): every numeric column in
# the session-metrics table family is an S column, so values align on the
# decimal point and the number block centres under its header. Requires
# \usepackage{siunitx} in the target preamble. Non-numeric cells (p-value
# specials, '--') and text headers over S columns must be braced.

_NUM_RE = re.compile(r"-?(\d+\.?\d*|\.\d+)")


def _scell(v: str) -> str:
    """Cell content for an S column: bare numbers pass, all else braced."""
    v = v.strip()
    return v if _NUM_RE.fullmatch(v) else "{" + v + "}"


def _pcell(p: str) -> str:
    """p-value cell for an S column (input from _mwu/_wilcoxon cells)."""
    p = (p.replace("p = ", "").replace("p < ", "< ")
         .replace("$", "").strip())
    if p.startswith("<"):
        return "{$" + p.replace(" ", "") + "$}"
    return _scell(p)


def _welch_p(a: pd.Series, c: pd.Series) -> str:
    """Welch t-test p as an S-ready cell (Nicole 03.09: t sanity twins)."""
    a, c = a.dropna(), c.dropna()
    if len(a) < 2 or len(c) < 2:
        return "{--}"
    return _pcell(_fmt_p(sps.ttest_ind(a, c, equal_var=False).pvalue))


def _pairedt_p(a: pd.Series, b: pd.Series) -> str:
    """Paired t-test p as an S-ready cell (aligned on index)."""
    both = pd.concat([a, b], axis=1).dropna()
    if len(both) < 3:
        return "{--}"
    return _pcell(_fmt_p(
        sps.ttest_rel(both.iloc[:, 0], both.iloc[:, 1]).pvalue))


def _sspecs(matrix: list[list[str]]) -> list[str]:
    """Per-column S[table-format=...] specs from formatted cell strings
    (matrix is row-major, label column excluded)."""
    specs = []
    for col in zip(*matrix):
        ip, dp, sign = 1, 0, ""
        for v in col:
            v = v.strip()
            if not _NUM_RE.fullmatch(v):
                continue
            if v.startswith("-"):
                sign, v = "-", v[1:]
            i, _, d = v.partition(".")
            ip = max(ip, len(i))
            dp = max(dp, len(d))
        specs.append(f"S[table-format={sign}{ip}.{dp}]")
    return specs


def _sheads(heads: list[str]) -> str:
    """Header cells over S columns: brace each so it centres as text."""
    return " & ".join("{" + h + "}" for h in heads)


def _render_session_metrics_table(groups, *, quartiles, size, colsep) -> str:
    """Robot-session metrics table for the results sections: one row per
    metric, each stat an 'ADHD | Control' pair, closed by the Mann-Whitney
    p value (U dropped for space, decided 30.08). Each pair is a
    right/left sub-column duo with the bar as fixed inter-column material,
    so the separators align vertically across all rows. The thesis variant
    carries the quartiles; the HRI variant drops Q1/Q3 to fit half a page.
    Interaction metrics only (decided 02.09): the intervention rows moved
    to the DiD summary, which owns the inattention-specific metrics."""
    sdirs = session_dirs()
    met = {k: session_metrics(d, k[1]) for k, d in sdirs.items()
           if k[0] in groups}
    rows = []
    for key, label, _, dec in _ROBOT_METRICS:
        if key.startswith("int_"):
            continue
        s_a = _metric_series(met, groups, "Robot", key, GROUP_ADHD)
        s_c = _metric_series(met, groups, "Robot", key, GROUP_CONTROL)
        # min/max are observed values (metric precision); the derived
        # stats get one decimal so count medians/quartiles keep their .5s.
        stats_ = [(s_a.min(), s_c.min(), dec)]
        if quartiles:
            stats_.append((s_a.quantile(.25), s_c.quantile(.25), 1))
        stats_.append((s_a.mean(), s_c.mean(), 1))
        stats_.append((s_a.median(), s_c.median(), 1))
        if quartiles:
            stats_.append((s_a.quantile(.75), s_c.quantile(.75), 1))
        stats_.append((s_a.max(), s_c.max(), dec))
        stats_.append((s_a.std(ddof=1), s_c.std(ddof=1), 1))
        _, p = _mwu_cells(s_a, s_c)
        rows.append([esc(_SHORT_METRIC_LABELS.get(label, label))]
                    + [_fmt_v(x, d) for a, c, d in stats_ for x in (a, c)]
                    + [_pcell(p), _welch_p(s_a, s_c)])
    stat_heads = (["Min", "$Q_1$", "Mean", "Median", "$Q_3$", "Max", "SD"]
                  if quartiles else ["Min", "Mean", "Median", "Max", "SD"])
    n_stats = len(stat_heads)
    specs = _sspecs([r[1:] for r in rows])
    # pairs keep the bar-anchored r/l alignment (a full decimal grid
    # across count and score rows would add ~100pt of reserved width);
    # the p column is an S column so its values align on the point.
    pair_spec = "r@{\\extracolsep{0pt}\\,$|$\\,}l@{\\extracolsep{\\fill}\\hspace{5pt}}" * n_stats
    heads = " & ".join(f"\\multicolumn{{2}}{{c}}{{{h}}}" for h in stat_heads)
    body = "\n".join(" & ".join(_scell(c) if i else c
                                for i, c in enumerate(r)) + " \\\\"
                     for r in rows)
    return f"""\\begingroup\\centering{size}
\\setlength{{\\tabcolsep}}{{{colsep}}}%
\\begin{{tabular*}}{{\\textwidth}}{{@{{}}l@{{\\extracolsep{{\\fill}}}}{pair_spec}{specs[-2]}{specs[-1]}@{{}}}}
\\toprule
 & \\multicolumn{{{2 * n_stats}}}{{c}}{{ADHD\\,$|$\\,No-ADHD}} &
   \\multicolumn{{2}}{{c}}{{$p$}} \\\\
\\cmidrule(lr){{2-{2 * n_stats + 1}}} \\cmidrule(lr){{{2 * n_stats + 2}-{2 * n_stats + 3}}}
 & {heads} & {{$p_U$}} & {{$p_t$}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular*}}\\par\\endgroup"""


def table_session_metrics(post, qtext, groups):
    """Thesis version: full seven-stat spread including the quartiles."""
    return "session_metrics_robot", _render_session_metrics_table(
        groups, quartiles=False, size="\\footnotesize", colsep="2pt")


def table_session_metrics_col(post, qtext, groups):
    """HRI version: no Q1/Q3, \\scriptsize — sized for half a page."""
    return "session_metrics_robot_col", _render_session_metrics_table(
        groups, quartiles=False, size="\\scriptsize", colsep="3pt")


def _interval_union(intervals: list[tuple[float, float]]) -> float:
    """Total length of the union of [start, end] intervals."""
    total, cur0, cur1 = 0.0, None, None
    for s0, s1 in sorted(intervals):
        if cur1 is None or s0 > cur1:
            if cur1 is not None:
                total += cur1 - cur0
            cur0, cur1 = s0, s1
        else:
            cur1 = max(cur1, s1)
    if cur1 is not None:
        total += cur1 - cur0
    return total


def _cross_condition_rows(groups, group=None) -> list[tuple[str, pd.DataFrame, int]]:
    """Session-metric rows shared by the cross-condition, fixed-condition
    and delta tables: intervention counts, session-mean signal values,
    per-participant median below-threshold episode durations, and
    %-of-observed-time within the engagement / emotion / both thresholds.
    Each row is (label, DataFrame[Robot, Control] indexed by pid, dec) for
    the given group (None = everyone); cells are NaN where a session or
    signal is unavailable — callers decide whether to pair-drop. The 80%
    signal-coverage gate applies: a gated (session, signal) blanks that
    signal's rows and the 'both'/'any' rows for that session.
    Episode-duration rows use observed (uncensored) episodes only."""
    sdirs = session_dirs()
    members = ({p for p, g in groups.items() if g == group}
               if group else set(groups))
    met = {k: session_metrics(d, k[1]) for k, d in sdirs.items()
           if k[0] in members}
    gated = gated_signals(sdirs)
    rows = []
    for key, label in (("int_eng", "Engagement interv."),
                       ("int_emo", "Emotion interv."),
                       ("int_tot", "All interv.")):
        df = pd.DataFrame({
            cond: _metric_series(met, groups, cond, key, group)
            for cond in ("Robot", "Control")})
        rows.append((label, df, 0))

    # Session-mean raw signal values over speech-excluded samples
    # (decided 05.09; no-op for no-robot sessions), gate-respecting.
    for sig, label in (("eng", "Mean engagement score"),
                       ("emo", "Mean neg.-emotion share")):
        vals = {}
        for (pid, cond), d in sdirs.items():
            if pid not in members or (pid, cond, sig) in gated:
                continue
            v = quiet_signal_mean(d, sig)
            if v is not None:
                vals[(pid, cond)] = v
        df = pd.Series(vals).unstack().reindex(
            columns=["Robot", "Control"])
        rows.append((label, df, 2))

    ep, spans = episode_records(groups, sdirs)
    ep = ep[ep.pid.isin(members)]

    # Episode durations: per-participant MEDIAN duration of observed
    # below-threshold episodes (start of detection to back-at-threshold).
    obs = ep.dropna(subset=["dur"])
    both_ok = {(pid, cond) for (pid, cond) in sdirs
               if pid in members and not any(
                   (pid, cond, s) in gated for s in ("eng", "emo"))}
    for sig, label in (("eng", "Diseng.\\ episode duration (s)"),
                       ("emo", "Neg.-emo.\\ episode duration (s)"),
                       ("any", "Any episode duration (s)")):
        if sig == "any":
            sub = obs[[(p, c) in both_ok
                       for p, c in zip(obs.pid, obs.cond)]]
        else:
            sub = obs[obs.sig == sig]
        med = sub.groupby(["pid", "cond"]).dur.median().unstack()
        med = med.reindex(columns=["Robot", "Control"])
        rows.append((label, med, 1))

    # %-within-threshold rows over the speech-excluded timeline (exact
    # subtraction; decided 05.09 — see quiet_within_pct). Gate applies
    # per (session, signal); 'both' needs both signals ungated.
    within = {}  # sig -> {(pid, cond): % within threshold}
    for (pid, cond), d in sdirs.items():
        if pid not in members:
            continue
        excl = speech_exclusions(d)
        for sig in ("eng", "emo"):
            if (pid, cond, sig) in gated:
                continue
            v = quiet_within_pct(d, (sig,), excl)
            if v is not None:
                within.setdefault(sig, {})[(pid, cond)] = v
        if (pid, cond) in both_ok:
            v = quiet_within_pct(d, ("eng", "emo"), excl)
            if v is not None:
                within.setdefault("both", {})[(pid, cond)] = v

    for sig, label in (("eng", "Time within eng.\\ threshold (\\%)"),
                       ("emo", "Time within emo.\\ threshold (\\%)"),
                       ("both", "Time within both thresholds (\\%)")):
        s = pd.Series(within.get(sig, {}))
        df = (s.unstack() if len(s) else pd.DataFrame()).reindex(
            columns=["Robot", "Control"])
        rows.append((label, df, 1))
    return rows


def _render_cross_table(groups, group, *, quartiles, size, colsep) -> str:
    """One group's robot-vs-control table: interventions (control =
    counterfactual upper bound), episode durations, and %-time within
    thresholds, each stat a 'Robot | Control' pair, closed by the paired
    Wilcoxon p. Same aligned-separator layout as the session-metrics
    table."""
    body_rows = []
    for label, df, dec in _cross_condition_rows(groups, group):
        df = df.dropna()  # paired design: both conditions required
        r, c = df["Robot"], df["Control"]
        dd = max(dec, 1)  # derived stats: >=1 decimal, more for scores
        stats_ = [(r.min(), c.min(), dec)]
        if quartiles:
            stats_.append((r.quantile(.25), c.quantile(.25), dd))
        stats_.append((r.mean(), c.mean(), dd))
        stats_.append((r.median(), c.median(), dd))
        if quartiles:
            stats_.append((r.quantile(.75), c.quantile(.75), dd))
        stats_.append((r.max(), c.max(), dec))
        stats_.append((r.std(ddof=1), c.std(ddof=1), dd))
        _, p, n = _wilcoxon_cells(r, c)
        body_rows.append(
            [label]
            + [_fmt_v(x, d).replace("100.0", "100")
               for a, b, d in stats_ for x in (a, b)]
            + [str(n), _pcell(p), _pairedt_p(r, c)])
    stat_heads = (["Min", "$Q_1$", "Mean", "Median", "$Q_3$", "Max", "SD"]
                  if quartiles else ["Min", "Mean", "Median", "Max", "SD"])
    n_stats = len(stat_heads)
    specs = _sspecs([r[1:] for r in body_rows])
    # quartile variant carries 7 pairs: a slimmer minimum inter-group gap
    # keeps the extra bar margins inside \textwidth at natural width.
    gap = "2pt" if quartiles else "5pt"
    pair_spec = (f"r@{{\\extracolsep{{0pt}}\\,$|$\\,}}"
                 f"l@{{\\extracolsep{{\\fill}}\\hspace{{{gap}}}}}") * n_stats
    heads = " & ".join(f"\\multicolumn{{2}}{{c}}{{{h}}}" for h in stat_heads)
    body = "\n".join(" & ".join(_scell(c) if i else c
                                for i, c in enumerate(r)) + " \\\\"
                     for r in body_rows)
    return f"""\\begingroup\\centering{size}
\\setlength{{\\tabcolsep}}{{{colsep}}}%
\\begin{{tabular*}}{{\\textwidth}}{{@{{}}l@{{\\extracolsep{{\\fill}}}}{pair_spec}{specs[-3]}{specs[-2]}{specs[-1]}@{{}}}}
\\toprule
 & \\multicolumn{{{2 * n_stats}}}{{c}}{{Robot\\,$|$\\,No-Robot}} &
   \\multicolumn{{3}}{{c}}{{Within-subject}} \\\\
\\cmidrule(lr){{2-{2 * n_stats + 1}}} \\cmidrule(lr){{{2 * n_stats + 2}-{2 * n_stats + 4}}}
 & {heads} & {{$n$}} & {{$p_W$}} & {{$p_t$}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular*}}\\par\\endgroup"""


def table_session_metrics_adhd(post, qtext, groups):
    """Thesis version: ADHD robot-vs-control, full stat spread."""
    return "session_metrics_adhd", _render_cross_table(
        groups, GROUP_ADHD, quartiles=False, size="\\footnotesize",
        colsep="3pt")


def table_session_metrics_adhd_col(post, qtext, groups):
    """HRI version: no Q1/Q3, \\scriptsize."""
    return "session_metrics_adhd_col", _render_cross_table(
        groups, GROUP_ADHD, quartiles=False, size="\\scriptsize",
        colsep="3pt")


def _render_group_stats_table(rows, groups, *, header, quartiles, size,
                              colsep) -> str:
    """ADHD-vs-control table over per-participant values: each stat an
    'ADHD | Control' pair, closed by Mann-Whitney n (pair) and p. `rows`
    is a list of (label, Series indexed by pid, dec)."""
    body_rows = []
    for label, s, dec in rows:
        s = s.dropna()
        a = s[s.index.map(groups.get) == GROUP_ADHD]
        c = s[s.index.map(groups.get) == GROUP_CONTROL]
        dd = max(dec, 1)
        stats_ = [(a.min(), c.min(), dec)]
        if quartiles:
            stats_.append((a.quantile(.25), c.quantile(.25), dd))
        stats_.append((a.mean(), c.mean(), dd))
        stats_.append((a.median(), c.median(), dd))
        if quartiles:
            stats_.append((a.quantile(.75), c.quantile(.75), dd))
        stats_.append((a.max(), c.max(), dec))
        stats_.append((a.std(ddof=1), c.std(ddof=1), dd))
        _, p = _mwu_cells(a, c)
        body_rows.append(
            [label]
            + [_fmt_v(v, d).replace("100.0", "100")
               for x, y, d in stats_ for v in (x, y)]
            + [f"{len(a)}$|${len(c)}", _pcell(p), _welch_p(a, c)])
    stat_heads = (["Min", "$Q_1$", "Mean", "Median", "$Q_3$", "Max", "SD"]
                  if quartiles else ["Min", "Mean", "Median", "Max", "SD"])
    n_stats = len(stat_heads)
    specs = _sspecs([r[1:] for r in body_rows])
    pair_spec = "r@{\\extracolsep{0pt}\\,$|$\\,}l@{\\extracolsep{\\fill}\\hspace{5pt}}" * n_stats  # bar-anchored pairs; fill between groups
    heads = " & ".join(f"\\multicolumn{{2}}{{c}}{{{h}}}" for h in stat_heads)
    body = "\n".join(" & ".join(_scell(c2) if i else c2
                                for i, c2 in enumerate(r)) + " \\\\"
                     for r in body_rows)
    return f"""\\begingroup\\centering{size}
\\setlength{{\\tabcolsep}}{{{colsep}}}%
\\begin{{tabular*}}{{\\textwidth}}{{@{{}}l@{{\\extracolsep{{\\fill}}}}{pair_spec}c{specs[-2]}{specs[-1]}@{{}}}}
\\toprule
 & \\multicolumn{{{2 * n_stats}}}{{c}}{{{header}}} &
   \\multicolumn{{3}}{{c}}{{Between groups}} \\\\
\\cmidrule(lr){{2-{2 * n_stats + 1}}} \\cmidrule(lr){{{2 * n_stats + 2}-{2 * n_stats + 4}}}
 & {heads} & $n$ & {{$p_U$}} & {{$p_t$}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular*}}\\par\\endgroup"""


def _fixed_condition_rows(groups, cond):
    """One condition's per-participant values for every metric row."""
    return [(label, df[cond], dec)
            for label, df, dec in _cross_condition_rows(groups)]


def _delta_rows(groups):
    """Per-participant robot-minus-control deltas for every metric row.
    Duration deltas are rounded to whole seconds for width."""
    out = []
    for label, df, dec in _cross_condition_rows(groups):
        d = df["Robot"] - df["Control"]
        if label.endswith("(s)"):
            d, dec = d.round(0), 0
        out.append((label, d, dec))
    return out


def table_metrics_robot_by_group(post, qtext, groups):
    """Thesis: robot-session values, ADHD vs control, MWU."""
    return "session_metrics_robot_by_group", _render_group_stats_table(
        _fixed_condition_rows(groups, "Robot"), groups,
        header="ADHD\\,$|$\\,No-ADHD (Robot session)", quartiles=False,
        size="\\footnotesize", colsep="3pt")


def table_metrics_robot_by_group_col(post, qtext, groups):
    """HRI variant of the robot-session group table."""
    return "session_metrics_robot_by_group_col", _render_group_stats_table(
        _fixed_condition_rows(groups, "Robot"), groups,
        header="ADHD\\,$|$\\,No-ADHD (Robot session)", quartiles=False,
        size="\\scriptsize", colsep="3pt")


def table_metrics_control_by_group(post, qtext, groups):
    """Thesis: control-session values, ADHD vs control, MWU."""
    return "session_metrics_control_by_group", _render_group_stats_table(
        _fixed_condition_rows(groups, "Control"), groups,
        header="ADHD\\,$|$\\,No-ADHD (No-Robot session)", quartiles=False,
        size="\\footnotesize", colsep="3pt")


def table_metrics_control_by_group_col(post, qtext, groups):
    """HRI variant of the control-session group table."""
    return "session_metrics_control_by_group_col", _render_group_stats_table(
        _fixed_condition_rows(groups, "Control"), groups,
        header="ADHD\\,$|$\\,No-ADHD (No-Robot session)", quartiles=False,
        size="\\scriptsize", colsep="3pt")


def table_metrics_delta_by_group(post, qtext, groups):
    """Thesis: robot-minus-control deltas, ADHD vs control, MWU — the
    nonparametric group-by-condition interaction check."""
    return "session_metrics_delta_by_group", _render_group_stats_table(
        _delta_rows(groups), groups,
        header="ADHD\\,$|$\\,No-ADHD ($\\Delta$ Robot $-$ No-Robot)",
        quartiles=False, size="\\footnotesize", colsep="3pt")


def table_metrics_delta_by_group_col(post, qtext, groups):
    """HRI variant of the delta table."""
    return "session_metrics_delta_by_group_col", _render_group_stats_table(
        _delta_rows(groups), groups,
        header="ADHD\\,$|$\\,No-ADHD ($\\Delta$ Robot $-$ No-Robot)",
        quartiles=False, size="\\scriptsize", colsep="3pt")


def table_session_metrics_ctrl(post, qtext, groups):
    """Thesis version: control-group robot-vs-control counterpart."""
    return "session_metrics_ctrl", _render_cross_table(
        groups, GROUP_CONTROL, quartiles=False, size="\\footnotesize",
        colsep="3pt")


def table_session_metrics_ctrl_col(post, qtext, groups):
    """HRI version of the control-group table."""
    return "session_metrics_ctrl_col", _render_cross_table(
        groups, GROUP_CONTROL, quartiles=False, size="\\scriptsize",
        colsep="3pt")


# Full labels for the full-width DiD variant (the compact HRI-col
# variant keeps the abbreviated forms).
_DID_FULL_LABELS = {
    "Engagement interv.": "Engagement interventions",
    "Emotion interv.": "Emotion interventions",
    "All interv.": "All interventions",
    "Mean neg.-emotion share": "Mean negative-emotion share",
    "Diseng.\\ episode duration (s)": "Disengagement episode duration (s)",
    "Neg.-emo.\\ episode duration (s)": "Negative-emotion episode duration (s)",
    "Time within eng.\\ threshold (\\%)":
        "Time within engagement threshold (\\%)",
    "Time within emo.\\ threshold (\\%)":
        "Time within emotion threshold (\\%)",
}


def _render_did_table(groups, *, size, colsep, full_width=False) -> str:
    """Consolidated factorial summary (layout decided 02.09, user's
    structure): per group a Robot / No-Robot mean pair plus that group's
    paired Wilcoxon p, then a Mann-Whitney block with the ADHD-vs-No-ADHD
    p at each condition and on the robot-minus-control deltas (the
    interaction). Nine numbers per row = the complete 2x2 report; all of
    them also appear in the per-comparison tables. Five separate rank
    tests, not one model — say so in the caption."""
    body_rows = []
    for label, df, dec in _cross_condition_rows(groups):
        dd = max(dec, 1)
        is_a = df.index.map(groups.get) == GROUP_ADHD
        delta = df["Robot"] - df["Control"]
        cells = []
        for sel in (is_a, ~is_a):
            sub = df.loc[sel]
            cells += [_scell(_fmt_v(sub[c].dropna().mean(), dd))
                      for c in ("Robot", "Control")]
            _, p_w, n_w = _wilcoxon_cells(sub["Robot"], sub["Control"])
            cells += [_scell(str(n_w)), _pcell(p_w),
                      _pairedt_p(sub["Robot"], sub["Control"])]
        for ser in (df["Robot"], df["Control"], delta):
            _, p_u = _mwu_cells(ser[is_a], ser[~is_a])
            cells += [_pcell(p_u), _welch_p(ser[is_a], ser[~is_a])]
        # MWU ns equal the per-condition value counts (visible in the
        # by-group tables); only the paired-Wilcoxon ns are shown here.
        if full_width:
            label = _DID_FULL_LABELS.get(label, label)
        body_rows.append([label] + cells)
    specs = _sspecs([r[1:] for r in body_rows])
    body = "\n".join(" & ".join(r) + " \\\\" for r in body_rows)
    norob = "No-Robot" if full_width else "No-rob."
    if full_width:
        # landscape (decided 04.09): the thesis wraps this in a
        # sidewaystable (rotating pkg), so the target width is \textheight
        env, env_arg = "tabular*", "{\\textheight}"
        colspec = ("@{}l@{\\extracolsep{\\fill}}" + "".join(specs) + "@{}")
    else:
        env, env_arg = "tabular", ""
        colspec = "l" + "".join(specs)
    return f"""\\begingroup\\centering{size}
\\setlength{{\\tabcolsep}}{{{colsep}}}%
\\begin{{{env}}}{env_arg}{{{colspec}}}
\\toprule
 & \\multicolumn{{5}}{{c}}{{ADHD (within)}} &
   \\multicolumn{{5}}{{c}}{{No-ADHD (within)}} &
   \\multicolumn{{6}}{{c}}{{Between groups}} \\\\
\\cmidrule(lr){{2-6}} \\cmidrule(lr){{7-11}} \\cmidrule(lr){{12-17}}
 &  &  &  &  &  &  &  &  &  &  & \\multicolumn{{2}}{{c}}{{Robot}} &
   \\multicolumn{{2}}{{c}}{{{norob}}} &
   \\multicolumn{{2}}{{c}}{{$\\Delta$}} \\\\
 & {_sheads(["Robot", norob, "$n$", "$p_W$", "$p_t$", "Robot", norob,
             "$n$", "$p_W$", "$p_t$", "$p_U$", "$p_t$", "$p_U$", "$p_t$",
             "$p_U$", "$p_t$"])} \\\\
\\midrule
{body}
\\bottomrule
\\end{{{env}}}\\par\\endgroup"""


def table_metrics_did(post, qtext, groups):
    """Thesis version of the consolidated factorial (DiD) summary."""
    return "session_metrics_did", _render_did_table(
        groups, size="\\footnotesize", colsep="1pt", full_width=True)


def table_metrics_did_col(post, qtext, groups):
    """HRI version of the consolidated factorial (DiD) summary."""
    return "session_metrics_did_col", _render_did_table(
        groups, size="\\scriptsize", colsep="3pt")


def _clip_len(intervals, lo: float, hi: float) -> float:
    """Total length of the union of intervals clipped to [lo, hi]."""
    clipped = [(max(s, lo), min(e, hi)) for s, e in intervals
               if min(e, hi) > max(s, lo)]
    return _interval_union(clipped)


def _half_split_rows(groups, members) -> list[tuple[str, dict, int]]:
    """Per-(pid, cond, half) values for the vigilance-decrement tables:
    intervention counts, mean raw signal values, and %-time within
    thresholds, split at the nominal session midpoint (22.5 min).
    Episodes are computed on the full session and their intervals clipped
    to each half, so episodes straddling the midpoint contribute to both.
    Episode-duration rows are omitted (too sparse when halved). The 80%
    coverage gate applies per whole (session, signal), as everywhere."""
    sdirs = session_dirs()
    gated = gated_signals(sdirs)
    mid, end = SESSION_MAX_MIN * 60 / 2, SESSION_MAX_MIN * 60
    halves = ((1, 0.0, mid), (2, mid, end))
    rows = {label: {} for label in (
        "Engagement interv.", "Emotion interv.", "All interv.",
        "Mean engagement score", "Mean neg.-emotion share",
        "Time within eng.\\ threshold (\\%)",
        "Time within emo.\\ threshold (\\%)",
        "Time within both thresholds (\\%)")}
    for (pid, cond), d in sdirs.items():
        if pid not in members:
            continue
        # intervention counts per half (same event kinds as session_metrics)
        for base, label in (("engagement", "Engagement interv."),
                            ("emotion", "Emotion interv.")):
            kind = (f"intervention_{base}_sent" if cond == "Robot"
                    else f"counterfactual_{base}")
            ts = [float(r["t_session_s"]) for r in _log_rows(d, "events.csv")
                  if r["event_type"] == kind and r["t_session_s"]]
            for h, lo, hi in halves:
                rows[label][(pid, cond, h)] = sum(1 for t in ts if lo <= t < hi)
        for h, _, _ in halves:
            rows["All interv."][(pid, cond, h)] = (
                rows["Engagement interv."][(pid, cond, h)]
                + rows["Emotion interv."][(pid, cond, h)])
        # signal metrics per half over the speech-excluded timeline
        # (decided 05.09): quiet means restricted to the half window,
        # %-within via quiet_within_pct with the half as an extra clip.
        excl = speech_exclusions(d)
        for sig, csv, col, label in (
                ("eng", "engagement.csv", "score", "Mean engagement score"),
                ("emo", "emotion.csv", "negative_share",
                 "Mean neg.-emotion share")):
            if (pid, cond, sig) in gated:
                continue
            pts = [(float(r["t_session_s"]), float(r[col]))
                   for r in _log_rows(d, csv)
                   if r.get(col) and r.get("t_session_s")
                   and float(r["t_session_s"]) >= 0]
            pts = [(t, v) for t, v in pts
                   if not any(s0 <= t <= s1 for s0, s1 in excl)]
            for h, lo, hi in halves:
                xs = [v for t, v in pts if lo <= t < hi]
                if xs:
                    rows[label][(pid, cond, h)] = sum(xs) / len(xs)
        eng_ok = (pid, cond, "eng") not in gated
        emo_ok = (pid, cond, "emo") not in gated
        for sig, ok, label in (
                ("eng", eng_ok, "Time within eng.\\ threshold (\\%)"),
                ("emo", emo_ok, "Time within emo.\\ threshold (\\%)")):
            if not ok:
                continue
            for h, lo, hi in halves:
                v = quiet_within_pct(d, (sig,), excl, lo=lo, hi=hi)
                if v is not None:
                    rows[label][(pid, cond, h)] = v
        if eng_ok and emo_ok:
            for h, lo, hi in halves:
                v = quiet_within_pct(d, ("eng", "emo"), excl, lo=lo, hi=hi)
                if v is not None:
                    rows["Time within both thresholds (\\%)"][(pid, cond, h)] = v
    decs = {"Mean engagement score": 2, "Mean neg.-emotion share": 2,
            "Engagement interv.": 0, "Emotion interv.": 0, "All interv.": 0}
    return [(label, vals, decs.get(label, 1))
            for label, vals in rows.items()]


def _render_halves_table(groups, member_group, *, size, colsep) -> str:
    """First-half vs second-half table for one participant set: per half
    a Robot / No-Robot mean pair with that half's paired Wilcoxon p, then
    a Wilcoxon block testing 1st vs 2nd within each condition and on the
    per-participant half-deltas (the half-by-condition interaction). All
    seven tests are within-subject -> Wilcoxon signed-rank throughout
    (Mann-Whitney would wrongly treat the halves as independent)."""
    members = (set(groups) if member_group is None
               else {p for p, g in groups.items() if g == member_group})
    body_rows = []
    for label, vals, dec in _half_split_rows(groups, members):
        dd = max(dec, 1)
        ser = {(c, h): pd.Series({pid: v for (pid, cc, hh), v in vals.items()
                                  if cc == c and hh == h})
               for c in ("Robot", "Control") for h in (1, 2)}
        cells = []
        for h in (1, 2):
            r, c = ser[("Robot", h)], ser[("Control", h)]
            cells += [_scell(_fmt_v(r.mean() if len(r) else None, dd)),
                      _scell(_fmt_v(c.mean() if len(c) else None, dd))]
            _, p, n = _wilcoxon_cells(r, c)
            cells += [_pcell(p), _pairedt_p(r, c)]
        for a, b in ((ser[("Robot", 1)], ser[("Robot", 2)]),
                     (ser[("Control", 1)], ser[("Control", 2)])):
            _, p, _ = _wilcoxon_cells(a, b)
            cells += [_pcell(p), _pairedt_p(a, b)]
        dr = ser[("Robot", 2)] - ser[("Robot", 1)]
        dc = ser[("Control", 2)] - ser[("Control", 1)]
        _, p, n = _wilcoxon_cells(dr, dc)
        cells += [_pcell(p), _pairedt_p(dr, dc), _scell(str(n))]
        body_rows.append([label] + cells)
    specs = _sspecs([r[1:] for r in body_rows])
    body = "\n".join(" & ".join(r) + " \\\\" for r in body_rows)
    return f"""\\begingroup\\centering{size}
\\setlength{{\\tabcolsep}}{{{colsep}}}%
\\begin{{tabular*}}{{\\textwidth}}{{@{{}}l@{{\\extracolsep{{\\fill}}}}{"".join(specs)}@{{}}}}
\\toprule
 & \\multicolumn{{4}}{{c}}{{First half}} &
   \\multicolumn{{4}}{{c}}{{Second half}} &
   \\multicolumn{{7}}{{c}}{{1st vs 2nd (within-subject)}} \\\\
\\cmidrule(lr){{2-5}} \\cmidrule(lr){{6-9}} \\cmidrule(lr){{10-16}}
 &  &  &  &  &  &  &  &  & \\multicolumn{{2}}{{c}}{{Robot}} &
   \\multicolumn{{2}}{{c}}{{No-rob.}} &
   \\multicolumn{{2}}{{c}}{{$\\Delta$}} &  \\\\
 & {_sheads(["Robot", "No-rob.", "$p_W$", "$p_t$", "Robot",
             "No-rob.", "$p_W$", "$p_t$", "$p_W$", "$p_t$",
             "$p_W$", "$p_t$", "$p_W$", "$p_t$", "$n$"])} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular*}}\\par\\endgroup"""


def table_metrics_halves_adhd(post, qtext, groups):
    """Thesis appendix: session halves, ADHD participants."""
    return "session_metrics_halves_adhd", _render_halves_table(
        groups, GROUP_ADHD, size="\\scriptsize", colsep="0.7pt")


def table_metrics_halves_noadhd(post, qtext, groups):
    """Thesis appendix: session halves, No-ADHD participants."""
    return "session_metrics_halves_noadhd", _render_halves_table(
        groups, GROUP_CONTROL, size="\\scriptsize", colsep="0.7pt")


def table_metrics_halves_all(post, qtext, groups):
    """Thesis appendix: session halves, all participants."""
    return "session_metrics_halves_all", _render_halves_table(
        groups, None, size="\\scriptsize", colsep="0.7pt")


def _pooled_rows(groups):
    """Per-participant mean across both conditions for every metric row
    (Nicole 03.09: ADHD-vs-control 'independent of condition'). A
    participant with one gated signal-session contributes the other
    session's value alone (mean skips NaN)."""
    return [(label, df.mean(axis=1), dec)
            for label, df, dec in _cross_condition_rows(groups)]


def table_metrics_all(post, qtext, groups):
    """Thesis: robot vs control paired over ALL participants (Nicole
    03.09: 'the robot might help anyone')."""
    return "session_metrics_all", _render_cross_table(
        groups, None, quartiles=True, size="\\scriptsize",
        colsep="1.5pt")


def table_metrics_all_col(post, qtext, groups):
    """HRI-sized variant of the all-participants cross table."""
    return "session_metrics_all_col", _render_cross_table(
        groups, None, quartiles=False, size="\\scriptsize", colsep="3pt")


def table_metrics_pooled_by_group(post, qtext, groups):
    """Thesis: ADHD vs control on per-participant means pooled across
    both conditions."""
    return "session_metrics_pooled_by_group", _render_group_stats_table(
        _pooled_rows(groups), groups,
        header="ADHD\\,$|$\\,No-ADHD (conditions pooled)", quartiles=False,
        size="\\footnotesize", colsep="3pt")


_GREYRULE = ("\\arrayrulecolor{black!25}\\cmidrule{{1-{n}}}"
             "\\arrayrulecolor{black}")


def _render_within_scopes_table(groups, *, size, colsep) -> str:
    """Consolidated within-subjects table (user layout 05.09 v2): each
    metric is a full-width bold header line, followed by All / ADHD /
    No-ADHD scope rows; Robot|No-Robot stat pairs without quartiles."""
    scopes = [("All", None), ("ADHD", GROUP_ADHD), ("No-ADHD", GROUP_CONTROL)]
    per_scope = {name: {l: (df, dec) for l, df, dec
                        in _cross_condition_rows(groups, g)}
                 for name, g in scopes}
    labels = [l for l, _, _ in _cross_condition_rows(groups)]
    blocks = []
    for label in labels:
        rows = []
        for name, _ in scopes:
            df, dec = per_scope[name][label]
            df = df.dropna()
            r, c = df["Robot"], df["Control"]
            dd = max(dec, 1)
            stats_ = [(r.min(), c.min(), dec), (r.mean(), c.mean(), dd),
                      (r.median(), c.median(), dd), (r.max(), c.max(), dec),
                      (r.std(ddof=1), c.std(ddof=1), dd)]
            _, p, n = _wilcoxon_cells(r, c)
            rows.append([name]
                        + [_scell(_fmt_v(x, d).replace("100.0", "100"))
                           for a, b, d in stats_ for x in (a, b)]
                        + [_scell(str(n)), _pcell(p), _pairedt_p(r, c)])
        blocks.append((_DID_FULL_LABELS.get(label, label), rows))
    specs = _sspecs([r[1:] for _, rows in blocks for r in rows])
    # pair halves are S columns (decimal-aligned) with a thin-space
    # margin around the bar; \extracolsep zeroed inside the bar boundary
    # and restored after so the fill only stretches between stat groups.
    pair_spec = "".join(
        f"{sl}@{{\\extracolsep{{0pt}}\\,$|$\\,}}{sr}"
        "@{\\extracolsep{\\fill}\\hspace{5pt}}"
        for sl, sr in zip(specs[0:10:2], specs[1:10:2]))
    ncols = 15
    lines = []
    for bi, (label, rows) in enumerate(blocks):
        if bi:
            lines.append("\\arrayrulecolor{black!25}"
                         f"\\cmidrule{{1-{ncols}}}"
                         "\\arrayrulecolor{black}")
        for i, r in enumerate(rows):
            lab = (f"\\multirow[t]{{{len(rows)}}}{{2.35cm}}"
                   f"{{\\raggedright {label}}}" if i == 0 else "")
            lines.append(f"{lab} & " + " & ".join(r) + " \\\\")
    body = "\n".join(lines)
    return f"""\\begingroup\\centering{size}
\\setlength{{\\tabcolsep}}{{{colsep}}}%
\\setlength{{\\aboverulesep}}{{0.15ex}}\\setlength{{\\belowrulesep}}{{0.3ex}}%
\\begin{{tabular*}}{{\\textheight}}{{@{{}}ll@{{\\extracolsep{{\\fill}}}}{pair_spec}{specs[-3]}{specs[-2]}{specs[-1]}@{{}}}}
\\toprule
 & & \\multicolumn{{10}}{{c}}{{Robot\\,$|$\\,No-Robot}} &
   \\multicolumn{{3}}{{c}}{{Within-subject}} \\\\
\\cmidrule(lr){{3-12}} \\cmidrule(lr){{13-15}}
Measure & Scope & \\multicolumn{{2}}{{c}}{{Min}} &
  \\multicolumn{{2}}{{c}}{{Mean}} & \\multicolumn{{2}}{{c}}{{Median}} &
  \\multicolumn{{2}}{{c}}{{Max}} & \\multicolumn{{2}}{{c}}{{SD}} &
  {{$n$}} & {{$p_W$}} & {{$p_t$}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular*}}\\par\\endgroup"""


def table_metrics_within_combined(post, qtext, groups):
    """Consolidated within-subjects table (All/ADHD/No-ADHD sub-rows)."""
    return "session_metrics_within_combined", _render_within_scopes_table(
        groups, size="\\footnotesize", colsep="3pt")


def _render_between_scopes_table(groups, *, size, colsep) -> str:
    """Consolidated between-subjects table (v2): full-width metric
    header lines; Robot / No-Robot / Delta scope rows below each."""
    sources = [("Robot", {l: (sr, dec) for l, sr, dec
                          in _fixed_condition_rows(groups, "Robot")}),
               ("No-Robot", {l: (sr, dec) for l, sr, dec
                             in _fixed_condition_rows(groups, "Control")}),
               ("$\\Delta$", {l: (sr, dec) for l, sr, dec
                              in _delta_rows(groups)})]
    labels = [l for l, _, _ in _cross_condition_rows(groups)]
    blocks = []
    for label in labels:
        rows = []
        for name, rowmap in sources:
            sr, dec = rowmap[label]
            sr = sr.dropna()
            a = sr[sr.index.map(groups.get) == GROUP_ADHD]
            c = sr[sr.index.map(groups.get) == GROUP_CONTROL]
            dd = max(dec, 1)
            stats_ = [(a.min(), c.min(), dec), (a.mean(), c.mean(), dd),
                      (a.median(), c.median(), dd), (a.max(), c.max(), dec),
                      (a.std(ddof=1), c.std(ddof=1), dd)]
            _, p = _mwu_cells(a, c)
            rows.append([name]
                        + [_scell(_fmt_v(x, d).replace("100.0", "100"))
                           for x1, y1, d in stats_ for x in (x1, y1)]
                        + [_scell(f"{len(a)}$|${len(c)}"), _pcell(p),
                           _welch_p(a, c)])
        blocks.append((_DID_FULL_LABELS.get(label, label), rows))
    specs = _sspecs([r[1:] for _, rows in blocks for r in rows])
    # decimal-aligned pair halves, thin-space bar margin (see within table)
    pair_spec = "".join(
        f"{sl}@{{\\extracolsep{{0pt}}\\,$|$\\,}}{sr}"
        "@{\\extracolsep{\\fill}\\hspace{5pt}}"
        for sl, sr in zip(specs[0:10:2], specs[1:10:2]))
    ncols = 15
    lines = []
    for bi, (label, rows) in enumerate(blocks):
        if bi:
            lines.append("\\arrayrulecolor{black!25}"
                         f"\\cmidrule{{1-{ncols}}}"
                         "\\arrayrulecolor{black}")
        for i, r in enumerate(rows):
            lab = (f"\\multirow[t]{{{len(rows)}}}{{2.35cm}}"
                   f"{{\\raggedright {label}}}" if i == 0 else "")
            lines.append(f"{lab} & " + " & ".join(r) + " \\\\")
    body = "\n".join(lines)
    return f"""\\begingroup\\centering{size}
\\setlength{{\\tabcolsep}}{{{colsep}}}%
\\setlength{{\\aboverulesep}}{{0.15ex}}\\setlength{{\\belowrulesep}}{{0.3ex}}%
\\begin{{tabular*}}{{\\textheight}}{{@{{}}ll@{{\\extracolsep{{\\fill}}}}{pair_spec}c{specs[-2]}{specs[-1]}@{{}}}}
\\toprule
 & & \\multicolumn{{10}}{{c}}{{ADHD\\,$|$\\,No-ADHD}} &
   \\multicolumn{{3}}{{c}}{{Between groups}} \\\\
\\cmidrule(lr){{3-12}} \\cmidrule(lr){{13-15}}
Measure & Scope & \\multicolumn{{2}}{{c}}{{Min}} &
  \\multicolumn{{2}}{{c}}{{Mean}} & \\multicolumn{{2}}{{c}}{{Median}} &
  \\multicolumn{{2}}{{c}}{{Max}} & \\multicolumn{{2}}{{c}}{{SD}} &
  $n$ & {{$p_U$}} & {{$p_t$}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular*}}\\par\\endgroup"""


def table_metrics_between_combined(post, qtext, groups):
    """Consolidated between-subjects table (Robot/No-Robot/Delta rows)."""
    return "session_metrics_between_combined", _render_between_scopes_table(
        groups, size="\\footnotesize", colsep="3pt")


def _render_halves_scopes_table(groups, *, size, colsep) -> str:
    """Consolidated halves table (v2): full-width metric header lines;
    All / ADHD / No-ADHD scope rows with the halves column structure."""
    scopes = [("All", set(groups)),
              ("ADHD", {p for p, g in groups.items() if g == GROUP_ADHD}),
              ("No-ADHD", {p for p, g in groups.items()
                           if g == GROUP_CONTROL})]
    per_scope = {name: {l: (vals, dec) for l, vals, dec
                        in _half_split_rows(groups, members)}
                 for name, members in scopes}
    labels = [l for l, _, _ in _half_split_rows(groups, set(groups))]
    blocks = []
    for label in labels:
        rows = []
        for name, _ in scopes:
            vals, dec = per_scope[name][label]
            dd = max(dec, 1)
            ser = {(c2, h): pd.Series(
                {pid: v for (pid, cc, hh), v in vals.items()
                 if cc == c2 and hh == h})
                for c2 in ("Robot", "Control") for h in (1, 2)}
            cells = [name]
            for h in (1, 2):
                r, c2 = ser[("Robot", h)], ser[("Control", h)]
                cells += [_scell(_fmt_v(r.mean() if len(r) else None, dd)),
                          _scell(_fmt_v(c2.mean() if len(c2) else None, dd))]
                _, p, _ = _wilcoxon_cells(r, c2)
                cells += [_pcell(p), _pairedt_p(r, c2)]
            for a, b in ((ser[("Robot", 1)], ser[("Robot", 2)]),
                         (ser[("Control", 1)], ser[("Control", 2)])):
                _, p, _ = _wilcoxon_cells(a, b)
                cells += [_pcell(p), _pairedt_p(a, b)]
            dr = ser[("Robot", 2)] - ser[("Robot", 1)]
            dc = ser[("Control", 2)] - ser[("Control", 1)]
            _, p, n = _wilcoxon_cells(dr, dc)
            cells += [_pcell(p), _pairedt_p(dr, dc), _scell(str(n))]
            rows.append(cells)
        blocks.append((_DID_FULL_LABELS.get(label, label), rows))
    specs = _sspecs([r[1:] for _, rows in blocks for r in rows])
    ncols = 17
    lines = []
    for bi, (label, rows) in enumerate(blocks):
        if bi:
            lines.append("\\arrayrulecolor{black!25}"
                         f"\\cmidrule{{1-{ncols}}}"
                         "\\arrayrulecolor{black}")
        for i, r in enumerate(rows):
            lab = (f"\\multirow[t]{{{len(rows)}}}{{1.9cm}}"
                   f"{{\\raggedright {label}}}" if i == 0 else "")
            lines.append(f"{lab} & " + " & ".join(r) + " \\\\")
    body = "\n".join(lines)
    return f"""\\begingroup\\centering{size}
\\setlength{{\\tabcolsep}}{{{colsep}}}%
\\setlength{{\\aboverulesep}}{{0.15ex}}\\setlength{{\\belowrulesep}}{{0.3ex}}%
\\begin{{tabular*}}{{\\textheight}}{{@{{}}ll@{{\\extracolsep{{\\fill}}}}{"".join(specs)}@{{}}}}
\\toprule
 & & \\multicolumn{{4}}{{c}}{{First half}} &
   \\multicolumn{{4}}{{c}}{{Second half}} &
   \\multicolumn{{7}}{{c}}{{1st vs 2nd (within-subject)}} \\\\
\\cmidrule(lr){{3-6}} \\cmidrule(lr){{7-10}} \\cmidrule(lr){{11-17}}
 & & & & & & & & & & \\multicolumn{{2}}{{c}}{{Robot}} &
   \\multicolumn{{2}}{{c}}{{No-rob.}} &
   \\multicolumn{{2}}{{c}}{{$\\Delta$}} & \\\\
Measure & Scope & {_sheads(["Robot", "No-rob.", "$p_W$", "$p_t$", "Robot",
             "No-rob.", "$p_W$", "$p_t$", "$p_W$", "$p_t$",
             "$p_W$", "$p_t$", "$p_W$", "$p_t$", "$n$"])} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular*}}\\par\\endgroup"""


def table_metrics_halves_combined(post, qtext, groups):
    """Consolidated halves table (All/ADHD/No-ADHD sub-rows)."""
    return "session_metrics_halves_combined", _render_halves_scopes_table(
        groups, size="\\footnotesize", colsep="2pt")


def table_metrics_halves_pooled(post, qtext, groups):
    """Thesis appendix: first vs second half pooled over conditions AND
    groups (Nicole/outline 04.09) — per participant and half, the mean of
    the robot and control values; paired 1st-vs-2nd across all 22."""
    body_rows = []
    for label, vals, dec in _half_split_rows(groups, set(groups)):
        dd = max(dec, 1)
        halves = {}
        for h in (1, 2):
            sub = pd.Series({(pid, cond): v
                             for (pid, cond, hh), v in vals.items()
                             if hh == h})
            halves[h] = sub.groupby(level=0).mean() if len(sub) else \
                pd.Series(dtype=float)
        h1, h2 = halves[1], halves[2]
        _, p, n = _wilcoxon_cells(h1, h2)
        body_rows.append([
            label,
            _scell(_fmt_v(h1.mean() if len(h1) else None, dd)),
            _scell(_fmt_v(h2.mean() if len(h2) else None, dd)),
            _scell(str(n)), _pcell(p), _pairedt_p(h1, h2)])
    specs = _sspecs([r[1:] for r in body_rows])
    body = "\n".join(" & ".join(r) + " \\\\" for r in body_rows)
    tex = f"""\\begingroup\\centering\\footnotesize
\\setlength{{\\tabcolsep}}{{4pt}}%
\\begin{{tabular}}{{l{"".join(specs)}}}
\\toprule
 & \\multicolumn{{2}}{{c}}{{Mean (all participants)}} &
   \\multicolumn{{3}}{{c}}{{1st vs 2nd (within)}} \\\\
\\cmidrule(lr){{2-3}} \\cmidrule(lr){{4-6}}
 & {{First half}} & {{Second half}} & {{$n$}} & {{$p_W$}} & {{$p_t$}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}\\par\\endgroup"""
    return "session_metrics_halves_pooled", tex


def _render_feature_stats(post, qtext, groups, *, size, colsep) -> str:
    """Companion stats table to the feature-means chart. Layout decided
    04.09: rotated block labels on the left (mirroring the chart's
    bracket labels), group ns under the column labels, light-grey rules
    between blocks. Needs multirow + graphicx + colortbl."""
    blocks, ns = [], (0, 0)
    pids_a = [p for p, g in groups.items() if g == GROUP_ADHD]
    for title, prefix, n_items in FEATURE_BLOCKS:
        items = []
        for i in range(1, n_items + 1):
            col = f"{prefix}{i}"
            if col not in post.columns:
                continue
            v = to_rank(post[col], "LIKERT5")
            a = v[post["PID"].isin(pids_a)]
            c = v[~post["PID"].isin(pids_a)]
            _, p_u = _mwu_cells(a, c)
            items.append((
                esc(SHORT_LABELS.get(col, strip_stem(qtext.get(col, col)))),
                f"{a.mean():.2f}", f"{c.mean():.2f}",
                _pcell(p_u), _welch_p(a, c)))
            ns = (a.dropna().size, c.dropna().size)
        blocks.append((title, items))
    lines = []
    for bi, (title, items) in enumerate(blocks):
        stack = title.replace(" ", "\\\\")
        for i, (lbl, ma, mc, pu, pt) in enumerate(items):
            rot = ""
            if i == 0:
                rot = (f"\\multirow{{{len(items)}}}{{*}}"
                       f"{{\\rotatebox[origin=c]{{90}}{{\\tiny"
                       f"\\bfseries\\shortstack{{{stack}}}}}}}")
            lines.append(f"{rot} & {lbl} & {ma} & {mc} & {pu} & {pt} \\\\")
        if bi < len(blocks) - 1:
            lines.append("\\arrayrulecolor{black!25}\\cmidrule{2-6}"
                         "\\arrayrulecolor{black}")
    body = "\n".join(lines)
    na, nc = ns
    return f"""\\begingroup\\centering{size}
\\renewcommand{{\\arraystretch}}{{1.2}}%
\\setlength{{\\tabcolsep}}{{{colsep}}}%
\\begin{{tabular}}{{clS[table-format=1.2]S[table-format=1.2]S[table-format=1.3]S[table-format=1.3]}}
\\toprule
 & & \\multicolumn{{2}}{{c}}{{Mean}} & \\multicolumn{{2}}{{c}}{{$p$}} \\\\
\\cmidrule(lr){{3-4}} \\cmidrule(lr){{5-6}}
 & & {{ADHD}} & {{No-ADHD}} & {{$p_U$}} & {{$p_t$}} \\\\
 & & {{($n={na}$)}} & {{($n={nc}$)}} & & \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}\\par\\endgroup"""


def table_feature_stats(post, qtext, groups):
    """Thesis companion table to the feature-means chart."""
    return "feature_stats", _render_feature_stats(
        post, qtext, groups, size="\\footnotesize", colsep="4pt")


def table_feature_stats_col(post, qtext, groups):
    """HRI-sized variant of the feature stats table."""
    return "feature_stats_col", _render_feature_stats(
        post, qtext, groups, size="\\scriptsize", colsep="3pt")


def table_metrics_pooled_by_group_col(post, qtext, groups):
    """HRI-sized variant of the pooled group table."""
    return "session_metrics_pooled_by_group_col", _render_group_stats_table(
        _pooled_rows(groups), groups,
        header="ADHD\\,$|$\\,No-ADHD (conditions pooled)", quartiles=False,
        size="\\scriptsize", colsep="3pt")


def _reengagement_gaps(groups) -> pd.DataFrame:
    """Post-recovery engaged gaps (Nicole 03.09, 'guilt effect'): for
    every episode with an observed recovery, the time until the same
    signal's next episode starts (or until the last poll = right-censored
    at session end). Cue attribution matches episode_records (same event
    kinds, same -5s tolerance); the coverage gate applies."""
    sdirs = session_dirs()
    gated = gated_signals(sdirs)
    recs = []
    for (pid, cond), d in sdirs.items():
        if pid not in groups:
            continue
        events = _log_rows(d, "events.csv")
        for sig in ("eng", "emo"):
            if (pid, cond, sig) in gated:
                continue
            polls = signal_polls(d, sig)
            eps = extract_episodes(polls)
            if not polls or not eps:
                continue
            base = "engagement" if sig == "eng" else "emotion"
            kind = (f"intervention_{base}" if cond == "Robot"
                    else f"counterfactual_{base}")
            cues = [float(r["t_session_s"]) for r in events
                    if r["event_type"] == kind and r["t_session_s"]]
            t_last = polls[-1][0]
            for i, e in enumerate(eps):
                if e["t1"] is None:
                    continue
                cued = any(e["t0"] - 5 <= t <= e["t1"] for t in cues)
                nxt = eps[i + 1]["t0"] if i + 1 < len(eps) else None
                recs.append({
                    "pid": pid, "cond": cond, "sig": sig, "cued": cued,
                    "gap": (nxt if nxt is not None else t_last) - e["t1"],
                    "censored": nxt is None,
                })
    return pd.DataFrame(recs)


_DIALOGUE_CHAIN_GAP_S = 10.0  # speech gap that still counts as one dialogue


def _post_cue_recovery(groups):
    """Simplified re-engagement analysis (user design 04.09).
    Robot sessions: for every intervention, wait until the ensuing
    dialogue ends (the cue utterance plus any speech, either actor,
    chained while gaps stay below _DIALOGUE_CHAIN_GAP_S), then measure
    the time until the signal is first back at threshold (0-ish if it
    recovered during the dialogue). Control sessions: plain observed
    episode durations (below-threshold start to back-at-threshold).
    NB the two clocks start at different points (post-dialogue vs
    episode start) — descriptive comparison only. Coverage gate applies;
    never-recovered cases are censored and only counted."""
    sdirs = session_dirs()
    gated = gated_signals(sdirs)
    rob, ctl, cens = [], [], 0
    for (pid, cond), d in sdirs.items():
        if pid not in groups:
            continue
        for sig in ("eng", "emo"):
            if (pid, cond, sig) in gated:
                continue
            polls = signal_polls(d, sig)
            if not polls:
                continue
            if cond == "Control":
                for e in extract_episodes(polls):
                    if e["t1"] is not None:
                        ctl.append({"pid": pid, "sig": sig,
                                    "dur": e["t1"] - e["t0"]})
                    else:
                        cens += 1
                continue
            base = "engagement" if sig == "eng" else "emotion"
            cues = [float(r["t_session_s"])
                    for r in _log_rows(d, "events.csv")
                    if r["event_type"] == f"intervention_{base}"
                    and r["t_session_s"]]
            speech = sorted(
                (float(r["t_start_s"]), float(r["t_end_s"]))
                for r in _log_rows(d, "speech.csv") if r["t_start_s"])
            for t0 in cues:
                segs = [sg for sg in speech if sg[0] >= t0]
                if not segs:
                    continue
                end = segs[0][1]
                for s0, s1 in segs[1:]:
                    if s0 - end <= _DIALOGUE_CHAIN_GAP_S:
                        end = max(end, s1)
                    else:
                        break
                rec = next((t for t, act in polls
                            if t >= end and not act), None)
                if rec is None:
                    cens += 1
                    continue
                rob.append({"pid": pid, "sig": sig,
                            "dur": max(rec - end, 0.0)})
    return pd.DataFrame(rob), pd.DataFrame(ctl), cens


def table_reengagement(post, qtext, groups):
    """Post-intervention re-engagement (simplified design 04.09): robot =
    time from end of the post-intervention dialogue to back-at-threshold;
    control = observed episode duration. Pooled descriptives plus a
    paired per-participant-median comparison."""
    rob, ctl, cens = _post_cue_recovery(groups)
    signame = {"eng": "Engagement", "emo": "Negative emotion"}
    rows = []
    for sig in ("eng", "emo"):
        r = rob[rob.sig == sig].set_index("pid").dur
        c = ctl[ctl.sig == sig].set_index("pid").dur
        mean_r, mean_c = r.groupby(level=0).mean(), c.groupby(level=0).mean()
        med_r, med_c = (r.groupby(level=0).median(),
                        c.groupby(level=0).median())
        # Wilcoxon on per-participant medians (typical case), paired t on
        # per-participant means (tail-inclusive) — see the table note.
        _, p, n = _wilcoxon_cells(med_r, med_c)
        pt = _pairedt_p(mean_r, mean_c)
        cell = (lambda st: f"{st.mean():.0f} ({st.std(ddof=1):.0f}) "
                f"[{len(st)}]" if len(st) else "--")
        rows.append(
            f"{signame[sig]} & {cell(r)} & {cell(c)} & "
            f"{mean_r.mean():.0f} & {mean_c.mean():.0f} & "
            f"{med_r.median():.0f} & {med_c.median():.0f} & {n} & "
            f"{_pcell(p)} & {pt} \\\\")
    body = "\n".join(rows)
    tex = f"""\\begingroup\\centering\\footnotesize
\\setlength{{\\tabcolsep}}{{1.5pt}}%
\\begin{{tabular}}{{lccccccccc}}
\\toprule
 & \\multicolumn{{2}}{{c}}{{Time below threshold (s), pooled}} &
   \\multicolumn{{7}}{{c}}{{Per-participant summaries (paired)}} \\\\
\\cmidrule(lr){{2-3}} \\cmidrule(lr){{4-10}}
 &  &  & \\multicolumn{{2}}{{c}}{{Mean}} &
   \\multicolumn{{2}}{{c}}{{Median}} & & & \\\\
Signal & Robot: after dialogue & No-Robot: full episode &
  Robot & No-Rob. & Robot & No-Rob. & $n$ & {{$p_W$}} & {{$p_t$}} \\\\
 & \\multicolumn{{2}}{{c}}{{mean (SD) [count]}} & & & & & & & \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}\\par\\vspace{{0.4em}}
\\noindent{{\\small Robot sessions: clock starts when the dialogue
following an intervention ends (speech chained across gaps below
{_DIALOGUE_CHAIN_GAP_S:.0f}\\,s); No-Robot sessions: clock starts at
episode onset. The two clocks therefore start at different points —
descriptive comparison only. {cens} never-recovered cases censored.
Durations are right-skewed: $p_W$ is the Wilcoxon over per-participant
medians (typical episode), $p_t$ the paired $t$ over per-participant
means (tail-inclusive) — they can disagree, and both are shown.}}\\par
\\endgroup"""
    return "session_reengagement", tex


# Improvement suggestions of ADHD participants, O'Connell-style. This is
# AUTHOR-CODED content, frozen 03.09.2026 from the ADHD open-ended
# answers (n=12; counts are participants mentioning the suggestion at
# least once) — it does NOT recompute from data. If new ADHD
# participants are added, re-code by hand and update here.
_SUGGESTIONS_N_ADHD = 12
_SUGGESTIONS = [
    ("Inattention detection", [
        ("Improve detection accuracy", 3),
        ("Add a manual override for repeated interventions", 1),
    ]),
    ("Interaction", [
        ("Provide chat-style transcripts of the conversation", 2),
        ("Reduce response latency", 1),
    ]),
    ("Personality", [
        ("Give less praise", 1),
        ("Summarise less, give more actionable advice", 1),
        ("Ask fewer questions", 2),
    ]),
    ("Appearance", [
        ("Remove the physical embodiment", 1),
        ("Add a screen for text and diagrams", 1),
    ]),
    ("Structural guidance", [
        ("Offer timer-based, regular check-ins", 2),
        ("Track time spent on specific tasks", 1),
    ]),
    ("Access", [
        ("Add internet access and multi-modal input", 1),
    ]),
]


def _render_suggestions_table(*, cat_w, stmt_w, size) -> str:
    """Category | Suggestion | Participants (layout decided 04.09):
    the category prints on its first row only; categories are ordered by
    total mention count (desc), suggestions within a category by their
    own count (desc), ties keeping the coded order."""
    cats = sorted(_SUGGESTIONS,
                  key=lambda c: -sum(n for _, n in c[1]))
    lines = []
    for category, items in cats:
        items = sorted(items, key=lambda it: -it[1])
        for i, (stmt, n) in enumerate(items):
            pct = round(100 * n / _SUGGESTIONS_N_ADHD)
            cat_cell = f"\\textbf{{{category}}}" if i == 0 else ""
            keep = "*" if i < len(items) - 1 else ""
            lines.append(f"{cat_cell} & {stmt} & {n} ({pct}\\%) "
                         f"\\\\{keep}")
        lines.append("\\arrayrulecolor{black!25}\\cmidrule{1-3}"
                     "\\arrayrulecolor{black}")
    body = "\n".join(lines[:-1])  # drop trailing rule
    return f"""\\begingroup\\centering{size}
\\begin{{tabular*}}{{\\textwidth}}{{@{{}}>{{\\raggedright\\arraybackslash}}p{{{cat_w}}}>{{\\raggedright\\arraybackslash}}p{{{stmt_w}}}@{{\\extracolsep{{\\fill}}}}r@{{}}}}
\\toprule
\\textbf{{Category}} & \\textbf{{Suggestion}} & \\textbf{{Participants}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular*}}\\par\\endgroup"""


def table_suggestions(post, qtext, groups):
    """Thesis version of the ADHD improvement-suggestions table."""
    return "improvement_suggestions", _render_suggestions_table(
        cat_w="0.26\\textwidth", stmt_w="0.52\\textwidth",
        size="\\small")


def table_suggestions_col(post, qtext, groups):
    """HRI column-width version."""
    return "improvement_suggestions_col", _render_suggestions_table(
        cat_w="0.26\\columnwidth", stmt_w="0.48\\columnwidth",
        size="\\footnotesize")


def chart_feature_means_adhd(post, qtext, groups):
    """Thesis version, ADHD group only (restored 04.09): single blue
    bars, no legend, value labels on the bar ends."""
    data = _feature_data(post, qtext, groups)
    return "feature_means_adhd", _render_feature_chart(
        data, axis_w="0.48\\textwidth", label_w="0.42\\textwidth",
        pitch="0.58cm", bar_pt=6.5, label_font="\\small",
        title_font="\\small\\bfseries", tick_anchors=True,
        value_labels=True, adhd_only=True)


def chart_feature_means_adhd_col(post, qtext, groups):
    """ACM column-width version of the ADHD-only feature chart."""
    data = _feature_data(post, qtext, groups)
    return "feature_means_adhd_col", _render_feature_chart(
        data, axis_w="0.40\\columnwidth", label_w="0.46\\columnwidth",
        pitch="0.38cm", bar_pt=4.5, label_font="\\scriptsize",
        title_font="\\scriptsize\\bfseries", tick_anchors=False,
        value_labels=True, value_font="\\tiny", xmax=5.6,
        span_ext="0.18cm", adhd_only=True)


CHART_BUILDERS = [chart_feature_means, chart_feature_means_col,
                  chart_feature_means_adhd, chart_feature_means_adhd_col,
                  table_session_metrics, table_session_metrics_col,
                  table_session_metrics_adhd, table_session_metrics_adhd_col,
                  table_session_metrics_ctrl, table_session_metrics_ctrl_col,
                  table_metrics_robot_by_group,
                  table_metrics_robot_by_group_col,
                  table_metrics_control_by_group,
                  table_metrics_control_by_group_col,
                  table_metrics_delta_by_group,
                  table_metrics_delta_by_group_col,
                  table_metrics_did, table_metrics_did_col,
                  table_metrics_halves_adhd, table_metrics_halves_noadhd,
                  table_metrics_halves_all,
                  table_metrics_all, table_metrics_all_col,
                  table_metrics_pooled_by_group,
                  table_metrics_pooled_by_group_col,
                  table_reengagement,
                  table_metrics_halves_pooled,
                  table_metrics_within_combined,
                  table_metrics_between_combined,
                  table_metrics_halves_combined,
                  table_feature_stats, table_feature_stats_col,
                  table_suggestions, table_suggestions_col]


# ============================================================================
# Main + sync
# ============================================================================

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pre, _ = load_qualtrics(newest_file(FILE_PATTERNS["pre"]))
    pre = clean(pre, "PRE_PID", "pre")
    groups = assign_groups(pre)
    post, q_post = load_qualtrics(newest_file(FILE_PATTERNS["post"]))
    post = clean(post, "POST_PID", "post")

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    header = (f"% AUTO-GENERATED by generate_results.py on {stamp}\n"
              f"% Bare fragment — wrap in a figure/table environment in the "
              f"thesis;\n% do not edit by hand, rerun the script instead.\n")
    names = []
    for builder in CHART_BUILDERS:
        name, tex = builder(post, q_post, groups)
        path = os.path.join(OUTPUT_DIR, f"{name}.tex")
        with open(path, "w") as f:
            f.write(header + tex + "\n")
        names.append(name)
        print(f"  wrote results-charts/{name}.tex")

    preview = [header, "\\section*{Results chart preview}\n"]
    for name in names:
        preview.append(f"\\subsection*{{{name}}}\n"
                       f"\\input{{results-charts/{name}}}\n\\clearpage\n")
    with open(os.path.join(OUTPUT_DIR, "results_preview.tex"), "w") as f:
        f.write("\n".join(preview))
    print(f"  wrote results-charts/results_preview.tex ({len(names)} charts)")

    if "--sync" in sys.argv:
        for target in SYNC_TARGETS:
            sync_to_repo(target, names)


# Routing (04.09): every fragment ships to both repos — the HRI repo kept
# falling behind the thesis repo, and an unused fragment there is harmless
# (nothing \inputs it). Fragments needing multirow/siunitx/colortbl/rotating
# only render once acmart's preamble loads those packages.


def sync_to_repo(charts_dir: str, names: list[str]) -> None:
    """Copy the chart fragments into one target repo's results-charts/ and
    push (same mechanics as generate_appendix.sync_to_thesis_repo; every
    target repo must stay PRIVATE)."""
    repo = os.path.dirname(charts_dir)
    if not os.path.isdir(repo):
        print(f"--sync: {repo} not found — clone it first; skipping.")
        return
    os.makedirs(charts_dir, exist_ok=True)
    ship = names + ["results_preview"]
    for name in ship:
        shutil.copy2(os.path.join(OUTPUT_DIR, f"{name}.tex"),
                     os.path.join(charts_dir, f"{name}.tex"))
    changed = subprocess.run(
        ["git", "status", "--porcelain", "--", os.path.basename(charts_dir)],
        cwd=repo, capture_output=True, text=True).stdout.strip()
    if not changed:
        print(f"--sync: {os.path.basename(repo)}: charts unchanged — nothing to push.")
        return
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    subprocess.run(["git", "add", os.path.basename(charts_dir)],
                   cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-m",
         f"chore(results): regenerate results charts ({stamp})\n\n"
         "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"],
        cwd=repo, check=True)
    subprocess.run(["git", "push"], cwd=repo, check=True)
    print(f"--sync: pushed to {os.path.basename(repo)} — in Overleaf: "
          "Menu -> GitHub -> Pull GitHub changes.")


if __name__ == "__main__":
    main()
