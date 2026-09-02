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
import shutil
import subprocess
import sys
from datetime import datetime

import pandas as pd

from generate_appendix import (
    FILE_PATTERNS, GROUP_ADHD, GROUP_CONTROL, GROUPS_FILE,  # noqa: F401
    assign_groups, clean, esc, load_qualtrics, newest_file, strip_stem,
    to_rank,
    _ROBOT_METRICS, _metric_series, _mwu_cells, _wilcoxon_cells,
    _read_rows as _log_rows,
    session_dirs, session_metrics,
    EPISODE_CENSOR_GAP_S, episode_records, extract_episodes, gated_signals,
    signal_polls,
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
                          span_ext="0.3cm", value_extra_pt=0.0):
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
    if value_labels:
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
    \\addplot[xbar, fill={CTRL_COLOR}, draw={CTRL_COLOR}!70!black] coordinates {{{coords_c}}};
    \\addplot[xbar, fill={ADHD_COLOR}, draw={ADHD_COLOR}!70!black] coordinates {{{coords_a}}};
    \\legend{{Control, ADHD}}
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


def _render_session_metrics_table(groups, *, quartiles, size, colsep) -> str:
    """Robot-session metrics table for the results sections: one row per
    metric, each stat an 'ADHD | Control' pair, closed by the Mann-Whitney
    p value (U dropped for space, decided 30.08). Each pair is a
    right/left sub-column duo with the bar as fixed inter-column material,
    so the separators align vertically across all rows. The thesis variant
    carries the quartiles; the HRI variant drops Q1/Q3 to fit half a page."""
    sdirs = session_dirs()
    met = {k: session_metrics(d, k[1]) for k, d in sdirs.items()
           if k[0] in groups}
    rows = []
    for key, label, _, dec in _ROBOT_METRICS:
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
        cells = [f"{_fmt_v(a, d)} & {_fmt_v(c, d)}" for a, c, d in stats_]
        _, p = _mwu_cells(s_a, s_c)
        p = p.replace("p = ", "").replace("p < ", "< ")  # header says $p$
        rows.append(f"{esc(_SHORT_METRIC_LABELS.get(label, label))} & "
                    + " & ".join(cells) + f" & {p} \\\\")
    stat_heads = (["Min", "$Q_1$", "Mean", "Median", "$Q_3$", "Max", "SD"]
                  if quartiles else ["Min", "Mean", "Median", "Max", "SD"])
    n_stats = len(stat_heads)
    # each stat = right-aligned ADHD half + bar + left-aligned control half
    pair_spec = "r@{\\,$|$\\,}l" * n_stats
    heads = " & ".join(f"\\multicolumn{{2}}{{c}}{{{h}}}" for h in stat_heads)
    body = "\n".join(rows)
    return f"""\\begingroup\\centering{size}
\\setlength{{\\tabcolsep}}{{{colsep}}}%
\\begin{{tabular}}{{l{pair_spec}c}}
\\toprule
 & \\multicolumn{{{2 * n_stats}}}{{c}}{{ADHD\\,$|$\\,Control}} & MWU \\\\
\\cmidrule(lr){{2-{2 * n_stats + 1}}} \\cmidrule(lr){{{2 * n_stats + 2}-{2 * n_stats + 2}}}
 & {heads} & $p$ \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}\\par\\endgroup"""


def table_session_metrics(post, qtext, groups):
    """Thesis version: full seven-stat spread including the quartiles."""
    return "session_metrics_robot", _render_session_metrics_table(
        groups, quartiles=True, size="\\footnotesize", colsep="2.5pt")


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

    # Session-mean raw signal values (engagement `score`, emotion
    # `negative_share`), gate-respecting like every signal-level row.
    for sig, csv, col, label in (
            ("eng", "engagement.csv", "score", "Mean engagement score"),
            ("emo", "emotion.csv", "negative_share",
             "Mean neg.-affect share")):
        vals = {}
        for (pid, cond), d in sdirs.items():
            if pid not in members or (pid, cond, sig) in gated:
                continue
            xs = [float(r[col]) for r in _log_rows(d, csv)
                  if r.get(col) and r.get("t_session_s")
                  and float(r["t_session_s"]) >= 0]
            if xs:
                vals[(pid, cond)] = sum(xs) / len(xs)
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
    for sig, label in (("eng", "Eng.\\ episode duration (s)"),
                       ("emo", "Emo.\\ episode duration (s)"),
                       ("any", "Any episode duration (s)")):
        if sig == "any":
            sub = obs[[(p, c) in both_ok
                       for p, c in zip(obs.pid, obs.cond)]]
        else:
            sub = obs[obs.sig == sig]
        med = sub.groupby(["pid", "cond"]).dur.median().unstack()
        med = med.reindex(columns=["Robot", "Control"])
        rows.append((label, med, 1))

    within = {}  # sig -> {(pid, cond): % within threshold}
    below_time = ep.assign(bt=ep.dur.fillna(ep.low_bound)).groupby(
        ["pid", "cond", "sig"]).bt.sum()
    for (pid, cond, sig), span in spans.items():
        if pid not in members:
            continue
        bt = below_time.get((pid, cond, sig), 0.0)
        within.setdefault(sig, {})[(pid, cond)] = 100.0 * (1 - bt / span)

    # 'both' = time in no below-threshold episode of either signal, over
    # the merged two-signal poll timeline (gaps clamped like the
    # per-signal spans); needs both signals to pass the coverage gate.
    for (pid, cond), d in sdirs.items():
        if (pid, cond) not in both_ok:
            continue
        times, intervals = [], []
        for sig in ("eng", "emo"):
            polls = signal_polls(d, sig)
            times += [t for t, _ in polls]
            intervals += [(e["t0"],
                           e["t1"] if e["t1"] is not None else e["t_last"])
                          for e in extract_episodes(polls)]
        times.sort()
        if len(times) < 2:
            continue
        span = sum(min(t2 - t1, EPISODE_CENSOR_GAP_S)
                   for t1, t2 in zip(times, times[1:]))
        within.setdefault("both", {})[(pid, cond)] = 100.0 * (
            1 - _interval_union(intervals) / span)

    for sig, label in (("eng", "Within eng.\\ threshold (\\%)"),
                       ("emo", "Within emo.\\ threshold (\\%)"),
                       ("both", "Within both thresholds (\\%)")):
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
        cells = [f"{_fmt_v(a, d)} & {_fmt_v(b, d)}".replace("100.0", "100")
                 for a, b, d in stats_]
        _, p, n = _wilcoxon_cells(r, c)
        p = p.replace("p = ", "").replace("p < ", "< ")  # header says $p$
        body_rows.append(f"{label} & " + " & ".join(cells)
                         + f" & {n} & {p} \\\\")
    stat_heads = (["Min", "$Q_1$", "Mean", "Median", "$Q_3$", "Max", "SD"]
                  if quartiles else ["Min", "Mean", "Median", "Max", "SD"])
    n_stats = len(stat_heads)
    pair_spec = "r@{\\,$|$\\,}l" * n_stats
    heads = " & ".join(f"\\multicolumn{{2}}{{c}}{{{h}}}" for h in stat_heads)
    body = "\n".join(body_rows)
    return f"""\\begingroup\\centering{size}
\\setlength{{\\tabcolsep}}{{{colsep}}}%
\\begin{{tabular}}{{l{pair_spec}cc}}
\\toprule
 & \\multicolumn{{{2 * n_stats}}}{{c}}{{Robot\\,$|$\\,Control}} &
   \\multicolumn{{2}}{{c}}{{Wilcoxon}} \\\\
\\cmidrule(lr){{2-{2 * n_stats + 1}}} \\cmidrule(lr){{{2 * n_stats + 2}-{2 * n_stats + 3}}}
 & {heads} & $n$ & $p$ \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}\\par\\endgroup"""


def table_session_metrics_adhd(post, qtext, groups):
    """Thesis version: ADHD robot-vs-control, full stat spread."""
    return "session_metrics_adhd", _render_cross_table(
        groups, GROUP_ADHD, quartiles=True, size="\\footnotesize",
        colsep="1.25pt")


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
        cells = [f"{_fmt_v(x, d)} & {_fmt_v(y, d)}".replace("100.0", "100")
                 for x, y, d in stats_]
        _, p = _mwu_cells(a, c)
        p = p.replace("p = ", "").replace("p < ", "< ")  # header says $p$
        body_rows.append(f"{label} & " + " & ".join(cells)
                         + f" & {len(a)}$|${len(c)} & {p} \\\\")
    stat_heads = (["Min", "$Q_1$", "Mean", "Median", "$Q_3$", "Max", "SD"]
                  if quartiles else ["Min", "Mean", "Median", "Max", "SD"])
    n_stats = len(stat_heads)
    pair_spec = "r@{$|$}l" * n_stats
    heads = " & ".join(f"\\multicolumn{{2}}{{c}}{{{h}}}" for h in stat_heads)
    body = "\n".join(body_rows)
    return f"""\\begingroup\\centering{size}
\\setlength{{\\tabcolsep}}{{{colsep}}}%
\\begin{{tabular}}{{l{pair_spec}cc}}
\\toprule
 & \\multicolumn{{{2 * n_stats}}}{{c}}{{{header}}} &
   \\multicolumn{{2}}{{c}}{{Mann-Whitney}} \\\\
\\cmidrule(lr){{2-{2 * n_stats + 1}}} \\cmidrule(lr){{{2 * n_stats + 2}-{2 * n_stats + 3}}}
 & {heads} & $n$ & $p$ \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}\\par\\endgroup"""


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
        header="ADHD\\,$|$\\,Control (robot session)", quartiles=True,
        size="\\footnotesize", colsep="1pt")


def table_metrics_robot_by_group_col(post, qtext, groups):
    """HRI variant of the robot-session group table."""
    return "session_metrics_robot_by_group_col", _render_group_stats_table(
        _fixed_condition_rows(groups, "Robot"), groups,
        header="ADHD\\,$|$\\,Control (robot session)", quartiles=False,
        size="\\scriptsize", colsep="3pt")


def table_metrics_control_by_group(post, qtext, groups):
    """Thesis: control-session values, ADHD vs control, MWU."""
    return "session_metrics_control_by_group", _render_group_stats_table(
        _fixed_condition_rows(groups, "Control"), groups,
        header="ADHD\\,$|$\\,Control (control session)", quartiles=True,
        size="\\footnotesize", colsep="1pt")


def table_metrics_control_by_group_col(post, qtext, groups):
    """HRI variant of the control-session group table."""
    return "session_metrics_control_by_group_col", _render_group_stats_table(
        _fixed_condition_rows(groups, "Control"), groups,
        header="ADHD\\,$|$\\,Control (control session)", quartiles=False,
        size="\\scriptsize", colsep="3pt")


def table_metrics_delta_by_group(post, qtext, groups):
    """Thesis: robot-minus-control deltas, ADHD vs control, MWU — the
    nonparametric group-by-condition interaction check."""
    return "session_metrics_delta_by_group", _render_group_stats_table(
        _delta_rows(groups), groups,
        header="ADHD\\,$|$\\,Control ($\\Delta$ robot $-$ control)",
        quartiles=True, size="\\footnotesize", colsep="0.5pt")


def table_metrics_delta_by_group_col(post, qtext, groups):
    """HRI variant of the delta table."""
    return "session_metrics_delta_by_group_col", _render_group_stats_table(
        _delta_rows(groups), groups,
        header="ADHD\\,$|$\\,Control ($\\Delta$ robot $-$ control)",
        quartiles=False, size="\\scriptsize", colsep="3pt")


def table_session_metrics_ctrl(post, qtext, groups):
    """Thesis version: control-group robot-vs-control counterpart."""
    return "session_metrics_ctrl", _render_cross_table(
        groups, GROUP_CONTROL, quartiles=True, size="\\footnotesize",
        colsep="1.5pt")


def table_session_metrics_ctrl_col(post, qtext, groups):
    """HRI version of the control-group table."""
    return "session_metrics_ctrl_col", _render_cross_table(
        groups, GROUP_CONTROL, quartiles=False, size="\\scriptsize",
        colsep="3pt")


def _render_did_table(groups, *, size, colsep) -> str:
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
            cells += [_fmt_v(sub[c].dropna().mean(), dd)
                      for c in ("Robot", "Control")]
            _, p_w, _ = _wilcoxon_cells(sub["Robot"], sub["Control"])
            cells.append(p_w)
        for s in (df["Robot"], df["Control"], delta):
            _, p_u = _mwu_cells(s[is_a], s[~is_a])
            cells.append(p_u)
        cells = [c.replace("p = ", "").replace("p < ", "< ") for c in cells]
        body_rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
    body = "\n".join(body_rows)
    return f"""\\begingroup\\centering{size}
\\setlength{{\\tabcolsep}}{{{colsep}}}%
\\begin{{tabular}}{{lrrcrrcccc}}
\\toprule
 & \\multicolumn{{3}}{{c}}{{ADHD}} & \\multicolumn{{3}}{{c}}{{No-ADHD}} &
   \\multicolumn{{3}}{{c}}{{Mann-Whitney $p$}} \\\\
\\cmidrule(lr){{2-4}} \\cmidrule(lr){{5-7}} \\cmidrule(lr){{8-10}}
 & Robot & No-rob. & $p_W$ & Robot & No-rob. & $p_W$
 & Robot & No-rob. & $\\Delta$ \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}\\par\\endgroup"""


def table_metrics_did(post, qtext, groups):
    """Thesis version of the consolidated factorial (DiD) summary."""
    return "session_metrics_did", _render_did_table(
        groups, size="\\footnotesize", colsep="4pt")


def table_metrics_did_col(post, qtext, groups):
    """HRI version of the consolidated factorial (DiD) summary."""
    return "session_metrics_did_col", _render_did_table(
        groups, size="\\scriptsize", colsep="3pt")


CHART_BUILDERS = [chart_feature_means, chart_feature_means_col,
                  table_session_metrics, table_session_metrics_col,
                  table_session_metrics_adhd, table_session_metrics_adhd_col,
                  table_session_metrics_ctrl, table_session_metrics_ctrl_col,
                  table_metrics_robot_by_group,
                  table_metrics_robot_by_group_col,
                  table_metrics_control_by_group,
                  table_metrics_control_by_group_col,
                  table_metrics_delta_by_group,
                  table_metrics_delta_by_group_col,
                  table_metrics_did, table_metrics_did_col]


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


# Per-fragment routing (decided 02.09): the 2x2 breakdown tables go to
# the thesis appendix only; the consolidated DiD summary and everything
# else ship to both repos. results_preview also stays thesis-only (it
# \inputs fragments the HRI repo does not receive).
THESIS_ONLY_FRAGMENTS = {
    "session_metrics_adhd", "session_metrics_adhd_col",
    "session_metrics_ctrl", "session_metrics_ctrl_col",
    "session_metrics_robot_by_group", "session_metrics_robot_by_group_col",
    "session_metrics_control_by_group",
    "session_metrics_control_by_group_col",
    "session_metrics_delta_by_group", "session_metrics_delta_by_group_col",
    "results_preview",
}


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
    if "HRI" in os.path.basename(repo):
        ship = [n for n in ship if n not in THESIS_ONLY_FRAGMENTS]
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
