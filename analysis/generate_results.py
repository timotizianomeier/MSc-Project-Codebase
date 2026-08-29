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

from generate_appendix import (
    FILE_PATTERNS, GROUP_ADHD, GROUP_CONTROL, GROUPS_FILE,  # noqa: F401
    assign_groups, clean, esc, load_qualtrics, newest_file, strip_stem,
    to_rank,
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
    # ADHD group only (decided 29.08): control bars dropped from this chart.
    coords_a = " ".join(f"({a:.2f},{k})" for k, a, c in data_rows)

    value_nodes = ""
    if value_labels:
        value_nodes = "\n    ".join(
            f"\\node[font={value_font}, inner sep=1pt, anchor=west,"
            f" xshift=2pt] "
            f"at (axis cs:{a:.2f},{k}) {{{a:.2f}}};"
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
  ]
    \\addplot[xbar, fill={ADHD_COLOR}, draw={ADHD_COLOR}!70!black] coordinates {{{coords_a}}};
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
        pitch="0.58cm", bar_pt=6.5, label_font="\\small",
        title_font="\\small\\bfseries", tick_anchors=True,
        value_labels=True)


def chart_feature_means_col(post, qtext, groups):
    """ACM column-width version (HRI): fits \columnwidth in a two-column
    layout. Numeric ticks only (anchors in the caption); per-bar value
    labels in \\tiny, nudged slightly past the bar centres so the pair
    clears each other at the tight pitch."""
    data = _feature_data(post, qtext, groups)
    return "feature_means_col", _render_feature_chart(
        data, axis_w="0.40\\columnwidth", label_w="0.46\\columnwidth",
        pitch="0.38cm", bar_pt=4.5, label_font="\\scriptsize",
        title_font="\\scriptsize\\bfseries", tick_anchors=False,
        value_labels=True, value_font="\\tiny", value_extra_pt=1.6,
        xmax=5.6, span_ext="0.18cm")


CHART_BUILDERS = [chart_feature_means, chart_feature_means_col]


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
              f"% Bare chart fragment — wrap in a figure environment in the "
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


def sync_to_repo(charts_dir: str, names: list[str]) -> None:
    """Copy the chart fragments into one target repo's results-charts/ and
    push (same mechanics as generate_appendix.sync_to_thesis_repo; every
    target repo must stay PRIVATE)."""
    repo = os.path.dirname(charts_dir)
    if not os.path.isdir(repo):
        print(f"--sync: {repo} not found — clone it first; skipping.")
        return
    os.makedirs(charts_dir, exist_ok=True)
    for name in names + ["results_preview"]:
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
