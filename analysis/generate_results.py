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
THESIS_CHARTS_DIR = os.path.expanduser(
    "~/Projects/MSc-Project-Final-Report/results-charts")

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
    "POST_FEAT_INATT_3": "Prompts helped me refocus",
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

def chart_feature_means(post, qtext, groups) -> tuple[str, str]:
    """Full-width horizontal grouped bar chart: per feature question, the
    mean Likert score (coded 1-5) for the ADHD and Control groups side by
    side. Questions on the y axis, scale on the x axis with the response
    labels bracketed under the numeric ticks (after Lalwani et al. Fig. 3,
    restyled to match the appendix charts)."""
    # Row list mixes bold section-header rows (no bars, no value labels)
    # with question rows. PERFORMANCE: symbolic coordinates are short keys
    # (r1, r2, ...) — the visible text arrives via the yticklabels list
    # instead. pgfplots string-compares coordinate names constantly (per
    # coordinate, per tick, per node), and 100+-char sentence-length names
    # made a single chart dominate the whole thesis compile.
    # No header rows: each block instead gets a rotated title + vertical
    # rule spanning its question rows, drawn left of the label column
    # (coordinates captured inside the axis, drawn after it).
    all_keys: list[str] = []      # short symbolic coords in display order
    all_texts: list[str] = []     # what each row displays
    data_rows: list[tuple[str, float, float]] = []  # (key, adhd, ctrl)
    block_spans: list[tuple[str, str, str]] = []    # (title, first, last key)
    for title, prefix, n_items in FEATURE_BLOCKS:
        first_key = None
        last_key = None
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

    # Named coordinates at each block's first/last row (x=0 = axis left
    # edge); the bracket is drawn after \end{axis}, offset past the
    # 0.42\textwidth label column, extended half a row beyond both ends.
    span_coords = "\n    ".join(
        f"\\coordinate (b{i}t) at (axis cs:0,{first});"
        f" \\coordinate (b{i}b) at (axis cs:0,{last});"
        for i, (_, first, last) in enumerate(block_spans))
    # Titles break at spaces onto stacked lines so they stay within their
    # bracket's span when rotated.
    span_draws = "\n  ".join(
        f"\\draw ([xshift=-\\dimexpr0.42\\textwidth+8pt\\relax, yshift=0.3cm]b{i}t)"
        f" -- ([xshift=-\\dimexpr0.42\\textwidth+8pt\\relax, yshift=-0.3cm]b{i}b)"
        f" node[midway, rotate=90, anchor=south, align=center,"
        f" font=\\small\\bfseries]"
        f" {{{esc(title).replace(' ', chr(92) * 2)}}};"
        for i, (title, _, _) in enumerate(block_spans))

    sym = ",".join(all_keys)
    yticklabels = ",".join("{" + t + "}" for t in all_texts)
    coords_a = " ".join(f"({a:.2f},{k})" for k, a, c in data_rows)
    coords_c = " ".join(f"({c:.2f},{k})" for k, a, c in data_rows)

    # nodes near coords ignores the bar shift on a reversed symbolic xbar
    # axis, so place value labels manually at each bar's own vertical offset
    # (bar width 5.5pt -> the pair sits at +-2.75pt; ADHD is the lower bar).
    # Plot order: Control first = lower bar, so ADHD sits ON TOP of each
    # pair; `reverse legend` keeps the legend reading ADHD, Control.
    value_nodes = "\n    ".join(
        f"\\node[font=\\scriptsize, anchor=west, xshift=2pt, yshift=2.75pt] "
        f"at (axis cs:{a:.2f},{k}) {{{a:.2f}}}; "
        f"\\node[font=\\scriptsize, anchor=west, xshift=2pt, yshift=-2.75pt] "
        f"at (axis cs:{c:.2f},{k}) {{{c:.2f}}};"
        for k, a, c in data_rows)

    xticklabels = ",".join(
        f"{{{v}\\\\{{\\scriptsize ({lbl})}}}}" if lbl else f"{{{v}}}"
        for v, lbl in LIKERT_TICKS)

    # NO trim axis here (unlike the appendix charts): trim excludes the y
    # labels from the bounding box, so \\centering centers only the bars and
    # the figure appears shoved left. A standalone figure wants the full box.
    return "feature_means", f"""\\begin{{tikzpicture}}
  \\begin{{axis}}[
    xbar, bar width=5.5pt, y=0.75cm,
    scale only axis, width=0.48\\textwidth,
    symbolic y coords={{{sym}}}, ytick={{{sym}}}, y dir=reverse,
    yticklabels={{{yticklabels}}},
    yticklabel style={{font=\\small, align=right, text width=0.42\\textwidth}},
    xmin=0, xmax=5.6,
    xtick={{1,2,3,4,5}},
    xticklabels={{{xticklabels}}},
    x tick label style={{font=\\small, align=center}},
    axis x line*=bottom, axis y line*=left,
    legend style={{font=\\small, at={{(0.5,1.01)}}, anchor=south,
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


CHART_BUILDERS = [chart_feature_means]


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
        sync_to_thesis_repo(names)


def sync_to_thesis_repo(names: list[str]) -> None:
    """Copy the chart fragments into the thesis repo and push (see
    generate_appendix.sync_to_thesis_repo — same mechanics, same privacy
    rule: the thesis repo must stay PRIVATE)."""
    repo = os.path.dirname(THESIS_CHARTS_DIR)
    if not os.path.isdir(repo):
        print(f"--sync: {repo} not found — clone the thesis repo first; skipping.")
        return
    os.makedirs(THESIS_CHARTS_DIR, exist_ok=True)
    for name in names + ["results_preview"]:
        shutil.copy2(os.path.join(OUTPUT_DIR, f"{name}.tex"),
                     os.path.join(THESIS_CHARTS_DIR, f"{name}.tex"))
    changed = subprocess.run(
        ["git", "status", "--porcelain", "--", os.path.basename(THESIS_CHARTS_DIR)],
        cwd=repo, capture_output=True, text=True).stdout.strip()
    if not changed:
        print("--sync: charts unchanged — nothing to push.")
        return
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    subprocess.run(["git", "add", os.path.basename(THESIS_CHARTS_DIR)],
                   cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-m",
         f"chore(results): regenerate results charts ({stamp})\n\n"
         "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"],
        cwd=repo, check=True)
    subprocess.run(["git", "push"], cwd=repo, check=True)
    print("--sync: pushed to thesis repo — in Overleaf: Menu -> GitHub -> "
          "Pull GitHub changes.")


if __name__ == "__main__":
    main()
