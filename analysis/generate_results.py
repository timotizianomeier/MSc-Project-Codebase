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
    ("Body doubling / presence", "POST_FEAT_PRESENCE_", 3),
    ("Inattention detection", "POST_FEAT_INATT_", 5),
    ("Context-aware / task-aware support", "POST_FEAT_CONTEXT_", 4),
    ("Overall experience", "POST_FEAT_OVERALL_", 3),
]

LIKERT_TICKS = [
    (1, "Strongly disagree"), (2, "Disagree"), (3, "Neutral"),
    (4, "Agree"), (5, "Strongly agree"),
]


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
    # with question rows. Headers are pseudo symbolic coords: they appear in
    # the coordinate list and the explicit ytick list, but never in the data,
    # so pgfplots renders the label on an otherwise empty row.
    all_labels: list[str] = []   # symbolic coords in display order
    data_rows: list[tuple[str, float, float]] = []  # (label, adhd, ctrl)
    for title, prefix, n_items in FEATURE_BLOCKS:
        all_labels.append(f"\\textbf{{{esc(title)}}}")
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
            label = esc(strip_stem(qtext.get(col, col)))
            all_labels.append(label)
            data_rows.append((label, means[GROUP_ADHD], means[GROUP_CONTROL]))

    rows = data_rows  # for the height calculation below
    sym = ",".join("{" + l + "}" for l in all_labels)
    coords_a = " ".join(f"({a:.2f},{{{l}}})" for l, a, c in data_rows)
    coords_c = " ".join(f"({c:.2f},{{{l}}})" for l, a, c in data_rows)

    xticklabels = ",".join(
        f"{{{v}\\\\{{\\scriptsize ({lbl})}}}}" for v, lbl in LIKERT_TICKS)

    # ~1.05cm per row (questions AND header rows) keeps bars readable
    height_cm = 1.05 * len(all_labels) + 1.5
    return "feature_means", f"""\\begin{{tikzpicture}}[trim axis left, trim axis right]
  \\begin{{axis}}[
    xbar, bar width=5.5pt, y=1.05cm,
    scale only axis, width=0.48\\textwidth, height={height_cm:.1f}cm,
    symbolic y coords={{{sym}}}, ytick={{{sym}}}, y dir=reverse,
    yticklabel style={{font=\\small, align=right, text width=0.42\\textwidth}},
    xmin=0, xmax=5.6,
    xtick={{1,2,3,4,5}},
    xticklabels={{{xticklabels}}},
    x tick label style={{font=\\small, align=center}},
    xlabel={{Mean score}},
    axis x line*=bottom, axis y line*=left,
    legend style={{font=\\small, at={{(0.5,1.01)}}, anchor=south,
                   draw=none, fill=none}},
    legend columns=2,
    legend image code/.code={{\\draw[#1] (0cm,-0.06cm) rectangle (0.18cm,0.12cm);}},
  ]
    \\addplot[xbar, fill={ADHD_COLOR}, draw={ADHD_COLOR}!70!black,
        nodes near coords, point meta=rawx,
        nodes near coords={{\\pgfmathprintnumber[fixed, precision=2, fixed zerofill]{{\\pgfplotspointmeta}}}},
        every node near coord/.append style={{font=\\scriptsize, xshift=1pt}}] coordinates {{{coords_a}}};
    \\addplot[xbar, fill={CTRL_COLOR}, draw={CTRL_COLOR}!70!black,
        nodes near coords, point meta=rawx,
        nodes near coords={{\\pgfmathprintnumber[fixed, precision=2, fixed zerofill]{{\\pgfplotspointmeta}}}},
        every node near coord/.append style={{font=\\scriptsize, xshift=1pt}}] coordinates {{{coords_c}}};
    \\legend{{ADHD, Control}}
  \\end{{axis}}
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
