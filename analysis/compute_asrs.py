#!/usr/bin/env python3
"""
compute_asrs.py
===============
Step 1 of the two-step analysis flow:

    python compute_asrs.py           # score ASRS, write participant_groups.txt
    python generate_appendix.py      # build the LaTeX appendix from the groups

Reads the latest pre-session Qualtrics export, computes ASRS scores per
participant, prints a scoring table, and writes participant_groups.txt into
DATA_DIR (next to the raw CSVs, so it is never committed).

The group rule is: ADHD if ASRS_METRIC > ASRS_THRESHOLD (strictly above),
configured at the top of generate_appendix.py. With the defaults
(screener_positives > 3, i.e. at least 4 of 6 official Part A items in the
shaded range) this equals the standard Kessler et al. cutoff, which is also
what O'Connell et al. (2024) used for inclusion. Lalwani et al. (2025)
used "above 3" as a self-chosen threshold (per interview with H. Lalwani).

The written file is plain text and safe to hand-edit — generate_appendix.py
simply reads it, so you can override the automatic assignment or paste IDs
yourself. Rerunning this script OVERWRITES the file (a note is printed).

Add --dry-run to only print the table without writing the file.
"""

import os
import sys
from datetime import datetime

# Shared config + helpers live in generate_appendix.py (same folder).
from generate_appendix import (
    ASRS_METRIC, ASRS_PART_A, ASRS_THRESHOLD, DATA_DIR, FILE_PATTERNS,
    GROUPS_FILE, INCLUDE_PIDS, asrs_metrics, clean, load_qualtrics,
    newest_file,
)


def main():
    dry_run = "--dry-run" in sys.argv
    path = newest_file(FILE_PATTERNS["pre"])
    print(f"Scoring ASRS from: {os.path.basename(path)}")
    if INCLUDE_PIDS:
        print(f"INCLUDE_PIDS: {sorted(INCLUDE_PIDS, key=lambda x: (len(x), x))}")
    pre, _ = load_qualtrics(path)
    pre = clean(pre, "PRE_PID", "pre")

    part_a_cols = [f"A{i+1}(Q{item})" for i, (item, _) in
                   enumerate(v for v in ASRS_PART_A.values())]
    header = (f"{'PID':>4} | {'DX':>3} | " +
              " ".join(f"{c:>7}" for c in part_a_cols) +
              f" | {'pos':>3} {'mean1-5':>7} {'sum0-4':>6} {'n':>2} | group")
    print("\n" + header)
    print("-" * len(header))

    rows = []
    for _, row in pre.iterrows():
        m = asrs_metrics(row)
        dx = str(row.get("PRE_ADHD_DX", "")).strip()
        dx = {"1": "Yes", "2": "No"}.get(dx, dx)  # numeric export codes
        metric = m[ASRS_METRIC]
        group = "ADHD" if metric > ASRS_THRESHOLD else "CONTROL"
        cells = " ".join(
            f"{code:>5.0f}{'*' if pos else ' '} " if code == code else "    - "
            for (_, code, pos) in m["part_a"].values())
        print(f"{row['PID']:>4} | {dx:>3} | {cells}| "
              f"{m['screener_positives']:>3} {m['mean_score_1to5']:>7.2f} "
              f"{m['sum_score']:>6.0f} {m['n_items_answered']:>2} | {group}")
        rows.append((row["PID"], dx, m, group))

    print(f"\nRule: {ASRS_METRIC} > {ASRS_THRESHOLD}  "
          f"(* = Part A item in shaded/positive range)")

    # Flag disagreements between ASRS grouping and self-reported diagnosis —
    # worth a sentence in the thesis either way.
    for pid, dx, m, group in rows:
        if dx == "Yes" and group == "CONTROL":
            print(f"  NOTE: P{pid} reports a formal diagnosis but screens "
                  f"CONTROL under the current rule.")
        if dx == "No" and group == "ADHD":
            print(f"  NOTE: P{pid} reports no formal diagnosis but screens "
                  f"ADHD under the current rule.")

    adhd = [pid for pid, _, _, g in rows if g == "ADHD"]
    ctrl = [pid for pid, _, _, g in rows if g == "CONTROL"]
    print(f"\nADHD ({len(adhd)}): {', '.join(adhd) or '-'}")
    print(f"CONTROL ({len(ctrl)}): {', '.join(ctrl) or '-'}")

    if dry_run:
        print("\n--dry-run: participant_groups.txt NOT written.")
        return

    if os.path.exists(GROUPS_FILE):
        print(f"\nOverwriting existing {GROUPS_FILE}")
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(GROUPS_FILE, "w") as f:
        f.write(f"# Written by compute_asrs.py on {stamp}\n"
                f"# Source: {os.path.basename(path)}\n"
                f"# Rule: {ASRS_METRIC} > {ASRS_THRESHOLD}\n"
                f"# Safe to hand-edit; rerunning compute_asrs.py overwrites.\n"
                f"ADHD: {', '.join(adhd)}\n"
                f"CONTROL: {', '.join(ctrl)}\n")
    print(f"Wrote {GROUPS_FILE}")
    print("Next step: python generate_appendix.py")


if __name__ == "__main__":
    main()