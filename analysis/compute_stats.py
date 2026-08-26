#!/usr/bin/env python3
"""
compute_stats.py
================
Terminal report of the standard questionnaire analyses (after Lalwani et
al. 2025 and O'Connell et al. 2024) for manual carry-over into the thesis /
paper text. Prints numbers only — nothing is written into any repo.

Run:  .venv/bin/python compute_stats.py

Tests per comparison axis:
  paired (same participants twice: NARS pre/post, TLX robot/control)
      -> Wilcoxon signed-rank (primary, n is small) + paired t (reference)
  independent groups (ADHD vs Control)
      -> Mann-Whitney U

Shares loaders/grouping with generate_appendix.py, so the participant set
and grouping can never diverge from the appendix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from generate_appendix import (
    FILE_PATTERNS, GROUP_ADHD, GROUP_CONTROL,
    assign_groups, clean, load_qualtrics, newest_file, to_rank,
)

# ============================================================================
# CONFIG — VERIFY the two mappings below against the Questionnaire Sheet
# ============================================================================

# ESQ-R subscales: official instrument (Strait et al. 2020) has 5 subscales
# over 25 items, coded 0-3 (Never or rarely / Sometimes / Often / Very
# often). Map each subscale to YOUR item numbers (PRE_ESQR_<n>) — your
# Qualtrics order may not match the official listing (cf. the ASRS mapping).
# Leave empty until verified: the report then gives overall + per-item means.
# Mapping verified 26.08.2026 against Strait et al. (2020) Table 2 (25
# retained items, five factors) by matching item text to our Qualtrics
# order. 24/25 matched verbatim; our item 20 ("so wrapped up ... forget
# other things") is assigned to Time Management by elimination (paper item
# 42, the remaining Factor-2 slot; both are the working-memory item, though
# the paper's copyright-abbreviated description reads differently).
# Note: Factor 5 had the weakest internal consistency in the source
# (alpha = .65) — worth a caveat if reported on its own.
ESQR_SUBSCALES: dict[str, list[int]] = {
    "Plan management": [6, 7, 12, 13, 14, 16, 17, 18, 22, 23, 24],
    "Time management": [10, 11, 15, 20],
    "Materials organization": [3, 8, 9],
    "Emotional regulation": [4, 5, 21],
    "Behavioral regulation": [1, 2, 19, 25],
}

# NARS reverse-scored items: the official scale's positively-worded S3 items
# must be reverse-coded (1..5 -> 6-x) before totalling. Enter YOUR item
# numbers (same numbering in PRE_NARS_<n> and POST_NARS_<n>).
# Verified 26.08.2026 by item wording: our items 3 ("relaxed talking"),
# 5 ("make friends"), 6 ("comforted being with") are the positively-worded
# S3 trio of the official scale (Nomura et al. 2006); the remaining 11 are
# negatively worded. Higher total = more negative attitude.
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


# ============================================================================
# Helpers
# ============================================================================

def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df else pd.Series(dtype=float)


def _desc(s: pd.Series) -> str:
    s = s.dropna()
    if not len(s):
        return "n=0"
    return (f"n={len(s)} mean={s.mean():.2f} sd={s.std(ddof=1):.2f} "
            f"median={s.median():.2f} range={s.min():.2f}-{s.max():.2f}")


def _paired_tests(a: pd.Series, b: pd.Series, label_a: str, label_b: str) -> None:
    both = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    if len(both) < 3:
        print(f"    paired n={len(both)} — too few for tests")
        return
    d = both["a"] - both["b"]
    try:
        w = stats.wilcoxon(both["a"], both["b"])
        print(f"    Wilcoxon signed-rank ({label_a} vs {label_b}, n={len(both)}): "
              f"W={w.statistic:.1f}, p={w.pvalue:.3f}")
    except ValueError as e:  # all-zero differences
        print(f"    Wilcoxon: not computable ({e})")
    t = stats.ttest_rel(both["a"], both["b"])
    print(f"    paired t (reference): t({len(both)-1})={t.statistic:.2f}, "
          f"p={t.pvalue:.3f}; mean diff={d.mean():+.2f}")


def _mwu(adhd: pd.Series, ctrl: pd.Series, what: str) -> None:
    a, c = adhd.dropna(), ctrl.dropna()
    if len(a) < 2 or len(c) < 2:
        print(f"    Mann-Whitney U ({what}): too few observations")
        return
    u = stats.mannwhitneyu(a, c, alternative="two-sided")
    print(f"    Mann-Whitney U ADHD vs Control ({what}): U={u.statistic:.1f}, "
          f"p={u.pvalue:.3f} (ADHD mean {a.mean():.2f} vs Control {c.mean():.2f})")


def _by_group(df: pd.DataFrame, groups: dict, series: pd.Series):
    pids_a = [p for p, g in groups.items() if g == GROUP_ADHD]
    pids_c = [p for p, g in groups.items() if g == GROUP_CONTROL]
    mask_a = df["PID"].isin(pids_a)
    mask_c = df["PID"].isin(pids_c)
    return series[mask_a], series[mask_c]


def _sus_scores(post: pd.DataFrame) -> pd.Series:
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


def _mean_items(df: pd.DataFrame, prefix: str, items: list[int],
                reverse: set[int] | None = None, scale_max: int = 5) -> pd.Series:
    cols = []
    for i in items:
        s = _num(df, f"{prefix}{i}")
        if reverse and i in reverse:
            s = (scale_max + 1) - s
        cols.append(s)
    return pd.concat(cols, axis=1).mean(axis=1)


# ============================================================================
# Report
# ============================================================================

def main() -> None:
    pre, _ = load_qualtrics(newest_file(FILE_PATTERNS["pre"]))
    pre = clean(pre, "PRE_PID", "pre")
    groups = assign_groups(pre)
    post, q_post = load_qualtrics(newest_file(FILE_PATTERNS["post"]))
    post = clean(post, "POST_PID", "post")
    ctrl, _ = load_qualtrics(newest_file(FILE_PATTERNS["control"]))
    ctrl = clean(ctrl, "POST_PID", "control")

    n_a = sum(1 for g in groups.values() if g == GROUP_ADHD)
    n_c = len(groups) - n_a
    print("=" * 72)
    print(f"Standard analyses — N={len(groups)} (ADHD {n_a} / Control {n_c})")
    print("=" * 72)

    # ------------------------------------------------------------ 1. ESQ-R
    print("\n[1] ESQ-R (coded 0-3; official 4-point scale)")
    print("    Two criteria shown: (a) Lalwani's literal 'above 2.5 on a")
    print("    5-point scale' maps linearly to > 1.125 on 0-3; (b) the")
    print("    stricter scale-midpoint criterion is > 1.5 on 0-3 (= 3.0 on")
    print("    1-5). NB 2.5 is NOT the 1-5 midpoint.")
    pids_a = [p for p, g in groups.items() if g == GROUP_ADHD]
    pre_a = pre[pre["PID"].isin(pids_a)]
    pre_c = pre[~pre["PID"].isin(pids_a)]
    if ESQR_SUBSCALES:
        for name, items in ESQR_SUBSCALES.items():
            ma = _mean_items(pre_a, "PRE_ESQR_", items, scale_max=3)
            mc = _mean_items(pre_c, "PRE_ESQR_", items, scale_max=3)
            lal = "pass" if ma.mean() > 1.125 else "FAIL"
            mid = "pass" if ma.mean() > 1.5 else "FAIL"
            print(f"    {name:24s} ADHD mean={ma.mean():.2f} (0-3) "
                  f"[1-5: {ma.mean()*4/3+1:.2f}] — Lalwani>2.5: {lal}; "
                  f"midpoint: {mid}")
            print(f"    {'':24s} Control mean={mc.mean():.2f} (0-3) "
                  f"[1-5: {mc.mean()*4/3+1:.2f}]")
            _mwu(ma, mc, name)
    else:
        print("    !! ESQR_SUBSCALES not configured — per-subscale check")
        print("    !! unavailable; fill the mapping from the Questionnaire Sheet.")
        overall = _mean_items(pre_a, "PRE_ESQR_", list(range(1, N_ESQR_ITEMS + 1)),
                              scale_max=3)
        print(f"    ADHD overall (all 25 items): {_desc(overall)} "
              f"[rescaled 1-5 mean: {overall.mean()*4/3+1:.2f}]")
        item_means = [(i, _num(pre_a, f'PRE_ESQR_{i}').mean())
                      for i in range(1, N_ESQR_ITEMS + 1)]
        lows = [f"item {i} ({m:.2f})" for i, m in item_means if m <= 1.5]
        print(f"    items at/below 0-3 midpoint: {', '.join(lows) or 'none'}")

    # ------------------------------------------------------ 2. NARS pre/post
    print("\n[2] NARS pre vs post (14 items, 1-5)")
    rev = NARS_REVERSE_ITEMS or None
    if not NARS_REVERSE_ITEMS:
        print("    !! NARS_REVERSE_ITEMS not configured — totals computed")
        print("    !! WITHOUT reverse-coding; verify before quoting.")
    items = list(range(1, N_NARS_ITEMS + 1))
    pre_nars = _mean_items(pre, "PRE_NARS_", items, rev).rename("pre")
    post_nars = _mean_items(post, "POST_NARS_", items, rev).rename("post")
    pre_by = pre.set_index("PID").assign(v=pre_nars.values)["v"]
    post_by = post.set_index("PID").assign(v=post_nars.values)["v"]
    both = pd.concat([pre_by, post_by], axis=1, keys=["pre", "post"]).dropna()
    print(f"    pre  mean item score: {_desc(both['pre'])}")
    print(f"    post mean item score: {_desc(both['post'])}")
    _paired_tests(both["pre"], both["post"], "pre", "post")
    ga = both[both.index.isin(pids_a)]
    gc = both[~both.index.isin(pids_a)]
    _mwu(ga["post"] - ga["pre"], gc["post"] - gc["pre"], "NARS change score")

    # ------------------------------------------------- 3. TLX by condition
    print("\n[3] NASA-TLX robot vs control (0-100 sliders, paired within-subject)")
    print("    note: 6 comparisons — mention multiple-testing when reporting.")
    for dim in TLX_DIMS:
        col = f"POST_TLX_{dim}_1"
        r = post.set_index("PID")[col].pipe(pd.to_numeric, errors="coerce")
        c = ctrl.set_index("PID")[col].pipe(pd.to_numeric, errors="coerce")
        both = pd.concat([r, c], axis=1, keys=["robot", "control"]).dropna()
        print(f"  {dim.capitalize():12s} robot {both['robot'].mean():5.1f} "
              f"vs control {both['control'].mean():5.1f}")
        _paired_tests(both["robot"], both["control"], "robot", "control")

    # --------------------------------------------------------------- 4. SUS
    print("\n[4] SUS (robot condition)")
    sus = _sus_scores(post)
    print(f"    overall: {_desc(sus)}")
    a, c = _by_group(post, groups, sus)
    print(f"    ADHD:    {_desc(a)}")
    print(f"    Control: {_desc(c)}")
    _mwu(a, c, "SUS")
    print("    (Lalwani et al. report a mean SUS of 76, n=15.)")

    # ----------------------------------------------- 5. use again / recommend
    print("\n[5] Would use again / recommend (agree or strongly agree, >=4)")
    for col, label in (("POST_FEAT_OVERALL_2", "use again"),
                       ("POST_FEAT_OVERALL_3", "recommend")):
        v = to_rank(post[col], "LIKERT5")
        yes = int((v >= 4).sum())
        a, c = _by_group(post, groups, v)
        print(f"    {label:10s}: {yes}/{v.notna().sum()} overall "
              f"(ADHD {(a >= 4).sum()}/{a.notna().sum()}, "
              f"Control {(c >= 4).sum()}/{c.notna().sum()})")
    print("    (Lalwani et al.: 12/15 would use again and recommend.)")

    # ------------------------------------------------- 6. feature ratings
    print("\n[6] Feature ratings (1-5), overall and by group")
    for title, prefix, n_items in FEATURE_BLOCKS:
        print(f"  {title}")
        for i in range(1, n_items + 1):
            col = f"{prefix}{i}"
            v = to_rank(post[col], "LIKERT5")
            a, c = _by_group(post, groups, v)
            print(f"    Q{i}: overall {v.mean():.2f}  "
                  f"ADHD {a.mean():.2f}  Control {c.mean():.2f}")
            _mwu(a, c, f"{prefix}{i}")

    print("\nDone. Carry values over manually; nothing was written anywhere.")


if __name__ == "__main__":
    main()
