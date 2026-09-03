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

import csv
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

# Scoring config, questionnaire loaders, and all session-log analysis live
# in generate_appendix.py (single source of truth for numbers AND the
# generated tables/charts); this script only adds terminal test reporting.
from generate_appendix import (
    ESQR_SUBSCALES, FEATURE_BLOCKS, FILE_PATTERNS, FRUSTRATION_PREDICTORS,
    GROUP_ADHD, GROUP_CONTROL, N_ESQR_ITEMS, N_NARS_ITEMS,
    NARS_REVERSE_ITEMS, QUIET_BUFFER_S, SIGNAL_COVERAGE_MIN, TLX_DIMS,
    assign_groups, clean, load_qualtrics, newest_file, to_rank,
    episode_records, frustration_mechanism, gated_signals, landmark_pairs,
    quiet_engagement_medians,
    _read_rows as _log_rows,
    session_dirs as _session_dirs,
    session_metrics as _session_metrics,
    signal_polls as _polls,
    extract_episodes as _episodes,
    intervention_utterance_durations as _intervention_utterance_durations,
    replay_counterfactuals as _replay_counterfactuals,
    num_col as _num, mean_items as _mean_items, sus_scores as _sus_scores,
)

# ============================================================================
# Helpers
# ============================================================================


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
    t = stats.ttest_ind(a, c, equal_var=False)  # Welch t, sanity reference
    print(f"    Welch t (reference): t={t.statistic:.2f}, p={t.pvalue:.3f}")


def _by_group(df: pd.DataFrame, groups: dict, series: pd.Series):
    pids_a = [p for p, g in groups.items() if g == GROUP_ADHD]
    pids_c = [p for p, g in groups.items() if g == GROUP_CONTROL]
    mask_a = df["PID"].isin(pids_a)
    mask_c = df["PID"].isin(pids_c)
    return series[mask_a], series[mask_c]




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
            mo = _mean_items(pre, "PRE_ESQR_", items, scale_max=3)
            print(f"    {'':24s} Overall mean={mo.mean():.2f} (0-3) "
                  f"[1-5: {mo.mean()*4/3+1:.2f}]")
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
    ga = both[both.index.isin(pids_a)]
    gc = both[~both.index.isin(pids_a)]
    for gname, sub in (("Overall", both), ("ADHD", ga), ("Control", gc)):
        print(f"  {gname}")
        print(f"    pre  mean item score: {_desc(sub['pre'])}")
        print(f"    post mean item score: {_desc(sub['post'])}")
        _paired_tests(sub["pre"], sub["post"], "pre", "post")
    _mwu(ga["post"] - ga["pre"], gc["post"] - gc["pre"], "NARS change score")

    # ------------------------------------------------- 3. TLX by condition
    print("\n[3] NASA-TLX robot vs control (0-100 sliders, paired within-subject)")
    print("    note: 6 comparisons — mention multiple-testing when reporting.")
    for dim in TLX_DIMS:
        col = f"POST_TLX_{dim}_1"
        r = post.set_index("PID")[col].pipe(pd.to_numeric, errors="coerce")
        c = ctrl.set_index("PID")[col].pipe(pd.to_numeric, errors="coerce")
        both = pd.concat([r, c], axis=1, keys=["robot", "control"]).dropna()
        ba = both[both.index.isin(pids_a)]
        bc = both[~both.index.isin(pids_a)]
        print(f"  {dim.capitalize()}")
        for gname, sub in (("Overall", both), ("ADHD", ba), ("Control", bc)):
            print(f"    {gname:8s} robot {sub['robot'].mean():5.1f} "
                  f"vs control {sub['control'].mean():5.1f}")
            _paired_tests(sub["robot"], sub["control"], "robot", "control")
        _mwu(ba["robot"] - ba["control"], bc["robot"] - bc["control"],
             f"{dim} robot-minus-control delta")

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

    # ------------------------------------- 7. frustration mechanism (exploratory)
    print("\n[7] TLX Frustration mechanism — exploratory correlations")
    print("    Robot-minus-control frustration delta vs per-participant robot-")
    print("    session behaviour from the parsed logs. Seven correlations —")
    print("    treat every p as exploratory/uncorrected (a 'significant' hit")
    print("    among this many tests would itself need correction).")
    mech = frustration_mechanism(post, ctrl)
    mech["group"] = mech["pid"].map(groups)
    print(mech.round(2).sort_values("fr_delta", ascending=False)
          .to_string(index=False))
    for x, label in FRUSTRATION_PREDICTORS:
        for gname, msub in (("Overall", mech),
                            ("ADHD", mech[mech["group"] == GROUP_ADHD]),
                            ("Control", mech[mech["group"] == GROUP_CONTROL])):
            sub = msub.dropna(subset=[x])
            if len(sub) < 4:
                print(f"    Spearman {label} vs frustration delta [{gname}]: "
                      f"n={len(sub)} — too few")
                continue
            r = stats.spearmanr(sub[x], sub["fr_delta"])
            print(f"    Spearman {label} vs frustration delta [{gname}] "
                  f"(n={len(sub)}): rho={r.statistic:+.3f}, p={r.pvalue:.3f}")

    # ------------------------------------- 8. session-log descriptives
    print("\n[8] Session-log descriptives per condition and group")
    print("    Counterfactual caveat: control counts are an upper bound —")
    print("    no conversation exists there to reset the interaction")
    print("    cooldown (only counterfactual fires do), so gating is")
    print("    strictly looser than in the robot condition. See [9e].")
    sdirs = _session_dirs()
    gated = gated_signals(sdirs)
    if gated:
        print(f"    Signal-coverage gate (<{SIGNAL_COVERAGE_MIN:.0%} of the "
              "session covered by value-bearing polls) — excluded from all "
              "signal-level analyses in [9]:")
        for (pid, cond, sig), cov in sorted(gated.items(),
                                            key=lambda kv: (int(kv[0][0]),
                                                            kv[0][1])):
            print(f"      P{pid} {cond} {sig}: {100 * cov:.1f}% coverage")
    met = {k: _session_metrics(d, k[1]) for k, d in sdirs.items() if k[0] in groups}
    metric_rows = [
        ("robot_turns", "Robot responses", ("Robot",)),
        ("user_turns", "User turns (spoken)", ("Robot",)),
        ("robot_min", "Robot talk-time (min)", ("Robot",)),
        ("user_min", "User talk-time (min)", ("Robot",)),
        ("context", "Context submissions", ("Robot",)),
        ("int_eng", "Engagement interventions", ("Robot", "Control")),
        ("int_emo", "Emotion interventions", ("Robot", "Control")),
        ("int_tot", "All interventions", ("Robot", "Control")),
    ]
    for cond in ("Robot", "Control"):
        pids = sorted((p for p, c in met if c == cond), key=int)
        tag = "  [counterfactuals = upper bound]" if cond == "Control" else ""
        print(f"\n  {cond} sessions (n={len(pids)}){tag}")
        for key, label, conds in metric_rows:
            if cond not in conds:
                continue
            s = pd.Series({p: met[(p, cond)][key] for p in pids}, dtype=float)
            a = s[[p for p in pids if groups[p] == GROUP_ADHD]]
            c = s[[p for p in pids if groups[p] == GROUP_CONTROL]]
            print(f"    {label:<25} median {s.median():6.1f} "
                  f"[{s.quantile(.25):.1f}; {s.quantile(.75):.1f}]  "
                  f"SD {s.std(ddof=1):5.1f}  range {s.min():.0f}-{s.max():.0f}")
            print(f"    {'':<25} ADHD md {a.median():.1f} "
                  f"[{a.quantile(.25):.1f}; {a.quantile(.75):.1f}] | "
                  f"Ctrl md {c.median():.1f} "
                  f"[{c.quantile(.25):.1f}; {c.quantile(.75):.1f}]")
            _mwu(a, c, label)

    # ------------------------------------- 9. episodes & recovery
    print("\n[9] Below-threshold episodes and recovery")
    print("    Episode = >=2 consecutive polls with the monitor signal past")
    print("    its threshold (engagement rolling avg < 0.80 / windowed")
    print("    negative share > 0.60). Recovery = episode start to the")
    print("    first back-at-threshold poll. Sensing gaps >30 s or session")
    print("    end censor an episode (recovery unobserved).")
    ep, spans = episode_records(groups, sdirs)
    ep["group"] = ep.pid.map(groups)

    print("\n[9a] Episode inventory")
    for sig, signame in (("eng", "engagement"), ("emo", "emotion")):
        for cond in ("Robot", "Control"):
            sub = ep[(ep.sig == sig) & (ep.cond == cond)]
            for gname, g in (("Overall", sub),
                             ("ADHD", sub[sub.group == GROUP_ADHD]),
                             ("Control", sub[sub.group == GROUP_CONTROL])):
                obs = g.dur.dropna()
                line = (f"  {signame:<10} {cond:<7} [{gname:<7}] "
                        f"episodes={len(g)} "
                        f"(censored {int(g.censored.sum())}, "
                        f"cued {int(g.cued.sum())})")
                if len(obs):
                    line += (f"  recovery median {obs.median():.0f}s "
                             f"[{obs.quantile(.25):.0f}; "
                             f"{obs.quantile(.75):.0f}]")
                print(line)
            # Gated series are excluded, not counted as 0 episodes.
            spids = sorted((p for p, c in sdirs if c == cond and p in groups
                            and (p, c, sig) not in gated), key=int)
            cnt = (sub.groupby("pid").size()
                   .reindex(spids, fill_value=0).astype(float))
            _mwu(cnt[[p for p in spids if groups[p] == GROUP_ADHD]],
                 cnt[[p for p in spids if groups[p] == GROUP_CONTROL]],
                 f"{signame} {cond} episodes per participant")

    print("\n[9b] Within-robot: cue delivered vs gate-suppressed episodes")
    print("     WARNING immortal-time bias: an episode only receives a cue")
    print("     if it survives until the gates open, so fast-recovering")
    print("     episodes land in 'suppressed' by construction — the naive")
    print("     medians below are NOT evidence and are printed only for")
    print("     completeness. The landmark test is the valid comparison:")
    print("     remaining time after the cue vs remaining time of the")
    print("     suppressed episodes still ongoing at the same elapsed")
    print("     time (risk-set matched medians, Wilcoxon signed-rank over")
    print("     delivered episodes). Residual caveats: gating follows")
    print("     conversation timing (not randomised) and episodes pool")
    print("     across participants (clustering unmodelled).")
    for sig, signame in (("eng", "engagement"), ("emo", "emotion")):
        sub_all = ep[(ep.sig == sig) & (ep.cond == "Robot")]
        for gname, sub in (("Overall", sub_all),
                           ("ADHD", sub_all[sub_all.group == GROUP_ADHD]),
                           ("Control", sub_all[sub_all.group == GROUP_CONTROL])):
            cued = sub[sub.cued].dur.dropna()
            sup = sub[~sub.cued].dur.dropna()
            if not (len(cued) and len(sup)):
                print(f"  {signame} [{gname}]: too few episodes")
                continue
            print(f"  {signame} [{gname}]: delivered n={len(cued)} "
                  f"median {cued.median():.0f}s"
                  f" | suppressed n={len(sup)} median {sup.median():.0f}s"
                  "   [naive, length-biased]")
            rce = sub.rec_cue_end.dropna()
            if len(rce):
                print(f"    recovery from cue-utterance END (delivered only, "
                      f"n={len(rce)}): median {rce.median():.0f}s "
                      f"[{rce.quantile(.25):.0f}; {rce.quantile(.75):.0f}]")
            for lm, lm_label in (("start", "cue START"), ("end", "cue END  ")):
                dvals, mvals = landmark_pairs(sub, lm)
                if len(dvals) >= 5:
                    try:
                        w = stats.wilcoxon(dvals, mvals)
                        wtxt = f"W={w.statistic:.1f}, p={w.pvalue:.3f}"
                    except ValueError as exc:
                        wtxt = f"not computable ({exc})"
                    print(f"    landmark at {lm_label} (n={len(dvals)}): "
                          f"remaining median {dvals.median():.0f}s vs matched "
                          f"suppressed {mvals.median():.0f}s; Wilcoxon {wtxt}")
                else:
                    print(f"    landmark at {lm_label}: only {len(dvals)} "
                          "matchable delivered episodes — too few to test")
    print("    NOTE if the two landmarks disagree in direction, the")
    print("    within-robot contrast is not robust — report the")
    print("    after-cue-end recovery descriptively instead. Group-level")
    print("    landmark matching draws the risk set from that group only.")

    print("\n[9c] Cross-condition (paired per participant; camera caveat")
    print("     applies unless [9d] clears it)")
    for sig, signame in (("eng", "engagement"), ("emo", "emotion")):
        sub = ep[ep.sig == sig]
        med = sub.dropna(subset=["dur"]).groupby(["pid", "cond"]).dur.median().unstack()
        below = sub.assign(bt=sub.dur.fillna(sub.low_bound)).groupby(
            ["pid", "cond"]).bt.sum().unstack().reindex(
            sorted({p for p, c, s in spans if s == sig}, key=int)).fillna(0.0)
        pct = pd.DataFrame({
            cond: pd.Series({p: 100 * below.loc[p, cond] / spans[(p, cond, sig)]
                             for p in below.index if (p, cond, sig) in spans})
            for cond in ("Robot", "Control")})
        for gname in ("Overall", "ADHD", "Control"):
            if gname == "Overall":
                gmed, gpct = med, pct
            else:
                gpids = [p for p, g in groups.items() if g == gname]
                gmed = med[med.index.isin(gpids)]
                gpct = pct[pct.index.isin(gpids)]
            print(f"  {signame} [{gname}] — median recovery (s), participants "
                  f"with episodes in both conditions:")
            if {"Robot", "Control"} <= set(gmed.columns):
                _paired_tests(gmed["Robot"], gmed["Control"],
                              "Robot", "Control")
            print(f"  {signame} [{gname}] — % of observed time past threshold "
                  f"({len(gpct)} participants):")
            print(f"    Robot md {gpct['Robot'].median():.1f}% / "
                  f"Control md {gpct['Control'].median():.1f}%")
            _paired_tests(gpct["Robot"], gpct["Control"], "Robot", "Control")

    print("\n[9d] Camera check — raw engagement medians outside interaction")
    print(f"     windows (samples within {QUIET_BUFFER_S:.0f}s after any speech")
    print("     excluded in the robot session; control has no speech). If the")
    print("     robot-vs-control offset persists here, it is optics/geometry,")
    print("     not head motion during speech.")
    q = quiet_engagement_medians(groups, sdirs)
    for gname in ("Overall", GROUP_ADHD, GROUP_CONTROL):
        gq = q if gname == "Overall" else q[q.index.map(groups.get) == gname]
        print(f"  [{gname}]")
        print(f"    Robot all-samples median of medians "
              f"{gq['robot_all'].median():.3f} | "
              f"Robot quiet-only {gq['robot_quiet'].median():.3f} "
              f"(median {gq['n_quiet'].median():.0f} quiet samples/session) | "
              f"Control {gq['control'].median():.3f}")
        print("    Robot QUIET vs Control (the fair comparison):")
        _paired_tests(gq["robot_quiet"], gq["control"],
                      "Robot-quiet", "Control")
        print("    Robot ALL vs Control (reference):")
        _paired_tests(gq["robot_all"], gq["control"], "Robot-all", "Control")

    print("\n[9e] Replay sensitivity — control counterfactuals with each")
    print("     fire carrying an intervention utterance's gating footprint")
    print("     (speaking gate closed during it, interaction cooldown from")
    print("     its end). Still an upper bound: ordinary conversation —")
    print("     the dominant suppressor in robot sessions — has no control")
    print("     counterpart and is not simulated.")
    idurs = pd.Series(_intervention_utterance_durations(sdirs))
    print(f"    intervention utterance length (robot sessions, n={len(idurs)}): "
          f"mean {idurs.mean():.1f}s, median {idurs.median():.1f}s")
    dur = float(idurs.mean())
    tot = {g: [0, 0, 0] for g in ("Overall", GROUP_ADHD, GROUP_CONTROL)}
    print("    pid  group    logged  replay(0s)  replay(utterance)")
    for pid, cond in sorted(met, key=lambda k: int(k[0])):
        if cond != "Control":
            continue
        d = sdirs[(pid, cond)]
        logged = met[(pid, cond)]["int_tot"]
        r0, rd = _replay_counterfactuals(d, 0.0), _replay_counterfactuals(d, dur)
        for g in ("Overall", groups[pid]):
            tot[g][0] += logged; tot[g][1] += r0; tot[g][2] += rd
        print(f"    P{pid:<3} {groups[pid]:<8} {logged:5d} {r0:9d} {rd:14d}")
    for g in ("Overall", GROUP_ADHD, GROUP_CONTROL):
        print(f"    TOTAL {g:<8} {tot[g][0]:4d} {tot[g][1]:9d} {tot[g][2]:14d}")
    print("    (replay(0s) validates the replay against the deployed logic)")

    print("\n[10] Post-intervention re-engagement (Nicole 03.09)")
    print("     Time to re-engage after a cue = rec_cue_end from [9b];")
    print("     engaged gap = recovery to next same-signal episode start")
    print("     (observed gaps only, censored counts noted; cued/uncued")
    print("     split is naive/immortal-time-biased, descriptive only).")
    from generate_results import _reengagement_gaps
    gaps = _reengagement_gaps(groups)
    for sig, signame in (("eng", "engagement"), ("emo", "emotion")):
        rce = ep[(ep.sig == sig) & (ep.cond == "Robot")].rec_cue_end.dropna()
        if len(rce):
            print(f"  {signame}: re-engage after cue (robot) n={len(rce)} "
                  f"median {rce.median():.0f}s "
                  f"[{rce.quantile(.25):.0f}; {rce.quantile(.75):.0f}]")
        sub = gaps[(gaps.sig == sig) & ~gaps.censored]
        rob = sub[sub.cond == "Robot"]
        print(f"  {signame}: gap to next episode — robot cued "
              f"md {rob[rob.cued].gap.median():.0f}s (n={rob.cued.sum()}) | "
              f"uncued md {rob[~rob.cued].gap.median():.0f}s "
              f"(n={(~rob.cued).sum()})   [naive]")
        med = sub.groupby(["pid", "cond"]).gap.median().unstack().reindex(
            columns=["Robot", "Control"])
        _paired_tests(med["Robot"], med["Control"], "Robot", "Control")
    print(f"  censored gaps excluded: {int(gaps.censored.sum())}")

    print("\nDone. Carry values over manually; nothing was written anywhere.")


if __name__ == "__main__":
    main()
