"""Bayesian follow-up (Nicole 10.09) — runs the four experiments proposed in
the small-n note (paper-facing exploration; the thesis stays frequentist):

  A. ROPE / equivalence: refit the Gaussian hierarchical models of
     bayes_comparison.py and report the posterior mass inside a region of
     practical equivalence per metric family (score/share +-0.05, %-time
     +-5 pp, durations +-10 s). Decision per Kruschke: P(in ROPE) >= .95
     "practically equivalent", <= .05 "practically different", else undecided.
  B. Prior sensitivity: (i) JZS Bayes factors under Cauchy scales r = 0.354
     ... 1.414; (ii) the two headline metrics refit under a sceptical
     literature prior (Lalwani 2025 / O'Connell 2024 / Zuckerman 2016 report
     NO measurable effect on any outcome we measure — so the literature can
     only justify a prior centred on zero), the bambi default, and an
     optimistic prior (medium positive effect).
  C. Poll-level hierarchical models: the 5 s engagement score and negative-
     emotion share, speech-excluded and coverage-gated, with participant
     random intercept + random robot slope, fitted naively and with a lag-1
     autoregressive term (steady-state effect = robot / (1 - lag)).
  D. Bayesian paired analysis of NASA-TLX frustration (the study's one
     significant survey result): BF10, hierarchical posterior, ROPE +-5.

Run: cd analysis && .venv/bin/python bayes_followup.py
Writes output/bayes_followup.tex (compile with pdflatex).
"""
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats as sps

import generate_results as gr
import generate_appendix as ga

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)
COUNT_METRICS = {"Engagement interv.", "Emotion interv.", "All interv."}
SEED = 42
T0 = time.time()

pre, _ = gr.load_qualtrics(gr.newest_file(gr.FILE_PATTERNS["pre"]))
pre = gr.clean(pre, "PRE_PID", "pre")
groups = gr.assign_groups(pre)


def elapsed() -> str:
    return f"[{(time.time() - T0) / 60:5.1f} min]"


def rope_halfwidth(label: str) -> float:
    if "score" in label or "share" in label:
        return 0.05
    if "duration" in label:
        return 10.0
    return 5.0  # %-time metrics


def summarize(draws: np.ndarray, rope: float | None = None) -> dict:
    lo, hi = np.percentile(draws, [2.5, 97.5])
    out = {"mean": float(draws.mean()), "lo": float(lo), "hi": float(hi),
           "pdir": float(max((draws > 0).mean(), (draws < 0).mean()))}
    if rope is not None:
        out["p_rope"] = float((np.abs(draws) < rope).mean())
    return out


def fmt_ci(s: dict, d: int = 2) -> str:
    return f"{s['mean']:+.{d}f} [{s['lo']:+.{d}f}, {s['hi']:+.{d}f}]"


def rope_verdict(p: float) -> str:
    return "equivalent" if p >= 0.95 else ("different" if p <= 0.05 else "undecided")


def long_frame(df: pd.DataFrame) -> pd.DataFrame:
    long = df.stack().rename("value").reset_index()
    long.columns = ["pid", "cond", "value"]
    long = long.dropna(subset=["value"]).reset_index(drop=True)
    long["robot"] = (long["cond"] == "Robot").astype(float) - 0.5
    long["adhd"] = (long["pid"].map(groups.get) == gr.GROUP_ADHD).astype(float) - 0.5
    return long


import arviz as az  # noqa: E402
import bambi as bmb  # noqa: E402


def fit(formula: str, data: pd.DataFrame, priors=None, draws=1500, family="gaussian"):
    model = bmb.Model(formula, data, family=family, priors=priors)
    return model.fit(draws=draws, tune=draws, chains=2, cores=1, random_seed=SEED,
                     progressbar=False, target_accept=0.95)


# ================================================================== A. ROPE
print(f"\n=== A. ROPE / equivalence on the session-level Gaussian models === {elapsed()}")
rope_rows = []
for label, df, dec in gr._cross_condition_rows(groups, None):
    if label in COUNT_METRICS:
        continue
    long = long_frame(df)
    rope = rope_halfwidth(label)
    idata = fit("value ~ robot*adhd + (1|pid)", long)
    rec = {"metric": gr._DID_FULL_LABELS.get(label, label), "rope": rope,
           "n": f"{len(long)}/{long['pid'].nunique()}"}
    for var, tag in (("robot", "cond"), ("adhd", "grp"), ("robot:adhd", "int")):
        rec[tag] = summarize(idata.posterior[var].values.ravel(), rope)
    rope_rows.append(rec)
    print(f"{rec['metric'][:40]:<42} ROPE +-{rope:<5} "
          f"cond P(in)={rec['cond']['p_rope']:.2f} {rope_verdict(rec['cond']['p_rope']):<10} "
          f"grp P(in)={rec['grp']['p_rope']:.2f} {rope_verdict(rec['grp']['p_rope']):<10} "
          f"int P(in)={rec['int']['p_rope']:.2f} {rope_verdict(rec['int']['p_rope'])}")

# ================================================= B. prior sensitivity
print(f"\n=== B(i). JZS Bayes factors under Cauchy prior scales === {elapsed()}")
R_SCALES = (0.354, 0.707, 1.0, 1.414)
bf_rows = []
for label, df, dec in gr._cross_condition_rows(groups, None):
    metric = gr._DID_FULL_LABELS.get(label, label)
    scopes = {"all": df.dropna()}
    scopes["adhd"] = scopes["all"][scopes["all"].index.map(groups.get) == gr.GROUP_ADHD]
    scopes["noadhd"] = scopes["all"][scopes["all"].index.map(groups.get) == gr.GROUP_CONTROL]
    for scope, sub in scopes.items():
        if len(sub) < 3:
            continue
        t = sps.ttest_rel(sub["Robot"], sub["Control"]).statistic
        bfs = [float(pg.bayesfactor_ttest(t, nx=len(sub), paired=True, r=r)) for r in R_SCALES]
        bf_rows.append({"metric": metric, "scope": scope, "n": len(sub), "bfs": bfs})
    delta = (df["Robot"] - df["Control"]).dropna()
    d_a = delta[delta.index.map(groups.get) == gr.GROUP_ADHD]
    d_c = delta[delta.index.map(groups.get) == gr.GROUP_CONTROL]
    if len(d_a) >= 2 and len(d_c) >= 2:
        t = sps.ttest_ind(d_a, d_c, equal_var=False).statistic
        bfs = [float(pg.bayesfactor_ttest(t, nx=len(d_a), ny=len(d_c), paired=False, r=r))
               for r in R_SCALES]
        bf_rows.append({"metric": metric, "scope": "delta", "n": f"{len(d_a)}/{len(d_c)}", "bfs": bfs})
counts_null = [sum(1 for r in bf_rows if r["bfs"][i] < 1 / 3) for i in range(len(R_SCALES))]
counts_alt = [sum(1 for r in bf_rows if r["bfs"][i] > 3) for i in range(len(R_SCALES))]
print(f"cells: {len(bf_rows)}; r = {R_SCALES}")
print(f"  BF10 < 1/3 (evidence for null): {counts_null}")
print(f"  BF10 > 3   (evidence for diff): {counts_alt}")

print(f"\n=== B(ii). Headline metrics under literature-sceptical / default / optimistic priors === {elapsed()}")
HEADLINE = {"Mean engagement score": 0.05, "Time within both thresholds (\\%)": 5.0}
prior_rows = []
for label, df, dec in gr._cross_condition_rows(groups, None):
    if label not in HEADLINE:
        continue
    long = long_frame(df)
    sd = float(long["value"].std())
    rope = HEADLINE[label]
    # sceptical: literature found no effect -> centred on 0, sd = a small effect (0.25 SD)
    # optimistic: medium positive effect (d = 0.5) with the same width
    variants = {
        "sceptical (lit.)": {"robot": bmb.Prior("Normal", mu=0.0, sigma=0.25 * sd)},
        "default": None,
        "optimistic": {"robot": bmb.Prior("Normal", mu=0.5 * sd, sigma=0.25 * sd)},
    }
    for name, priors in variants.items():
        idata = fit("value ~ robot*adhd + (1|pid)", long, priors=priors)
        s = summarize(idata.posterior["robot"].values.ravel(), rope)
        prior_rows.append({"metric": gr._DID_FULL_LABELS.get(label, label), "prior": name,
                           "sd": sd, "rope": rope, **s})
        print(f"{label[:34]:<36} {name:<18} cond {fmt_ci(s, 3)} P(dir)={s['pdir']:.2f} "
              f"P(in ROPE)={s['p_rope']:.2f}")

# ================================================= C. poll-level models
print(f"\n=== C. Poll-level hierarchical models (5 s polls, speech-excluded, gated) === {elapsed()}")
sdirs = ga.session_dirs()
gated = ga.gated_signals(sdirs)


def poll_frame(sig: str) -> pd.DataFrame:
    csv, col = (("engagement.csv", "score") if sig == "eng" else ("emotion.csv", "negative_share"))
    rows = []
    for (pid, cond), d in sdirs.items():
        if pid not in groups or (pid, cond, sig) in gated:
            continue
        excl = ga.speech_exclusions(d)
        series = []
        for r in ga._read_rows(d, csv):
            if not r.get(col) or not r.get("t_session_s"):
                continue
            t = float(r["t_session_s"])
            if t < 0 or ga._in_excl(excl, t):
                continue
            series.append((t, float(r[col])))
        series.sort()
        for (t_prev, v_prev), (t, v) in zip(series, series[1:]):
            if t - t_prev > ga.EPISODE_CENSOR_GAP_S:
                continue  # a gap censors the lag, like the episode logic
            rows.append({"pid": pid, "cond": cond, "t": t, "value": v, "lag": v_prev})
    out = pd.DataFrame(rows)
    out["robot"] = (out["cond"] == "Robot").astype(float) - 0.5
    out["adhd"] = (out["pid"].map(groups.get) == gr.GROUP_ADHD).astype(float) - 0.5
    return out


poll_rows = []
for sig, name, rope in (("eng", "Engagement score (5 s polls)", 0.05),
                        ("emo", "Negative-emotion share (5 s polls)", 0.05)):
    data = poll_frame(sig)
    n_desc = f"{len(data)} polls / {data['pid'].nunique()} pp / {data.groupby(['pid', 'cond']).ngroups} sessions"
    print(f"{name}: {n_desc}")
    for variant, formula in (("naive", "value ~ robot*adhd + (1 + robot|pid)"),
                             ("AR(1)", "value ~ robot*adhd + lag + (1 + robot|pid)")):
        idata = fit(formula, data, draws=1000)
        post = idata.posterior
        rec = {"metric": name, "variant": variant, "n": n_desc, "rope": rope}
        for var, tag in (("robot", "cond"), ("adhd", "grp"), ("robot:adhd", "int")):
            rec[tag] = summarize(post[var].values.ravel(), rope)
        if variant == "AR(1)":
            lag = post["lag"].values.ravel()
            rec["lag"] = summarize(lag)
            rec["cond_lr"] = summarize(post["robot"].values.ravel() / (1 - lag), rope)
        try:
            rec["rhat"] = float(az.rhat(idata, var_names=["robot", "adhd", "robot:adhd"]).to_array().max())
        except Exception:
            rec["rhat"] = float("nan")
        poll_rows.append(rec)
        extra = (f" lag {fmt_ci(rec['lag'])} steady-state cond {fmt_ci(rec['cond_lr'], 3)} "
                 f"P(in ROPE)={rec['cond_lr']['p_rope']:.2f}" if variant == "AR(1)" else "")
        print(f"  {variant:<6} cond {fmt_ci(rec['cond'], 3)} P(dir)={rec['cond']['pdir']:.2f} "
              f"P(in ROPE)={rec['cond']['p_rope']:.2f} | int {fmt_ci(rec['int'], 3)} "
              f"P={rec['int']['pdir']:.2f} | rhat {rec['rhat']:.3f}{extra}  {elapsed()}")

# ================================================= D. TLX frustration
print(f"\n=== D. NASA-TLX frustration, Bayesian paired === {elapsed()}")
post_q, _ = ga.load_qualtrics(ga.newest_file(ga.FILE_PATTERNS["post"]))
post_q = ga.clean(post_q, "POST_PID", "post")
ctrl_q, _ = ga.load_qualtrics(ga.newest_file(ga.FILE_PATTERNS["control"]))
ctrl_q = ga.clean(ctrl_q, "POST_PID", "control")
col = "POST_TLX_FRUSTRATION_1"
r_ = post_q.set_index("PID")[col].pipe(pd.to_numeric, errors="coerce")
c_ = ctrl_q.set_index("PID")[col].pipe(pd.to_numeric, errors="coerce")
pair = pd.concat([r_, c_], axis=1, keys=["Robot", "Control"]).dropna()
tlx_rows = []
for scope, sub in (("all", pair), ("adhd", pair[pair.index.map(groups.get) == gr.GROUP_ADHD]),
                   ("noadhd", pair[pair.index.map(groups.get) == gr.GROUP_CONTROL])):
    t = sps.ttest_rel(sub["Robot"], sub["Control"]).statistic
    bfs = [float(pg.bayesfactor_ttest(t, nx=len(sub), paired=True, r=r)) for r in R_SCALES]
    _, pw, _ = ga._wilcoxon_cells(sub["Robot"], sub["Control"])
    long = long_frame(sub)
    idata = fit("value ~ robot + (1|pid)", long)
    s = summarize(idata.posterior["robot"].values.ravel(), 5.0)
    tlx_rows.append({"scope": scope, "n": len(sub), "diff": float((sub["Robot"] - sub["Control"]).mean()),
                     "pw": pw, "bfs": bfs, **s})
    print(f"  {scope:<7} n={len(sub):>2} mean diff {tlx_rows[-1]['diff']:+.1f} {pw} "
          f"BF10(r=.707)={bfs[1]:.2f} [r sweep {', '.join(f'{b:.2f}' for b in bfs)}] "
          f"posterior {fmt_ci(s, 1)} P(dir)={s['pdir']:.2f} P(|d|<5)={s['p_rope']:.2f}")

# ================================================= tex out
def star(s: dict) -> str:
    return "$^{*}$" if (s["lo"] > 0 or s["hi"] < 0) else ""


la = [f"{r['metric']} & $\\pm{r['rope']:g}$ & {r['n']} & "
      + " & ".join(f"{fmt_ci(r[t], 2)}{star(r[t])} & {r[t]['p_rope']:.2f} & {rope_verdict(r[t]['p_rope'])}"
                   for t in ("cond", "grp", "int")) + " \\\\"
      for r in rope_rows]
SCOPE_NAMES = {"all": "All (paired)", "adhd": "ADHD (paired)", "noadhd": "No-ADHD (paired)", "delta": "$\\Delta$ between"}
lb = [f"{r['metric']} & {SCOPE_NAMES[r['scope']]} & {r['n']} & "
      + " & ".join(f"{b:.2f}" for b in r["bfs"]) + " \\\\" for r in bf_rows]
lb_summary = ("\\midrule\n\\multicolumn{3}{l}{Cells with $\\mathrm{BF}_{10} < 1/3$ (of "
              f"{len(bf_rows)})}} & " + " & ".join(str(c) for c in counts_null)
              + " \\\\\n\\multicolumn{3}{l}{Cells with $\\mathrm{BF}_{10} > 3$} & "
              + " & ".join(str(c) for c in counts_alt) + " \\\\")
lb2 = [f"{r['metric']} & {r['prior']} & {fmt_ci(r, 3)} & {r['pdir']:.2f} & {r['p_rope']:.2f} \\\\"
       for r in prior_rows]
lc = []
for r in poll_rows:
    lc.append(f"{r['metric']} & {r['variant']} & {fmt_ci(r['cond'], 3)}{star(r['cond'])} & "
              f"{r['cond']['pdir']:.2f} & {r['cond']['p_rope']:.2f} & "
              f"{fmt_ci(r['int'], 3)}{star(r['int'])} & {r['int']['pdir']:.2f} & "
              + (f"{fmt_ci(r['lag'], 2)} & {fmt_ci(r['cond_lr'], 3)} & {r['cond_lr']['p_rope']:.2f}"
                 if "lag" in r else "-- & -- & --") + " \\\\")
ld = [f"{SCOPE_NAMES[r['scope']]} & {r['n']} & {r['diff']:+.1f} & {r['pw']} & "
      + " & ".join(f"{b:.2f}" for b in r["bfs"]) + f" & {fmt_ci(r, 1)}{star(r)} & {r['pdir']:.2f} & {r['p_rope']:.2f} \\\\"
      for r in tlx_rows]
n_polls = {r["metric"]: r["n"] for r in poll_rows}

tex = r"""\documentclass[10pt]{article}
\usepackage[margin=1.4cm,landscape]{geometry}
\usepackage{booktabs}
\begin{document}\pagestyle{empty}
\begin{center}\Large Bayesian follow-up experiments\end{center}
\noindent\textbf{A. ROPE / equivalence} --- Gaussian hierarchical model, posterior mean [95\% CrI],
$P(\mathrm{in})$ = posterior mass inside $\pm$ROPE; verdict: equivalent $\ge .95$, different $\le .05$.\par\vspace{0.4em}
\begin{center}\scriptsize\setlength{\tabcolsep}{3pt}
\begin{tabular}{llc rcl rcl rcl}
\toprule
 & & & \multicolumn{3}{c}{Condition} & \multicolumn{3}{c}{Group} & \multicolumn{3}{c}{Interaction} \\
\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}
Measure & ROPE & $n$ & CrI & $P(\mathrm{in})$ & verdict & CrI & $P(\mathrm{in})$ & verdict & CrI & $P(\mathrm{in})$ & verdict \\
\midrule
""" + "\n".join(la) + r"""
\bottomrule
\end{tabular}\end{center}
\vspace{0.8em}
\noindent\textbf{B(i). JZS Bayes factors under four Cauchy prior scales} ($r$; default 0.707).\par\vspace{0.4em}
\begin{center}\scriptsize\setlength{\tabcolsep}{4pt}
\begin{tabular}{llc rrrr}
\toprule
Measure & Scope & $n$ & $r=0.354$ & $r=0.707$ & $r=1.0$ & $r=1.414$ \\
\midrule
""" + "\n".join(lb) + "\n" + lb_summary + r"""
\bottomrule
\end{tabular}\end{center}
\vspace{0.8em}
\noindent\textbf{B(ii). Headline metrics under a literature-sceptical, the default, and an optimistic prior on the condition effect}
(sceptical: Normal(0, 0.25\,SD); optimistic: Normal(+0.5\,SD, 0.25\,SD)).\par\vspace{0.4em}
\begin{center}\scriptsize\setlength{\tabcolsep}{4pt}
\begin{tabular}{llrcc}
\toprule
Measure & Prior & Condition [CrI] & $P(\mathrm{dir})$ & $P(\mathrm{in\ ROPE})$ \\
\midrule
""" + "\n".join(lb2) + r"""
\bottomrule
\end{tabular}\end{center}
\vspace{0.8em}
\noindent\textbf{C. Poll-level hierarchical models} (value $\sim$ robot $\times$ adhd [+ lag] + (1 + robot $|$ participant);
""" + "; ".join(f"{k}: {v}" for k, v in n_polls.items()) + r""". Steady-state = robot/(1 $-$ lag). ROPE $\pm 0.05$).\par\vspace{0.4em}
\begin{center}\scriptsize\setlength{\tabcolsep}{3pt}
\begin{tabular}{ll rcc rc rrc}
\toprule
Measure & Model & Condition [CrI] & $P(\mathrm{dir})$ & $P(\mathrm{in})$ & Interaction [CrI] & $P(\mathrm{dir})$ & Lag [CrI] & Steady-state cond.\ [CrI] & $P(\mathrm{in})$ \\
\midrule
""" + "\n".join(lc) + r"""
\bottomrule
\end{tabular}\end{center}
\vspace{0.8em}
\noindent\textbf{D. NASA-TLX frustration, robot $-$ no-robot} (0--100; hierarchical posterior of the condition effect; ROPE $\pm 5$).\par\vspace{0.4em}
\begin{center}\scriptsize\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcrl rrrr rcc}
\toprule
Scope & $n$ & Mean diff & $p_W$ & $\mathrm{BF}_{10}$ $r{=}.354$ & $.707$ & $1.0$ & $1.414$ & Posterior [CrI] & $P(\mathrm{dir})$ & $P(|d|<5)$ \\
\midrule
""" + "\n".join(ld) + r"""
\bottomrule
\end{tabular}\end{center}
\end{document}
"""
open(os.path.join(OUT_DIR, "bayes_followup.tex"), "w").write(tex)
print(f"\nwrote {os.path.join(OUT_DIR, 'bayes_followup.tex')} {elapsed()}")
