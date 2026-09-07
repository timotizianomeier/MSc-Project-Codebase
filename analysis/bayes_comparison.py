"""Bayesian first pass (Nicole 07.09) — PAPER-facing exploration only;
the thesis stays frequentist. Two layers per session-level metric:

  A. JZS Bayes factors (pingouin): BF10 for robot-vs-no-robot (paired,
     per scope All/ADHD/No-ADHD) and for the between-group deltas
     (independent), printed next to the frequentist p's. NB these are
     t-statistic-based BFs — the Bayesian twin of the t-tests we no
     longer report — so treat them as a translation aid, not gospel.
  B. Bayesian hierarchical model (bambi/PyMC):
     value ~ robot*adhd + (1|pid), factors effects-coded +-0.5 exactly
     like glmm_comparison.glmm_rows, gaussian family (poisson for the
     three intervention counts). Reported: posterior mean, 95% credible interval,
     P(direction) for condition, group, interaction — the no-p-value
     analogue of the LMM table.

Interpretation bands used in the verdict column: BF10 > 3 moderate,
> 10 strong evidence for a difference; BF10 < 1/3 moderate evidence FOR
the null (what frequentist non-significance can never give you); a
credible interval excluding 0 ~ 'credibly non-zero'.

Run: cd analysis && .venv/bin/python <path>/bayes_comparison.py
Writes bayesdoc.tex next to itself; compile with pdflatex.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pingouin as pg

import generate_results as gr

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "output")  # gitignored
os.makedirs(OUT_DIR, exist_ok=True)
COUNT_METRICS = {"Engagement interv.", "Emotion interv.", "All interv."}

pre, _ = gr.load_qualtrics(gr.newest_file(gr.FILE_PATTERNS["pre"]))
pre = gr.clean(pre, "PRE_PID", "pre")
groups = gr.assign_groups(pre)


def fmt_bf(bf):
    if bf != bf:
        return "--"
    if bf >= 100:
        return f"{bf:.0f}"
    return f"{bf:.2f}"


def fmt_p(p):
    if p != p:
        return "--"
    return "<.001" if p < 0.001 else f"{p:.3f}"


# ---------------------------------------------------------------- part A
print("\n=== A. Bayes factors (JZS, t-based) vs frequentist p ===")
bf_rows = []
for label, df, dec in gr._cross_condition_rows(groups, None):
    rec = {"metric": gr._DID_FULL_LABELS.get(label, label)}
    for scope, g in (("all", None), ("adhd", gr.GROUP_ADHD),
                     ("noadhd", gr.GROUP_CONTROL)):
        sub = df if g is None else df[df.index.map(groups.get) == g]
        sub = sub.dropna()
        if len(sub) >= 3:
            t = pg.ttest(sub["Robot"], sub["Control"], paired=True)
            bf = float(t["BF10"].iloc[0])
            _, pw, _ = gr._wilcoxon_cells(sub["Robot"], sub["Control"])
        else:
            bf, pw = float("nan"), "--"
        rec[f"bf_{scope}"] = bf
        rec[f"pw_{scope}"] = (pw.replace("$", "").replace("p = ", "")
                              .replace("p < ", "<") if isinstance(pw, str)
                              else pw)
    delta = (df["Robot"] - df["Control"]).dropna()
    d_a = delta[delta.index.map(groups.get) == gr.GROUP_ADHD]
    d_c = delta[delta.index.map(groups.get) == gr.GROUP_CONTROL]
    if len(d_a) >= 2 and len(d_c) >= 2:
        t = pg.ttest(d_a, d_c, paired=False)
        rec["bf_delta"] = float(t["BF10"].iloc[0])
        _, pu = gr._mwu_cells(d_a, d_c)
        rec["pu_delta"] = (pu.replace("$", "").replace("p = ", "")
                           .replace("p < ", "<"))
    else:
        rec["bf_delta"], rec["pu_delta"] = float("nan"), "--"
    bf_rows.append(rec)
    print(f"{rec['metric'][:42]:<44}"
          f"all BF={fmt_bf(rec['bf_all']):>7} (pW {rec['pw_all']:>6})  "
          f"ADHD BF={fmt_bf(rec['bf_adhd']):>7} (pW {rec['pw_adhd']:>6})  "
          f"noADHD BF={fmt_bf(rec['bf_noadhd']):>7} "
          f"(pW {rec['pw_noadhd']:>6})  "
          f"delta BF={fmt_bf(rec['bf_delta']):>6} (pU {rec['pu_delta']})")

# ---------------------------------------------------------------- part B
print("\n=== B. Bayesian hierarchical model (bambi/PyMC) ===")
import arviz as az
import bambi as bmb

hier_rows = []
for label, df, dec in gr._cross_condition_rows(groups, None):
    long = df.stack().rename("value").reset_index()
    long.columns = ["pid", "cond", "value"]
    long = long.dropna(subset=["value"]).reset_index(drop=True)
    long["robot"] = (long["cond"] == "Robot").astype(float) - 0.5
    long["adhd"] = (long["pid"].map(groups.get)
                    == gr.GROUP_ADHD).astype(float) - 0.5
    fams = (["poisson", "negativebinomial"]
            if label in COUNT_METRICS else ["gaussian"])
    if label in COUNT_METRICS:
        long["value"] = long["value"].round().astype(int)
    for fam in fams:
      try:
        model = bmb.Model("value ~ robot*adhd + (1|pid)", long, family=fam)
        idata = model.fit(draws=1500, tune=1500, chains=2, cores=1,
                          random_seed=42, progressbar=False,
                          target_accept=0.95)
      except Exception as e:
        print(f"{label} [{fam}]: FIT FAILED — {e}")
        continue
      post = idata.posterior
      rec = {"metric": gr._DID_FULL_LABELS.get(label, label), "family": fam,
             "n": f"{len(long)}/{long['pid'].nunique()}"}
      try:  # arviz API drifted (hdi_prob removed); rhat is best-effort
          bad = float(az.rhat(idata, var_names=[
              "robot", "adhd", "robot:adhd"]).to_array().max())
      except Exception:
          bad = float("nan")
      rec["r_hat"] = bad
      for var, tag in (("robot", "cond"), ("adhd", "grp"),
                       ("robot:adhd", "int")):
          draws = post[var].values.ravel()
          lo, hi = np.percentile(draws, [2.5, 97.5])  # equal-tailed 95% CrI
          pdir = max((draws > 0).mean(), (draws < 0).mean())
          rec[f"{tag}_mean"] = float(draws.mean())
          rec[f"{tag}_lo"] = float(lo)
          rec[f"{tag}_hi"] = float(hi)
          rec[f"{tag}_pdir"] = float(pdir)
          rec[f"{tag}_cred"] = (lo > 0) or (hi < 0)
      hier_rows.append(rec)
      flag = " <-- r_hat!" if bad > 1.01 else ""
      print(f"{rec['metric'][:40]:<42} [{fam[:5]}]"
            f" cond {rec['cond_mean']:+.3f} [{rec['cond_lo']:+.3f},"
            f"{rec['cond_hi']:+.3f}] P={rec['cond_pdir']:.2f}"
            f" | grp {rec['grp_mean']:+.3f} P={rec['grp_pdir']:.2f}"
            f" | int {rec['int_mean']:+.3f} [{rec['int_lo']:+.3f},"
            f"{rec['int_hi']:+.3f}] P={rec['int_pdir']:.2f}{flag}")

# ---------------------------------------------------------------- verdicts
print("\n=== Verdicts: where Bayes and frequentist part ways ===")
findings = []
for rec in bf_rows:
    for scope in ("all", "adhd", "noadhd", "delta"):
        bf = rec.get(f"bf_{scope}", float("nan"))
        if bf == bf and bf >= 3:
            findings.append(f"BF10={fmt_bf(bf)} [{scope}] {rec['metric']}")
        if bf == bf and bf <= 1 / 3:
            findings.append(f"BF10={fmt_bf(bf)} [{scope}] {rec['metric']}"
                            " — evidence FOR the null")
for rec in hier_rows:
    for tag, name in (("cond", "condition"), ("grp", "group"),
                      ("int", "interaction")):
        if rec.get(f"{tag}_cred"):
            findings.append(
                f"CrI excludes 0: {name} of {rec['metric']} "
                f"({rec[f'{tag}_mean']:+.3f} "
                f"[{rec[f'{tag}_lo']:+.3f},{rec[f'{tag}_hi']:+.3f}])")
print("\n".join(findings) if findings else "none")

# ---------------------------------------------------------------- tex out
def esc_pct(s):
    return s  # labels already escaped by the pipeline


lines_a = []
for r in bf_rows:
    lines_a.append(
        f"{r['metric']} & {fmt_bf(r['bf_all'])} & {{{r['pw_all']}}} & "
        f"{fmt_bf(r['bf_adhd'])} & {{{r['pw_adhd']}}} & "
        f"{fmt_bf(r['bf_noadhd'])} & {{{r['pw_noadhd']}}} & "
        f"{fmt_bf(r['bf_delta'])} & {{{r['pu_delta']}}} \\\\")
lines_b = []
for r in hier_rows:
    def ci(tag):
        return (f"{r[f'{tag}_mean']:+.2f} [{r[f'{tag}_lo']:+.2f}, "
                f"{r[f'{tag}_hi']:+.2f}]"
                + ("$^{*}$" if r[f"{tag}_cred"] else ""))
    lines_b.append(
        f"{r['metric']} & {r['family']} & {r['n']} & "
        f"{ci('cond')} & {r['cond_pdir']:.2f} & "
        f"{ci('grp')} & {r['grp_pdir']:.2f} & "
        f"{ci('int')} & {r['int_pdir']:.2f} \\\\")
tex = r"""\documentclass[10pt]{article}
\usepackage[margin=1.6cm,landscape]{geometry}
\usepackage{booktabs,siunitx}
\begin{document}\pagestyle{empty}
\begin{center}\Large Bayesian first pass vs.\ frequentist results\end{center}
\noindent\textbf{A. JZS Bayes factors} (t-based; BF10 $>3$ moderate,
$>10$ strong evidence for a difference; $<1/3$ moderate evidence
\emph{for the null}) next to the reported $p_W$/$p_U$.\par\vspace{0.5em}
\begin{center}\small\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrlrlrlrl}
\toprule
 & \multicolumn{2}{c}{All (paired)} & \multicolumn{2}{c}{ADHD (paired)}
 & \multicolumn{2}{c}{No-ADHD (paired)} & \multicolumn{2}{c}{$\Delta$ between} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
Measure & {BF10} & {$p_W$} & {BF10} & {$p_W$} & {BF10} & {$p_W$}
 & {BF10} & {$p_U$} \\
\midrule
""" + "\n".join(lines_a) + r"""
\bottomrule
\end{tabular}\end{center}
\vspace{1em}
\noindent\textbf{B. Bayesian hierarchical model}
(value $\sim$ condition $\times$ group + (1$|$participant), effects-coded;
posterior mean [95\% CrI], $P$ = probability the effect has the
posterior-majority sign; $^{*}$ = CrI excludes 0; poisson family for
counts, so those effects are on the log scale).\par\vspace{0.5em}
\begin{center}\small\setlength{\tabcolsep}{4pt}
\begin{tabular}{llcrcrcrc}
\toprule
Measure & family & $n$ & {Condition [CrI]} & {$P$}
 & {Group [CrI]} & {$P$} & {Interaction [CrI]} & {$P$} \\
\midrule
""" + "\n".join(lines_b) + r"""
\bottomrule
\end{tabular}\end{center}
\end{document}
"""
open(os.path.join(OUT_DIR, "bayesdoc.tex"), "w").write(tex)
print("\nwrote", os.path.join(OUT_DIR, "bayesdoc.tex"))
