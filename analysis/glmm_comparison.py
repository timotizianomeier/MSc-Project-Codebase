"""GLMM robustness check (Nicole 04.09): per session-level metric, fit a
Gaussian linear mixed model  value ~ robot * adhd  with a random intercept
per participant (robot: 1 = Robot session, adhd: 1 = ADHD group), and put
its Wald p-values next to the pipeline's existing nonparametric/t twins:

  interaction (robot x adhd)  <->  MWU / Welch t on per-participant deltas
  condition main effect       <->  paired Wilcoxon / paired t (all 22)
  group main effect           <->  MWU / Welch t on condition-pooled means

For the three intervention-count metrics a Poisson GEE (exchangeable
working correlation, robust SEs) supplies the properly 'generalized'
interaction p as well.  Terminal summary + LaTeX table for a preview PDF.
NB with two observations per participant the Gaussian LMM interaction is
the model-based analogue of the delta comparison, so agreement is the
expected outcome; LMM uses all available sessions (incl. unpaired ones).

Run:  .venv/bin/python glmm_comparison.py   (writes output/glmmdoc.tex;
compile that with pdflatex for the comparison PDF).
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as sps

import generate_results as gr

warnings.filterwarnings("ignore")

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "output")  # gitignored, like the fragment output
os.makedirs(OUT_DIR, exist_ok=True)

pre, _ = gr.load_qualtrics(gr.newest_file(gr.FILE_PATTERNS["pre"]))
pre = gr.clean(pre, "PRE_PID", "pre")
groups = gr.assign_groups(pre)

COUNT_METRICS = {"Engagement interv.", "Emotion interv.", "All interv."}


def fmt_p(p):
    if p != p:
        return "--"
    return "<.001" if p < 0.001 else f"{p:.3f}"


rows = []
for label, df, dec in gr._cross_condition_rows(groups, None):
    # long format; LMM keeps participants with only one usable session
    long = df.stack().rename("value").reset_index()
    long.columns = ["pid", "cond", "value"]
    long = long.dropna(subset=["value"]).reset_index(drop=True)
    # effects coding (+-0.5): the 'main' coefficients then estimate each
    # factor's effect averaged over the other (a 0/1-coded model with the
    # interaction present would give simple effects at the reference
    # level instead); the interaction coefficient is coding-invariant.
    long["robot"] = (long["cond"] == "Robot").astype(float) - 0.5
    long["adhd"] = (long["pid"].map(groups.get)
                    == gr.GROUP_ADHD).astype(float) - 0.5

    # existing pipeline twins -------------------------------------------
    paired = df.dropna()
    delta = paired["Robot"] - paired["Control"]
    d_a = delta[delta.index.map(groups.get) == gr.GROUP_ADHD]
    d_c = delta[delta.index.map(groups.get) == gr.GROUP_CONTROL]
    pooled = df.mean(axis=1)  # per-participant mean across conditions
    m_a = pooled.dropna()[lambda s: s.index.map(groups.get) == gr.GROUP_ADHD]
    m_c = pooled.dropna()[lambda s: s.index.map(groups.get) == gr.GROUP_CONTROL]

    def safe(f, *a):
        try:
            return float(f(*a).pvalue)
        except Exception:
            return float("nan")

    p_int_u = (safe(lambda a, c: sps.mannwhitneyu(a, c), d_a, d_c)
               if len(d_a) and len(d_c) else float("nan"))
    p_int_t = (safe(lambda a, c: sps.ttest_ind(a, c, equal_var=False),
                    d_a, d_c)
               if len(d_a) > 1 and len(d_c) > 1 else float("nan"))
    p_cond_w = (safe(lambda r, c: sps.wilcoxon(r, c),
                     paired["Robot"], paired["Control"])
                if len(paired) > 1 and (delta != 0).any() else float("nan"))
    p_cond_t = (safe(lambda r, c: sps.ttest_rel(r, c),
                     paired["Robot"], paired["Control"])
                if len(paired) > 1 else float("nan"))
    p_grp_u = (safe(lambda a, c: sps.mannwhitneyu(a, c), m_a, m_c)
               if len(m_a) and len(m_c) else float("nan"))
    p_grp_t = safe(lambda a, c: sps.ttest_ind(a, c, equal_var=False),
                   m_a, m_c)

    # Gaussian LMM -------------------------------------------------------
    lmm = dict.fromkeys(["robot", "adhd", "robot:adhd"], float("nan"))
    n_obs, n_pid, note = len(long), long["pid"].nunique(), ""
    try:
        fit = smf.mixedlm("value ~ robot * adhd", long,
                          groups=long["pid"]).fit(reml=True)
        for k in lmm:
            lmm[k] = float(fit.pvalues[k])
        if not fit.converged:
            note = "nc"
    except Exception:
        note = "fail"

    # Poisson GEE for counts --------------------------------------------
    p_gee = float("nan")
    if label in COUNT_METRICS:
        try:
            gee = smf.gee("value ~ robot * adhd", groups="pid", data=long,
                          family=sm.families.Poisson(),
                          cov_struct=sm.cov_struct.Exchangeable()).fit()
            p_gee = float(gee.pvalues["robot:adhd"])
        except Exception:
            pass

    rows.append({
        "metric": gr._DID_FULL_LABELS.get(label, label),
        "n": f"{n_obs}/{n_pid}", "note": note,
        "p_int_lmm": lmm["robot:adhd"], "p_int_gee": p_gee,
        "p_int_u": p_int_u, "p_int_t": p_int_t,
        "p_cond_lmm": lmm["robot"], "p_cond_w": p_cond_w,
        "p_cond_t": p_cond_t,
        "p_grp_lmm": lmm["adhd"], "p_grp_u": p_grp_u, "p_grp_t": p_grp_t,
    })

# ------------------------------------------------------------------ output
hdr = (f"{'metric':<42} {'n':>6} | {'int LMM':>8} {'GEE':>6} {'pU(d)':>6} "
       f"{'pt(d)':>6} | {'cond LMM':>8} {'pW':>6} {'pt':>6} | "
       f"{'grp LMM':>8} {'pU':>6} {'pt':>6}")
print("\n" + hdr + "\n" + "-" * len(hdr))
for r in rows:
    print(f"{r['metric'][:42]:<42} {r['n']:>6} | "
          f"{fmt_p(r['p_int_lmm']):>8} {fmt_p(r['p_int_gee']):>6} "
          f"{fmt_p(r['p_int_u']):>6} {fmt_p(r['p_int_t']):>6} | "
          f"{fmt_p(r['p_cond_lmm']):>8} {fmt_p(r['p_cond_w']):>6} "
          f"{fmt_p(r['p_cond_t']):>6} | "
          f"{fmt_p(r['p_grp_lmm']):>8} {fmt_p(r['p_grp_u']):>6} "
          f"{fmt_p(r['p_grp_t']):>6}"
          + (f"  [{r['note']}]" if r["note"] else ""))

flips = [r["metric"] for r in rows for a, b in
         [("p_int_lmm", "p_int_u"), ("p_cond_lmm", "p_cond_w"),
          ("p_grp_lmm", "p_grp_u")]
         if r[a] == r[a] and r[b] == r[b]
         and (r[a] < .05) != (r[b] < .05)]
print("\nLMM-vs-nonparametric alpha=.05 disagreements:",
      sorted(set(flips)) or "none")

# LaTeX table ---------------------------------------------------------


def cell(p):
    if p != p:
        return "{--}"
    return "{$<.001$}" if p < 0.001 else f"{p:.3f}"


lines = []
for r in rows:
    lines.append(
        f"{r['metric']} & {r['n']} & "
        f"{cell(r['p_int_lmm'])} & {cell(r['p_int_gee'])} & "
        f"{cell(r['p_int_u'])} & {cell(r['p_int_t'])} & "
        f"{cell(r['p_cond_lmm'])} & {cell(r['p_cond_w'])} & "
        f"{cell(r['p_cond_t'])} & "
        f"{cell(r['p_grp_lmm'])} & {cell(r['p_grp_u'])} & "
        f"{cell(r['p_grp_t'])} \\\\")
body = "\n".join(lines)
tex = r"""\documentclass[10pt]{article}
\usepackage[margin=1.8cm,landscape]{geometry}
\usepackage{booktabs,siunitx}
\sisetup{table-format=1.3}
\begin{document}
\pagestyle{empty}
\begin{center}\Large GLMM robustness check vs.\ pipeline tests\end{center}
\noindent Gaussian linear mixed model per metric:
$\mathrm{value} \sim \mathrm{robot} \times \mathrm{adhd}$ with a random
intercept per participant; factors effects-coded ($\pm 0.5$: Robot vs.\
No-Robot, ADHD vs.\ No-ADHD), so each main-effect coefficient is that
factor's effect averaged over the other; Wald $p$-values (REML).
GEE = Poisson
generalized estimating equations with robust SEs, intervention counts
only.  $n$ = sessions/participants entering the LMM (unpaired sessions
included; the nonparametric twins use their usual pairings).  Signal
metrics carry the speech exclusion.
\vspace{0.8em}

\begin{center}\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{l c SSSS SSS SSS}
\toprule
 & & \multicolumn{4}{c}{Interaction (robot $\times$ group)}
 & \multicolumn{3}{c}{Condition main effect}
 & \multicolumn{3}{c}{Group main effect} \\
\cmidrule(lr){3-6} \cmidrule(lr){7-9} \cmidrule(lr){10-12}
Measure & {$n$} & {LMM} & {GEE} & {$p_U(\Delta)$} & {$p_t(\Delta)$}
 & {LMM} & {$p_W$} & {$p_t$} & {LMM} & {$p_U$} & {$p_t$} \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{center}
\end{document}
"""
texpath = os.path.join(OUT_DIR, "glmmdoc.tex")
open(texpath, "w").write(tex)
print("wrote", texpath)
