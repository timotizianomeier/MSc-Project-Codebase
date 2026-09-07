"""Verification aid for glmm_comparison.py (Timo's line-by-line check):
shows, for three representative metrics, that the Gaussian LMM's numbers
are reproducible by elementary calculations on the same data.

  1. interaction coefficient  ==  (mean ADHD delta) - (mean No-ADHD delta)
     when every participant is paired (2 obs each); with unpaired
     sessions kept in the LMM it moves slightly - both shown.
  2. interaction p  ~  equal-variance two-sample t on the per-participant
     deltas (the classic equivalence; Wald z vs t and unpaired sessions
     explain small gaps).
  3. full MixedLM summary for one metric to eyeball against the code.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as sps

import generate_results as gr

pre, _ = gr.load_qualtrics(gr.newest_file(gr.FILE_PATTERNS["pre"]))
pre = gr.clean(pre, "PRE_PID", "pre")
groups = gr.assign_groups(pre)

PICK = ["All interv.", "Mean engagement score",
        "Time within both thresholds (\\%)"]

for label, df, dec in gr._cross_condition_rows(groups, None):
    if label not in PICK:
        continue
    long = df.stack().rename("value").reset_index()
    long.columns = ["pid", "cond", "value"]
    long = long.dropna(subset=["value"]).reset_index(drop=True)
    long["robot"] = (long["cond"] == "Robot").astype(float) - 0.5
    long["adhd"] = (long["pid"].map(groups.get)
                    == gr.GROUP_ADHD).astype(float) - 0.5

    fit = smf.mixedlm("value ~ robot * adhd", long,
                      groups=long["pid"]).fit(reml=True)

    delta = (df["Robot"] - df["Control"]).dropna()
    d_a = delta[delta.index.map(groups.get) == gr.GROUP_ADHD]
    d_c = delta[delta.index.map(groups.get) == gr.GROUP_CONTROL]
    t_eq = sps.ttest_ind(d_a, d_c, equal_var=True)

    print(f"\n=== {label} ===")
    print(f"  LMM interaction coef = {fit.params['robot:adhd']:+.4f}   "
          f"hand calc (mean ADHD delta - mean No-ADHD delta) = "
          f"{d_a.mean() - d_c.mean():+.4f}")
    print(f"  LMM interaction p    = {fit.pvalues['robot:adhd']:.4f}   "
          f"equal-var t on deltas p = {t_eq.pvalue:.4f}   "
          f"(paired n: {len(d_a)} ADHD / {len(d_c)} No-ADHD; "
          f"LMM sessions: {len(long)})")
    if label == "All interv.":
        print("\n--- full MixedLM summary (All interv.) ---")
        print(fit.summary())
