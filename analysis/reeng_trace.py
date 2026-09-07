"""Verification aid for the re-engagement analysis (_post_cue_recovery):
re-implements the per-cue logic step by step with verbose printing for a
handful of cues, so each number can be checked against the raw session
CSVs (events.csv, speech.csv, engagement/emotion.csv). Picks the first
robot emotion session with a normal recovery, an in-dialogue recovery
(0 s) and a censored cue, whichever come first."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import generate_results as gr
from generate_results import (_DIALOGUE_CHAIN_GAP_S, _log_rows,
                              gated_signals, session_dirs, signal_polls)

pre, _ = gr.load_qualtrics(gr.newest_file(gr.FILE_PATTERNS["pre"]))
pre = gr.clean(pre, "PRE_PID", "pre")
groups = gr.assign_groups(pre)

sdirs = session_dirs()
gated = gated_signals(sdirs)
want = {"normal": None, "zero": None, "censored": None}

for (pid, cond), d in sorted(sdirs.items()):
    if cond != "Robot" or pid not in groups:
        continue
    for sig in ("emo", "eng"):
        if (pid, cond, sig) in gated:
            continue
        polls = signal_polls(d, sig)
        if not polls:
            continue
        base = "engagement" if sig == "eng" else "emotion"
        cues = [float(r["t_session_s"]) for r in _log_rows(d, "events.csv")
                if r["event_type"] == f"intervention_{base}"
                and r["t_session_s"]]
        speech = sorted((float(r["t_start_s"]), float(r["t_end_s"]))
                        for r in _log_rows(d, "speech.csv")
                        if r["t_start_s"])
        for t0 in cues:
            segs = [sg for sg in speech if sg[0] >= t0]
            if not segs:
                continue
            chain = [segs[0]]
            end = segs[0][1]
            for s0, s1 in segs[1:]:
                if s0 - end <= _DIALOGUE_CHAIN_GAP_S:
                    chain.append((s0, s1))
                    end = max(end, s1)
                else:
                    break
            rec = next((t for t, act in polls if t >= end and not act),
                       None)
            if rec is None:
                kind = "censored"
            elif rec - end <= 0:
                kind = "zero"
            else:
                kind = "normal"
            if want[kind] is None:
                want[kind] = (pid, cond, sig, t0, chain, end, rec, polls)
    if all(want.values()):
        break

for kind, item in want.items():
    if item is None:
        print(f"[{kind}] no example found")
        continue
    pid, cond, sig, t0, chain, end, rec, polls = item
    print(f"\n=== {kind.upper()} example: P{pid} {cond} signal={sig} ===")
    print(f"  intervention cue at t = {t0:.1f}s  (events.csv)")
    print(f"  dialogue chain ({len(chain)} speech segments, gaps <= "
          f"{_DIALOGUE_CHAIN_GAP_S:.0f}s):")
    for s0, s1 in chain[:6]:
        print(f"    speech {s0:7.1f} -> {s1:7.1f}s")
    if len(chain) > 6:
        print(f"    ... {len(chain) - 6} more segments")
    print(f"  dialogue end = {end:.1f}s")
    if rec is None:
        print("  no back-at-threshold poll before session end -> CENSORED")
    else:
        around = [(t, act) for t, act in polls if end - 12 <= t <= rec + 6]
        print(f"  polls around dialogue end (t, below-threshold?):")
        for t, act in around[:8]:
            print(f"    {t:7.1f}s  {'BELOW' if act else 'ok'}")
        print(f"  first at-threshold poll at t = {rec:.1f}s "
              f"-> duration = max({rec:.1f} - {end:.1f}, 0) "
              f"= {max(rec - end, 0):.1f}s")
