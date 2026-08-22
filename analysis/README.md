# Analysis scripts

## Appendix generator

Generates the thesis appendix fragments (tables + pgfplots charts) from the
Qualtrics CSV exports and pushes them into the Overleaf-linked thesis repo.

### Regenerate (the usual round)

1. Export all three surveys from Qualtrics ("Use numeric values") and drop
   the CSVs into the Box folder
   `MSc-Project-Storage-Timo/study-data/qualtrics/` — no renaming needed,
   the newest `pre-session-*` / `control-session-*` / `post-session-*` file
   of each kind is used automatically.
2. New participant since last time? Add the PID to `INCLUDE_PIDS` in
   `generate_appendix.py`.
3. ```bash
   cd ~/Projects/MSc-Project-Codebase/analysis
   .venv/bin/python compute_asrs.py        # rescore ASRS -> participant_groups.txt
   .venv/bin/python generate_appendix.py --sync
   ```
4. In Overleaf: Menu -> GitHub -> "Pull GitHub changes" -> Recompile.

Without `--sync` the fragments are only written to `analysis/output/`
(gitignored — they contain verbatim participant answers). With `--sync` they
are also copied into `~/Projects/MSc-Project-Final-Report/apx-subfiles/`,
committed, and pushed (the thesis repo is PRIVATE and must stay private).

If the push is rejected (edits made in Overleaf since the last sync):
`cd ~/Projects/MSc-Project-Final-Report && git pull --rebase && git push`.

### Where things are configured

All knobs sit at the top of `generate_appendix.py`: data/output/thesis paths,
chart layout (`LIKERT_ROWS_PER_PAGE`, `LIKERT_CHART_HEIGHT`), open-ended
subheaders (`OE_HEADERS`), participant whitelist (`INCLUDE_PIDS`), grouping
mode + ASRS scoring knobs, and `PID_DISPLAY` (display-only renumbering
P11->P1 etc. — filenames and raw data always keep true PIDs; authoritative
mapping: `Participant_Linking_File.xlsx` on Box).

`compute_asrs.py` overwrites `participant_groups.txt` (next to the CSVs on
Box) on every run — re-apply manual group overrides afterwards, or edit the
file by hand and skip rerunning the scorer. `--dry-run` prints without
writing. The scorer also flags participants whose ASRS group contradicts
their self-reported diagnosis.

### Session-log parser

`parse_app_log.py <app log>` — stdlib-only; parses one robot/control app log
into `analysis/logs/<name>_csv/` (engagement.csv, emotion.csv, speech.csv,
events.csv, session.json). Run per session during study days; the CSV dirs
are synced to Box `study-data/analysis-csv/` by the end-of-day rsync.

### Environment

`analysis/.venv` (pandas + numpy), separate from the study app's frozen venv:

```bash
python3 -m venv .venv && .venv/bin/pip install pandas numpy
```
