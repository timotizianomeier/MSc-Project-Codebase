# MSc Project — Study Assistant Robot for Students with ADHD

Code for a within-subjects study comparing a Reachy Mini study-assistant robot against
a no-robot control condition during 45-minute study sessions. 22 Imperial students took
part (12 with ADHD, 10 without), each doing one session per condition.

MSc Computing final project, Imperial College London, supervised by Prof. Nicole
Salomons at the PAIR Lab.

## Repository layout

| Path | What it is |
|------|------------|
| `study_assistant/` | The app that ran on the robot's Raspberry Pi. A fork of Pollen Robotics' [Reachy Mini conversation app](https://github.com/pollen-robotics/reachy_mini_conversation_app); the study adds the engagement and emotion monitors, the intervention gate, the participant task page, and session logging, and locks the robot to the `adhd_study_assistant_v2` persona. |
| `analysis/` | Everything that turns session logs and Qualtrics exports into the tables and charts in the report — see [`analysis/README.md`](analysis/README.md). |
| `experiments/` | Standalone spikes behind the design decisions: emotion-model comparisons (`emotiefflib`, `deepface`, `fer-2013`) and the persona and LLM test harness (`persona_harness`). |
| `logs/` | Annotated index of the study and rehearsal sessions. |

The no-robot control condition was run by
[`experiments/control/control_session.py`](experiments/control/control_session.py) — a
standalone runner using a plain webcam and no robot, which imports the monitors,
thresholds, cadences, recorder and participant page from the app so both conditions
share one implementation and interventions are logged as counterfactuals instead of
being delivered.

The engagement scorer is not in this repository: the app calls it over HTTP, and it
lives in [`engagement_detector`](https://github.com/timotizianomeier/engagement_detector)
— a fork of [LCAS/engagement_detector](https://github.com/LCAS/engagement_detector)
([Del Duchetto et al., 2020](https://doi.org/10.3389/frobt.2020.00116)) ported from ROS
to a standalone Python 3 / TF 2.15 scoring service. It has to be running for the robot
to sense engagement.

Working notes kept throughout the project: `daily-log.md` (what happened when),
`design-decisions.md` (why the system ended up this way), plus `todos.md` and
`fix-later.md`.

## Data

No participant data is in this repository. Questionnaire exports, session logs and
recordings live in the project's Box folder; the analysis scripts read them from there.
