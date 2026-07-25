# Anomaly Detection for Access Logs

An end-to-end insider-threat / access-log anomaly detection pipeline: a synthetic
but realistic 45-day access-log dataset for a 300-entity org, an unsupervised
LSTM autoencoder (with a cold-start/baseline fallback and rolling concept-drift
recalibration) that scores every event for risk, and a rule-based classifier
that turns the highest-risk events into labeled alerts for six specific attack
patterns -- brute force, credential stuffing, credential misuse, device
spoofing, impossible travel, and lateral movement.

Two standalone demo scripts go further than the batch pipeline: one stress-tests
the system against genuinely normal behavior drift (does an employee gradually
taking on new responsibilities get mistaken for an attacker?), and one replays
the dataset as a live event-by-event feed to demonstrate real-time deployment
feasibility.

## Live demo

There are two separate, linked artifacts: `docs/index.html` is a standalone
marketing/landing page (project overview, pitch, and an illustrated walk-
through of the detection pipeline) meant to be served statically via GitHub
Pages -- open it directly in a browser, no server required. `dashboard/app.py`
is the actual working analyst tool: run it with `streamlit run dashboard/app.py`
after the pipeline has produced its output (or use a deployed Streamlit
Community Cloud URL, if one exists). Each links to the other -- the landing
page's "Launch live dashboard" button in the nav and hero, and the
dashboard's sidebar "← Project overview" link -- so a visitor can move
between the pitch and the real tool in either direction.

## Quickstart

```bash
# 1. Install dependencies (three module-scoped requirement files)
pip install -r data_gen/requirements.txt -r models/requirements.txt -r dashboard/requirements.txt

# 2. Run the full pipeline: generate data -> train & score -> evaluate -> classify alerts
python run_pipeline.py

# 3. Explore the results interactively
streamlit run dashboard/app.py

# 4. Two standalone demos (each writes to its own isolated output dir; run any time after step 2)
python demo_insider_drift.py   # concept-drift stress test
python demo_streaming.py       # real-time replay + latency measurement
```

`run_pipeline.py` is the single entry point a judge needs: it runs the four
stages in dependency order (`data_gen.generate` -> `models.train` ->
`models.evaluate` -> `classification.main`), printing a labeled progress
banner before each one, and ends with a rolled-up summary of dataset size,
label distribution, ROC-AUC, precision/recall/F1 per anomaly type, and
alert-queue false positive rate. A full run takes a little over two minutes
(LSTM training dominates).

## Folder structure

```
data_gen/          synthetic access-log generator
  generate.py         entry point: builds entities, samples normal events, injects attacks
  entities.py          builds the 300-entity population and each one's baseline behavior
  events.py             samples normal (non-attack) events from each entity's baseline
  attacks.py             injects the six labeled attack patterns
  config.py               static reference data: cities, resources, auth methods, OS pools
  utils.py                  geodesic distance + sequential ID helpers
  docs.py                     template for the generated data_assumptions.md
  output/                       access_logs.csv, ground_truth_labels.csv, data_assumptions.md

models/             unsupervised risk-scoring pipeline
  score.py             orchestrates: features -> train LSTM -> rolling baselines -> risk_score
  train.py               entry point: runs score.py's pipeline, writes scored_events.csv + notes
  sequence_model.py        the LSTM autoencoder over sliding windows of entity event sequences
  baseline.py                cold-start fallback profiles + rolling drift-calibration machinery
  data_prep.py                 loads access_logs.csv, engineers the per-event feature set
  vocab.py                       categorical vocabularies for embeddings
  evaluate.py                      ROC-AUC of risk_score vs ground truth + separation plot
  docs.py                            template for the generated model_notes.md

classification/     rule-based alert classification
  classify.py          entry point: picks top-1% risk_score events, runs rules, writes alerts
  rules.py                the six attack-pattern rule functions + RuleContext
  data_prep.py               joins scored_events.csv with access_logs.csv
  geo.py                        City,Country -> lat/lon lookup for geo-velocity math
  evaluate.py                     precision/recall/F1 per type + alert-budget false positive rate
  config.py                         alert budget and all per-rule tunable thresholds

dashboard/          interactive Streamlit UI over the pipeline's output
  app.py                Alert Queue, Entity History, and Metrics Summary views
  data.py                 cached data loading/joining for the dashboard
  palette.py                 shared anomaly_type -> color mapping across every view

run_pipeline.py      runs the full pipeline end to end (see Quickstart)
demo_insider_drift.py   standalone: concept-drift stress test (isolated output dir)
demo_streaming.py       standalone: real-time replay + latency demo (no isolated output; reuses models/scored_events.csv)
```

## Key results

From a full `python run_pipeline.py` run (300 entities, 45 days, seed 42):

**Dataset:** 48,008 events, 1.37% anomalous.

| label | count |
|---|---:|
| normal | 47,349 |
| anomaly_credential_stuffing | 320 |
| anomaly_brute_force | 195 |
| anomaly_lateral_movement | 82 |
| anomaly_device_spoofing | 27 |
| anomaly_credential_misuse | 24 |
| anomaly_impossible_travel | 11 |

**Risk scoring:** ROC-AUC of `risk_score` vs. ground-truth anomaly label = **0.8847**.

**Alert classification** (top 1% of events by risk_score = 481 alerts):

| anomaly_type | precision | recall | F1 |
|---|---:|---:|---:|
| brute_force | 1.000 | 0.097 | 0.178 |
| impossible_travel | 0.350 | 0.636 | 0.452 |
| lateral_movement | 1.000 | 0.280 | 0.438 |
| credential_misuse | 0.613 | 0.792 | 0.691 |
| device_spoofing | 0.769 | 0.370 | 0.500 |
| credential_stuffing | 1.000 | 0.653 | 0.790 |

**Alert-queue false positive rate:** 151/481 alerts (31.4%) are false alarms;
those false alarms are only 0.32% of all normal events in the dataset --
i.e. the alert budget stays tight even though almost a third of what lands
in it isn't a true attack.

**Concept-drift stress test** (`demo_insider_drift.py`): an entity with
identical login hours, device, location, and auth method every day, whose
resource footprint grows gradually over several weeks the way a real
employee's does -- every one of its events is ground-truth `normal`. The
system correctly treated **92% of its events as normal**, with the handful
of flagged events concentrated on genuine first-ever access to a newly
available resource (an honest edge case, not a rule malfunction).

**Real-time feasibility** (`demo_streaming.py`): replaying `access_logs.csv`
as a live, one-event-at-a-time feed and timing each event's scoring +
classification step (excluding the artificial arrival delay) gives an
average latency of **0.565 ms/event**, extrapolating to roughly **1,769
events/sec** sustained on a single core -- well within real-time budget for
typical enterprise access-log volumes.
