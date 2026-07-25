---
title: "Behavioral Anomaly Detection for Access Logs"
subtitle: "System Report"
date: "2026-07-25"
---

# 1. Executive Summary

This project is an end-to-end insider-threat / access-log anomaly detection pipeline built and evaluated on a synthetic but behaviorally realistic dataset: **300 entities over a 45-day window, 48,008 total events, 1.37% of which are injected attack events** spanning six labeled attack patterns (brute force, credential stuffing, credential misuse, device spoofing, impossible travel, lateral movement).

The pipeline combines an **unsupervised LSTM autoencoder** (with a cold-start baseline fallback and rolling concept-drift recalibration) for per-event risk scoring with a **rule-based classifier** that turns the highest-risk events into typed, explainable alerts. On the most recent full pipeline run, risk scoring separates anomalous from normal events with **ROC-AUC = 0.8847**, and the rule engine classifies alerts within a top-1% risk budget (481 alerts) at an overall **alert-queue false-positive rate of 31.4%**, while sweeping only **0.32% of all normal traffic** into that queue. A concept-drift stress test confirms the system does not mistake gradual, legitimate behavior growth for an attack (91.9% of a drifting entity's events correctly scored as normal), and a real-time replay demo shows per-event scoring is fast enough (0.65 ms average, 8.8 ms p95) to sustain roughly 1,500+ events/sec on a single core — well within real-time budget for typical enterprise access-log volumes.

Beyond detection, the system includes an analyst-facing **Streamlit dashboard** (Alert Queue, Entity History, Metrics Summary, Attack Map) with three explainability layers: deterministic rule-based explanations, an **AI Triage Copilot** that drafts a live, LLM-generated incident summary per alert, and an **Anomaly Replay** view that frame-by-frame reconstructs the events leading up to a flagged alert.

*Note: the AI Triage Copilot is implemented with the Groq API (`llama-3.3-70b-versatile`), not Gemini — the integration was migrated from Gemini to Groq prior to this report, so this report describes the system as it currently runs.*

---

# 2. Problem Assumptions

Full detail lives in `data_gen/output/data_assumptions.md` (data generation) and `models/model_notes.md` (modeling). Summarized:

**Scale & reproducibility.** 300 entities over a 45-day window starting 2026-06-10, random seed fixed at 42 (numpy + Faker), so a run is reproducible byte-for-byte. This run produced 48,008 events: 47,349 normal and 659 injected attack events (1.37% of all events, actual injection rate 1.41%, drawn once uniformly from a 0.5%–3% band).

**Entity population.** Three entity types by weighted draw: `user` (70%), `service_account` (20%), `edge_device` (10%), each with its own baseline distributions for login hours, geo location, resource access set, session duration, and auth-method preference — drawn once per entity and held fixed for the run. A subset of each type (users 15%, service accounts 50%, edge devices 0%) is flagged privileged, which is the sole switch controlling whether an entity's sessions can carry a `command_sequence`.

**Attack injection.** A single attack rate is drawn once (this run: 1.41%), and incidents are generated round-robin across all six attack types until that event budget is met. Each type has its own deliberately isolated signal — e.g., impossible_travel only varies geography/timing (device, resources, auth stay baseline-consistent), device_spoofing only varies device fingerprint mid-session (time/location/resources stay normal) — so each rule can be evaluated against a clean, single-cause ground truth rather than a pattern with multiple overlapping tells.

**Feature engineering (models/).** Each event becomes a mix of embedded categoricals (`resource_accessed`, `auth_method`, `auth_result`, `protocol`, `os`, `geo_location`, `entity_type`) and engineered numerics (log session duration, cyclical hour/day-of-week encodings, log command-sequence length, log inter-event gap, and four "first time this entity has used X" novelty flags computed strictly forward-in-time to avoid leakage).

**Cold-start fallback.** Entities with fewer than 10 total events (or events among a longer-lived entity's first 14, before a full LSTM window exists) are scored against a per-entity-type heuristic baseline instead of the LSTM. In the latest run, 0 of 300 entities were cold-start at the entity level, but 4,200 of 48,008 events still fell to the baseline fallback (early-history events of otherwise LSTM-eligible entities).

---

# 3. System Architecture

The pipeline runs as four dependency-ordered stages (`run_pipeline.py`), followed by an interactive analyst layer:

**1. Data generator (`data_gen/`).** Synthesizes `access_logs.csv` and `ground_truth_labels.csv`: builds the 300-entity population with per-entity behavioral baselines, samples normal events from those baselines, and injects the six labeled attack patterns round-robin until the attack-event budget is met. Latest run: 48,008 events (47,349 normal, 659 anomalous — see the per-type breakdown in §4) written in a fully reproducible, seed-42 run.

**2. Baseline profiling & cold-start handling (`models/baseline.py`).** New or thin-history entities (fewer than 10 total events, or within their first 14 events) are scored against a per-entity-type heuristic profile — typical resource set, geo set, hour-of-day histogram, and auth failure rate — rather than the LSTM, since there isn't yet enough personal history to model individually. This same module also owns the rolling 7-day baseline/calibration refresh described in §5.

**3. LSTM sequence detection (`models/sequence_model.py`).** For entities past cold-start, a PyTorch encoder-decoder LSTM autoencoder consumes sliding 15-event windows (categorical fields embedded, concatenated with scaled numeric features), trained unsupervised to minimize reconstruction MSE. Only the last timestep's reconstruction error is kept as that event's raw anomaly score. Latest run: trained on 43,808 windows for 8 epochs, final training MSE 0.1622; 43,808 events scored via the LSTM path and 4,200 via the baseline fallback.

**4. Rule-based classification (`classification/`).** The raw scores (LSTM + fallback) are calibrated into a `risk_score` in (0,1) via a robust z-score + sigmoid, and only the top 1% of events by `risk_score` (481 alerts in the latest run) are evaluated against six rule functions — one per attack type (brute_force, impossible_travel, lateral_movement, credential_misuse, device_spoofing, credential_stuffing) — each checking a pattern-specific signal (e.g. auth-failure bursts, geo-velocity, resource-sequence escalation, off-hours+new-device, mid-session fingerprint change, or a cross-entity source-IP cluster). Alerts that clear the risk budget but match no rule stay `unclassified` (166 of 481 in the latest run) rather than being forced into a type.

**5. Incident-level grouping (`dashboard/app.py`).** A display-layer regrouping — consecutive alerts from the same `entity_id` within 30 minutes of each other are merged into one incident, with the incident's "winning" type being whichever named attack type has the highest average `risk_score` among its events. This turns 481 raw flagged events into 415 incidents in the Alert Queue view (see §4), without changing how any individual event was scored or classified.

**6. Explainability layer.** Three complementary layers, all fed by the same alert data:
   - **Rule-based explanations** — every alert carries a deterministic, human-readable string generated by the matching rule (e.g. *"credential_stuffing: a cluster of 3 related source_ips ... made 34 auth attempts against 34 distinct entity_ids within 30 min (85% failed)"*), so every classified alert is explainable without any model in the loop.
   - **AI Triage Copilot** (`dashboard/ai_triage.py`) — a live call to the Groq API (`llama-3.3-70b-versatile`) that drafts a SOC-analyst-style plain-English summary, a recommended response action, and a confidence level for whichever alert the analyst has selected, generated fresh on every selection.
   - **Anomaly Replay** (Entity History view) — a frame-by-frame reconstruction of the last 12 events leading up to a flagged alert for that entity, each frame annotated with how it deviates from the entity's established pre-window baseline (new resource, unusual hour, new source IP).

**7. Dashboard (`dashboard/app.py`).** A Streamlit analyst UI over the pipeline's output, with four main views plus the AI layers above:
   - **Alert Queue** — the incident-grouped, filterable, risk-ranked alert list, with the AI Triage Copilot panel.
   - **Entity History** — one entity's recent event timeline plus the Anomaly Replay expander.
   - **Metrics Summary** — the evaluation numbers in §4, rendered live from the pipeline's output.
   - **Attack Map** — an interactive orthographic globe plotting every classified alert at its `geo_location`, colored by `anomaly_type` (consistent palette across every view), with `impossible_travel` incidents additionally drawn as great-circle arcs between the two locations involved, shaded by `risk_score`.

---

# 4. Evaluation Metrics

All numbers below are from the latest `python run_pipeline.py` run (seed 42; deterministic, so identical across runs of the same code).

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

**Alert classification** (top 1% of events by risk_score → 481 candidates evaluated by the rule engine):

| anomaly_type | n_predicted | true_positives | false_positives | n_true_in_dataset | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| brute_force | 19 | 19 | 0 | 195 | 1.000 | 0.097 | 0.178 |
| impossible_travel | 20 | 7 | 13 | 11 | 0.350 | 0.636 | 0.452 |
| lateral_movement | 23 | 23 | 0 | 82 | 1.000 | 0.280 | 0.438 |
| credential_misuse | 31 | 19 | 12 | 24 | 0.613 | 0.792 | 0.691 |
| device_spoofing | 13 | 10 | 3 | 27 | 0.769 | 0.370 | 0.500 |
| credential_stuffing | 209 | 209 | 0 | 320 | 1.000 | 0.653 | 0.790 |

Unclassified alerts (flagged by `risk_score`, no rule matched): **166 / 481**.

**Alert-queue false positive rate:**
- **151 / 481 alerts (31.4%)** are false alarms — the fraction of the alert queue that isn't a true attack.
- **0.32%** of *all* normal events in the dataset (the stricter statistical FPR = FP / (FP + TN)) — i.e. the alert budget stays tight even though nearly a third of what lands in it isn't a true attack.

**Incident-level grouping** (Alert Queue's default view — see §3, stage 5): the 481 raw flagged events group into **415 incidents** (a 13.7% reduction), with 27 multi-event incidents absorbing 93 of the 481 raw events into single, chronologically-ordered incident cards. The reduction is concentrated in attack types that naturally burst multiple events against one entity in a short window — e.g. brute_force's 19 alerts collapse to 9 incidents and lateral_movement's 23 alerts collapse to 9 incidents — while credential_stuffing barely merges (209 alerts → 209 incidents), since by design each credential_stuffing incident hits a different entity exactly once (see §6's caveat on this rule).

**Supporting explainability, referenced from the dashboard views above:** the Alert Queue's AI Triage Copilot (Groq, `llama-3.3-70b-versatile`) and Entity History's Anomaly Replay both draw on this same `classified_alerts.csv` output — the Copilot summarizes one selected alert live, and the Replay view reconstructs the 12 events leading up to it. The Attack Map plots all 481 alerts geographically; a saved static view of the risk-score separation by label is at `models/score_separation.png`.

---

# 5. Concept Drift & Cold-Start Handling

**Rolling baseline refresh.** Both the per-entity-type baseline profile (§2/§3 stage 2) and the reference distribution used to calibrate raw scores into `risk_score` are recomputed every 7 simulated days, using only the previous 14 days of data — never the events about to be scored. This means "what counts as normal" drifts forward with the data instead of staying frozen at week one: if legitimate behavior gradually shifts (a resource migration, a new typical login window), the baseline and calibration reference both catch up within about two weeks instead of the pipeline slowly accumulating false positives against a stale definition of normal. The LSTM's *weights* are trained once, offline, on the full history — only the interpretation statistics (and the fallback baseline) are kept rolling; a production version would likely fine-tune the network periodically too.

**Cold-start fallback.** Entities with fewer than 10 total events, or within their first 14 events before a full 15-event LSTM window exists, are scored against the per-entity-type heuristic baseline described in §2 rather than the LSTM — an unavoidable design choice for entities the system hasn't seen enough of yet to model individually.

**Stress test result (`demo_insider_drift.py`).** A synthetic entity (`USR-DRIFT01`) was constructed with identical login hours, device, location, and auth method every single day for the full simulation — every one of its 74 events is ground-truth `normal` — but whose resource footprint grows gradually week over week (starting at 3 resources, ending at 11, by weeks 0–6), the way a real employee's responsibilities might expand. Against the alert-budget threshold computed across the full 48,082-event population, **68 of the entity's 74 events (91.9%, ≈92%) were correctly left out of the alert queue**. The 6 events that were flagged were every one of them the entity's *first-ever* access to a resource it had just gained (e.g. `file-share-general`, `source-code-repo`, `internal-wiki-admin`), landed as `unclassified` (matched no attack rule), and are an honest, expected edge case — genuine first-time novelty with no prior history to call typical — rather than a rule malfunction or a sign the system conflates gradual legitimate growth with an attack.

---

# 6. System Design & Scalability (Real-Time Streaming Feasibility)

**Architecture: lambda-style separation.** The system is architected with the same inference/training split as a classic lambda architecture:
- **Fast path (per-event inference)** — an already-trained, frozen LSTM plus the current rolling baseline/calibration statistics score and classify one arriving event at a time. This is the path `demo_streaming.py` measures below, and it does no retraining or batch recomputation per event.
- **Slow path (periodic recalibration)** — the 7-day rolling baseline/calibration refresh (§5) and, in a production deployment, periodic LSTM fine-tuning, run as a separate scheduled job decoupled from the request path, exactly the way a lambda architecture separates its speed layer from its batch layer.

**Real-time replay result (`demo_streaming.py`, default args: busiest alert day, 0.08s simulated arrival gap).** Replaying 1,171 events from the single busiest day in the dataset (2026-07-22) one at a time through the exact same rule engine used in batch, timing only the scoring + classification step (excluding the artificial arrival delay):
- **Average latency: 0.652 ms/event**
- **p95 latency: 8.836 ms/event**
- **Max latency: 17.707 ms/event**
- 70 alerts raised (6.0% of the 1,171 replayed events)
- Extrapolated sustained throughput: **≈1,533 events/sec on a single core**, if events arrived back to back at the average processing rate

This is well above typical enterprise access-log volumes, supporting real-time deployment feasibility on the fast path alone.

**credential_stuffing cross-entity caveat.** Unlike the other five rules — which each evaluate one entity's own history — credential_stuffing is deliberately a population-level rule: it looks for a cluster of 2–6 related source IPs making a burst of auth attempts against many distinct entities within a 30-minute window (`CREDENTIAL_STUFFING_WINDOW_MINUTES`), because that cross-entity breadth *is* the signal (a real credential-stuffing run tries each leaked credential once across many accounts, not many times against one). This means the rule is not strict O(1) per event; it scans a time-windowed slice of the population's recent auth attempts rather than only the arriving event's own entity history, so its per-event cost scales with how many auth attempts landed in the preceding 30 minutes across the whole entity population, not with a constant. In practice this stayed well within the latency numbers above (the p95/max figures already include every rule, credential_stuffing included, since it accounted for the largest single alert type — 209 of 481 — in this dataset), but it's the one rule whose cost profile would need explicit attention (e.g. a bounded, indexed sliding window rather than a full population scan) at much higher steady-state event rates.

---

# 7. Known Limitations

- **Recall is structurally capped by the top-1% alert budget.** Every rule only ever evaluates events that already cleared the top-1%-by-risk_score threshold (481 of 48,008 events); a true attack event that doesn't score in the top 1% is invisible to every rule regardless of how well that rule is written. This is the direct cause of low recall for otherwise perfect-precision rules like brute_force (precision 1.000, recall only 0.097 — 176 of 195 true brute_force events never entered the candidate pool at all) and lateral_movement (precision 1.000, recall 0.280).
- **Low-and-slow exfiltration is not implemented.** All six attack patterns in this system are relatively fast-acting (minutes to a single session); a slow, low-volume exfiltration pattern spread across weeks — deliberately staying under any single-window anomaly threshold — has no corresponding generator pattern, detection rule, or evaluation coverage in the current system. See §8.1 for a proposed approach.
- **Synthetic data may not capture real-world noise.** The dataset is generated from explicit, individually clean distributional assumptions (§2) with each attack pattern isolating a single detection signal; real access logs have messier, more correlated legitimate behavior and attacks that don't cleanly isolate one signal, so real-world precision/recall on these same rules would likely differ from the numbers in §4.
- **impossible_travel precision (0.350) is affected by its by-design neighboring-event check.** The rule flags geo-velocity against the *immediately preceding* event for that entity; 13 of its 20 predicted alerts were false positives, most plausibly because gaps between an entity's genuinely separate, legitimate sessions (e.g. across a multi-day span) can still exceed the 900 km/h plausibility ceiling when compared as a pair, even though neither underlying session is itself anomalous.
- **Incident-level grouping was previously a limitation — now addressed.** Before this feature existed, the Alert Queue showed all 481 individually flagged events as separate rows, so a single multi-event attack burst (e.g. a 6-step lateral_movement chain) appeared as 6 separate line items an analyst had to manually recognize as one incident. The grouping logic in §3/§4 now merges consecutive same-entity alerts within a 30-minute window into one incident card, cutting the queue from 481 events to 415 incidents and reducing the most fragmented attack types (brute_force: 19→9, lateral_movement: 23→9) to something closer to "one incident per real event," while leaving genuinely distinct alerts (like credential_stuffing's one-entity-one-event pattern) unmerged.

---

# 8. Future Enhancements

**8.1 Low-and-slow exfiltration detection via rolling-aggregate anomaly scoring.** Detect exfiltration that deliberately stays under any single-event or single-window threshold by tracking trailing 7/14/30-day resource-access-*count* per entity (not just per-event novelty) and scoring deviation from that entity's own historical rolling-count distribution. This matters because the current system's LSTM window (15 events) and rule windows (minutes) are both far too short to see a slow drip of access spread across weeks. Implementation would add a new feature stream — per-entity rolling counts of accesses to sensitive (tier-0/tier-1) resources, computed at 7/14/30-day trailing windows — feeding either a new rule (z-score deviation from the entity's own trailing baseline) or an additional input to the existing risk-scoring path.

**8.2 SHAP-based feature attribution as a deeper explainability layer.** Layer SHAP (or a comparable attribution method) on top of the LSTM autoencoder's reconstruction error to show *which specific input features* (which categorical embedding, which numeric feature) drove a given event's anomaly score, complementing the current rule-based "why" strings which only explain rule-matched alerts, not the raw risk_score itself or the 166 currently-unclassified alerts. This would matter most for exactly those unclassified alerts — where risk_score is high but no rule fired — giving an analyst a first, model-native explanation instead of nothing. Implementation-wise, this requires wrapping the trained PyTorch autoencoder's reconstruction loss as a scalar output per timestep and running KernelSHAP or DeepSHAP against the embedded+numeric feature vector, likely restricted to the alert-budget subset for tractability.

**8.3 Adaptive/learned concept drift detection replacing the fixed 7-day window.** Replace the current fixed-cadence, fixed-lookback rolling baseline (§5) with either formal change-point detection (e.g. CUSUM or Bayesian online change-point detection on the per-entity-type score distributions) or an exponentially-weighted moving baseline that adapts continuously rather than stepping every 7 days. This matters because a fixed 7-day cadence can be both too slow (a sudden legitimate change, like an org-wide tool migration, takes up to a week to be absorbed) and arbitrary (there's no evidence 7 days is the right constant for every kind of drift). Implementation would swap `models/baseline.py`'s recomputation trigger from a hardcoded day-count to a statistical trigger — e.g., recalibrate when a monitored KL-divergence or CUSUM statistic between the live score distribution and the reference distribution crosses a threshold — while keeping the same underlying baseline/calibration math.

**8.4 Graph-based entity-resource relationship modeling to catch multi-entity coordinated attacks.** Model entities and resources as nodes in a bipartite (or richer, typed) graph and use a GNN or graph embedding method (e.g. GraphSAGE, node2vec) to learn representations that capture *relational* anomalies — coordinated access patterns across multiple entities that no single-entity rule or LSTM window can see. This matters because every current detection path (LSTM windows, all six rules except credential_stuffing) is fundamentally per-entity; a slow, distributed multi-account campaign designed to keep each individual entity's behavior looking normal would be invisible today. Implementation would build a periodically-updated access graph from `access_logs.csv`, train embeddings offline, and add graph-embedding distance/anomaly-score as a new feature into either the existing risk-scoring pipeline or a dedicated coordinated-attack rule.

**8.5 Production streaming deployment via a Kafka-style message queue.** Replace `demo_streaming.py`'s in-process replay loop with a real message queue (Kafka, Kinesis, or similar) feeding the existing per-event scoring/classification logic essentially unchanged, with LSTM retraining and baseline recalibration running as separate scheduled jobs (batch or cron-triggered) that publish updated model weights/baseline statistics for the streaming consumers to pick up. This directly operationalizes the lambda-style split already described in §6 — the fast path in this codebase is already stateless enough (loads a frozen model + current baseline stats) to drop behind a queue consumer with minimal change; the main new work is the consumer harness, exactly-once/at-least-once delivery handling, and a model/baseline hot-reload mechanism.

**8.6 Active learning loop from SOC analyst feedback.** Let analysts mark alerts in the dashboard as true/false positive, and feed that feedback back into (a) the six rules' tunable thresholds in `classification/config.py` (e.g. auto-tightening `CREDENTIAL_MISUSE_OFF_HOURS_MAX_PROB` if credential_misuse's false positives cluster around a specific hour range) and, eventually, (b) the LSTM's risk_score-to-alert threshold itself. This matters because the current 31.4% alert-queue false-positive rate is a static property of hand-tuned thresholds; a feedback loop would let the system's real-world precision improve over time without a full retraining cycle. Implementation: a lightweight feedback table keyed by `event_id`/`entity_id`, a periodic job that recomputes precision/recall per rule against accumulated feedback, and a bounded, human-reviewed threshold-adjustment step (not fully automatic, to avoid a feedback loop silently drifting thresholds based on biased or sparse analyst input).

**8.7 Multi-modal fusion with DNS logs, endpoint telemetry, and HR system data.** Extend the feature set beyond access logs alone to incorporate DNS query logs (e.g. beaconing/C2 domain patterns), endpoint telemetry (process execution, EDR alerts), and HR system data (role changes, termination dates, leave status) as additional context per entity and per event. This matters because access logs alone can't distinguish "an employee whose role just changed and is exploring new resources" from "a compromised account exploring resources it shouldn't" — HR context directly disambiguates exactly the kind of legitimate-growth case stress-tested in §5. Implementation would require a per-entity feature join across these additional sources (most requiring their own ingestion/normalization pipeline), added as extra engineered features alongside the current access-log-derived ones in `models/data_prep.py`.

**8.8 Federated / privacy-preserving training across multiple organizations.** Train shared model weights (or shared baseline statistics) across multiple organizations' access logs without centralizing the underlying sensitive log data, using federated learning (local training + secure aggregation of gradients/weights) or differential-privacy techniques on any shared statistics. This matters because insider-threat and credential-attack patterns generalize across organizations, but the underlying access logs themselves are exactly the kind of sensitive data no organization can share directly — federation is the standard way to get cross-org model benefit without a data-sharing agreement. Implementation would involve substantial architecture change: a federated training orchestrator (e.g. Flower or a custom parameter-server setup), per-org local training loops reusing the existing `models/sequence_model.py` architecture, and a secure aggregation step — a meaningfully larger undertaking than the other items in this roadmap, appropriate as a longer-term direction rather than a near-term addition.
