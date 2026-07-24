"""
demo_insider_drift.py

Concept-drift stress test, distinct from cold-start: simulates ONE entity
whose behavior is genuinely normal throughout -- same login hours, same
device, same location, same auth method, every day -- but whose resource
footprint gradually grows over several weeks, 1-2 new resources at a time,
the way a real employee's does as they take on new responsibilities. Every
one of its events is labeled `normal` in ground truth; nothing is injected
as an attack.

This entity is run through the SAME code as the rest of this project:
data_gen's own event-generation conventions, models/ for LSTM + baseline
risk scoring (including the rolling 7-day drift-handling baseline refresh),
and classification/ for the rule-based classifiers -- not a re-implementation
or a mock. The point is to see, empirically, whether gradual and genuinely
normal behavior evolution gets swept up in the same net as an attack.

It writes its own isolated dataset and pipeline outputs under
demo_insider_drift_output/ and never touches the canonical data_gen/output/,
models/, or classification/ artifacts.

Usage:
    python demo_insider_drift.py
"""

import datetime as dt
import json
import os

import numpy as np
import pandas as pd
from faker import Faker

from classification import classify as clf_classify
from classification import config as clf_cfg
from classification import data_prep as clf_data_prep
from data_gen import config as dg_cfg
from data_gen.generate import COLUMN_ORDER, build_dataset as build_base_dataset
from models import config as models_cfg
from models import score as models_score
from models.train import OUTPUT_COLUMNS as SCORED_OUTPUT_COLUMNS

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_insider_drift_output")
ACCESS_LOGS_PATH = os.path.join(OUT_DIR, "access_logs.csv")
GROUND_TRUTH_PATH = os.path.join(OUT_DIR, "ground_truth_labels.csv")
SCORED_EVENTS_PATH = os.path.join(OUT_DIR, "scored_events.csv")
CLASSIFIED_ALERTS_PATH = os.path.join(OUT_DIR, "classified_alerts.csv")

DRIFT_ENTITY_ID = "USR-DRIFT01"
DRIFT_SEED = 4207  # distinct from data_gen's own seed, so this run doesn't perturb the base population
N_STARTER_RESOURCES = 3
WEEKLY_NEW_RESOURCES_MIN, WEEKLY_NEW_RESOURCES_MAX = 1, 3  # rng.integers is exclusive at the top -> 1-2 per week

fake = Faker()
Faker.seed(DRIFT_SEED)


# ---------------------------------------------------------------------------
# 1. Simulate the drift entity
# ---------------------------------------------------------------------------
def _protocol_for(resource, rng):
    tier = dg_cfg.RESOURCE_TIER_LOOKUP[resource]
    options = dg_cfg.PROTOCOL_FOR_TIER[tier]
    return options[rng.integers(0, len(options))]


def _fingerprint_json(os_name, mac, resource, rng):
    return json.dumps({"os": os_name, "mac_address": mac, "protocol": _protocol_for(resource, rng)})


def _build_growth_schedule(rng, n_days):
    """Which resources this entity can reach, week by week. Week 0 is the
    starting baseline; from week 1 on, 1-2 new resources become available
    each week -- tier 1/2 only (no tier-0 admin systems), since "taking on
    new responsibilities" is not the same story as "suddenly gaining admin
    access" (that's closer to lateral_movement, a different, genuinely
    suspicious pattern this demo is not testing)."""
    starter_pool = list(dg_cfg.RESOURCES_BY_TIER[2])
    starter_idx = rng.choice(len(starter_pool), size=N_STARTER_RESOURCES, replace=False)
    resource_pool = [starter_pool[i] for i in starter_idx]

    growth_candidates = [r for r in (dg_cfg.RESOURCES_BY_TIER[1] + dg_cfg.RESOURCES_BY_TIER[2])
                          if r not in resource_pool]
    perm = rng.permutation(len(growth_candidates))
    growth_candidates = [growth_candidates[i] for i in perm]

    n_weeks = n_days // 7 + 1
    pool_by_week = {0: list(resource_pool)}
    introduced_day = {r: 0 for r in resource_pool}
    timeline = [{"week": 0, "added": [], "cumulative": list(resource_pool)}]

    cursor = 0
    for week in range(1, n_weeks):
        n_new = int(rng.integers(WEEKLY_NEW_RESOURCES_MIN, WEEKLY_NEW_RESOURCES_MAX))
        added = growth_candidates[cursor: cursor + n_new]
        cursor += len(added)
        resource_pool = resource_pool + added
        for r in added:
            introduced_day[r] = week * 7
        pool_by_week[week] = list(resource_pool)
        timeline.append({"week": week, "added": added, "cumulative": list(resource_pool)})

    return pool_by_week, introduced_day, timeline


def simulate_drift_entity(n_days=dg_cfg.N_DAYS, start_date=dg_cfg.START_DATE):
    rng = np.random.default_rng(DRIFT_SEED)

    home_city = dg_cfg.CITIES[rng.integers(0, len(dg_cfg.CITIES))]
    typical_hour_center = float(rng.uniform(9, 16))   # ordinary daytime worker
    typical_hour_spread = float(rng.uniform(1.5, 2.5))
    active_prob = float(rng.uniform(0.55, 0.75))
    events_per_day_lambda = float(rng.uniform(2.0, 3.5))
    session_mean = float(rng.uniform(10, 30))
    session_std = float(rng.uniform(4, 8))

    auth_w = dg_cfg.AUTH_METHOD_WEIGHTS["user"]
    auth_methods = list(auth_w.keys())
    auth_weights = [auth_w[m] for m in auth_methods]
    auth_method = auth_methods[int(np.argmax(auth_weights))]  # ONE fixed auth method throughout -- see module docstring

    home_ip = fake.ipv4_private()
    os_choice = dg_cfg.OS_POOL["user"][rng.integers(0, len(dg_cfg.OS_POOL["user"]))]
    mac = fake.mac_address()

    pool_by_week, introduced_day, timeline = _build_growth_schedule(rng, n_days)

    start = dt.datetime.fromisoformat(start_date)
    events = []
    for day_offset in range(n_days):
        week = day_offset // 7
        pool = pool_by_week[week]
        if rng.random() > active_prob:
            continue
        n_events_today = rng.poisson(events_per_day_lambda)
        if n_events_today <= 0:
            continue

        day_date = start + dt.timedelta(days=day_offset)
        for _ in range(n_events_today):
            hour = rng.normal(typical_hour_center, typical_hour_spread) % 24
            minute, second = int(rng.integers(0, 60)), int(rng.integers(0, 60))
            timestamp = day_date + dt.timedelta(hours=hour, minutes=minute, seconds=second)

            resource = pool[rng.integers(0, len(pool))]
            is_failure = rng.random() < 0.02  # same ordinary-typo noise rate as the rest of the dataset
            if is_failure:
                auth_result, session_duration = "failure", 0.0
            else:
                auth_result = "success"
                session_duration = round(float(max(0.05, rng.normal(session_mean, session_std))), 2)

            events.append({
                "entity_id": DRIFT_ENTITY_ID,
                "entity_type": "user",
                "timestamp": timestamp.isoformat(),
                "source_ip": home_ip,
                "geo_location": f"{home_city[0]}, {home_city[1]}",
                "resource_accessed": resource,
                "auth_method": auth_method,
                "auth_result": auth_result,
                "session_duration": session_duration,
                "command_sequence": "[]",
                "device_fingerprint": _fingerprint_json(os_choice, mac, resource, rng),
                "label": "normal",
                "_week": week,
                "_resource_introduced_day": introduced_day[resource],
                "_day_offset": day_offset,
            })

    return events, timeline


# ---------------------------------------------------------------------------
# 2. Assemble the isolated demo dataset (base population + drift entity)
# ---------------------------------------------------------------------------
def build_combined_dataset():
    print("Building the base 300-entity population (same generator, same seed as the canonical dataset) ...")
    base_df, base_stats = build_base_dataset()  # data_gen defaults: seed=42, n_entities=300, n_days=45

    print("Simulating the insider-drift entity ...")
    drift_events, timeline = simulate_drift_entity()
    drift_df = pd.DataFrame(drift_events)

    combined = pd.concat([base_df, drift_df], ignore_index=True, sort=False)
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    combined["event_id"] = [f"EVT-{i:07d}" for i in range(1, len(combined) + 1)]

    meta_cols = ["event_id", "_week", "_resource_introduced_day", "_day_offset"]
    drift_meta = combined.loc[combined["entity_id"] == DRIFT_ENTITY_ID, meta_cols].copy()

    access_logs = combined[COLUMN_ORDER].drop(columns=["label"])
    ground_truth = combined[["entity_id", "event_id", "label"]]

    os.makedirs(OUT_DIR, exist_ok=True)
    access_logs.to_csv(ACCESS_LOGS_PATH, index=False)
    ground_truth.to_csv(GROUND_TRUTH_PATH, index=False)

    print(f"  {len(base_df)} base events + {len(drift_df)} drift-entity events "
          f"= {len(combined)} total -> {ACCESS_LOGS_PATH}")
    return timeline, drift_meta


# ---------------------------------------------------------------------------
# 3. Run the real scoring + classification pipeline
# ---------------------------------------------------------------------------
def run_pipeline():
    print("\nRunning models/ scoring pipeline (LSTM autoencoder + baseline fallback + rolling drift calibration) ...")
    scored_df, model_stats = models_score.run_pipeline(verbose=True, access_logs_path=ACCESS_LOGS_PATH)
    scored_df[SCORED_OUTPUT_COLUMNS].sort_values("timestamp").to_csv(SCORED_EVENTS_PATH, index=False)

    print("\nRunning classification/ rule-based pipeline ...")
    classified_df = clf_classify.run(
        verbose=True,
        scored_events_path=SCORED_EVENTS_PATH,
        access_logs_path=ACCESS_LOGS_PATH,
        output_path=CLASSIFIED_ALERTS_PATH,
    )

    merged = clf_data_prep.load_merged(scored_events_path=SCORED_EVENTS_PATH, access_logs_path=ACCESS_LOGS_PATH)
    _, threshold = clf_classify.select_candidates(merged)

    return scored_df, classified_df, threshold


# ---------------------------------------------------------------------------
# 4. Report
# ---------------------------------------------------------------------------
def print_timeline(timeline):
    print("\n=== Simulated resource-footprint growth (the entity's own weekly story) ===")
    print(f"Entity {DRIFT_ENTITY_ID}: same login hours, same device, same location, same auth method "
          f"every single day. Only its resource footprint changes.")
    for week_info in timeline:
        week, added, cumulative = week_info["week"], week_info["added"], week_info["cumulative"]
        if week == 0:
            print(f"  Week 0: starts with {len(cumulative)} resources -- {', '.join(cumulative)}")
        elif added:
            print(f"  Week {week}: +{len(added)} new -- {', '.join(added)}  "
                  f"(total now {len(cumulative)})")
        else:
            print(f"  Week {week}: no change (total still {len(cumulative)})")


def print_report(scored_df, classified_df, threshold, drift_meta):
    entity_scored = scored_df[scored_df["entity_id"] == DRIFT_ENTITY_ID].sort_values("timestamp")
    entity_alerts = classified_df[classified_df["entity_id"] == DRIFT_ENTITY_ID]

    print(f"\n=== risk_score distribution for {DRIFT_ENTITY_ID} ({len(entity_scored)} events) ===")
    rs = entity_scored["risk_score"]
    print(f"  min={rs.min():.4f}  median={rs.median():.4f}  mean={rs.mean():.4f}  max={rs.max():.4f}")
    print(f"  Alert-budget threshold (top {clf_cfg.ALERT_BUDGET_FRACTION:.0%} of ALL {len(scored_df):,} events, "
          f"across every entity): risk_score >= {threshold:.4f}")
    n_over_threshold = int((rs >= threshold).sum())
    print(f"  {n_over_threshold} of this entity's {len(entity_scored)} events sit at/above that threshold.")

    print(f"\n=== Verdict: {len(entity_alerts)} of {len(entity_scored)} events were flagged in the alert queue ===")
    if len(entity_alerts) == 0:
        print(f"  PASS -- {DRIFT_ENTITY_ID} was never flagged. The system correctly treated its gradual, "
              f"weekly resource-footprint growth as normal, distinct from a cold-start entity (it always had "
              f">= {models_cfg.MIN_HISTORY_EVENTS} events of history) and distinct from lateral_movement "
              f"(new resources appeared a week apart, never {clf_cfg.LATERAL_MOVEMENT_LOOKBACK_MINUTES} minutes "
              f"apart in a tight escalating chain).")
        return

    print(f"  {len(entity_alerts)} event(s) were flagged. Explaining each one:\n")
    meta_by_event = drift_meta.set_index("event_id")
    resource_novelty = entity_scored.set_index("event_id")["resource_novelty"] \
        if "resource_novelty" in entity_scored.columns else None

    # How many times this entity had used this exact resource before, as of each
    # event -- computed from its own real access history, not asserted.
    entity_by_time = entity_scored.sort_values("timestamp").copy()
    entity_by_time["use_rank"] = entity_by_time.groupby("resource_accessed").cumcount() + 1
    use_rank_by_event = entity_by_time.set_index("event_id")["use_rank"]

    for row in entity_alerts.sort_values("risk_score", ascending=False).itertuples():
        meta = meta_by_event.loc[row.event_id]
        scored_row = entity_scored[entity_scored["event_id"] == row.event_id].iloc[0]
        use_rank = int(use_rank_by_event.loc[row.event_id])
        days_available_before_use = (
            pd.Timestamp(scored_row["timestamp"]).normalize()
            - (pd.Timestamp(dg_cfg.START_DATE) + pd.Timedelta(days=int(meta["_resource_introduced_day"])))
        ).days

        print(f"  - {row.event_id}  risk_score={row.risk_score:.4f}  anomaly_type={row.anomaly_type}")
        print(f"    rule engine says: {row.explanation}")
        print(f"    context: week {int(meta['_week'])} of the simulation; resource "
              f"'{scored_row['resource_accessed']}' -- this was access #{use_rank} to that resource for this "
              f"entity ({'its first-ever use' if use_rank == 1 else f'NOT the first use -- {use_rank - 1} prior access(es) already existed'}), "
              f"{days_available_before_use} day(s) after the resource became available to it"
              + (f"; resource_novelty feature = {resource_novelty.loc[row.event_id]:.0f}"
                 if resource_novelty is not None and row.event_id in resource_novelty.index else ""))
        if use_rank == 1:
            print(f"    likely explanation: this was this entity's genuine first-ever access to this resource "
                  f"-- there is no prior history for the model or the rules to have called it typical yet, so an "
                  f"elevated score here is an honest, expected false positive on a real edge case (first use of a "
                  f"newly-gained resource), not a rule malfunction.\n")
        else:
            print(f"    likely explanation: this was a REPEAT access, not a first use -- the elevated score isn't "
                  f"explained by resource novelty alone here; likely some other feature (timing, session length, "
                  f"or the epoch's rolling baseline calibration) contributed. Worth a closer look if this recurs.\n")

    print(f"  None of these were mislabeled as a specific attack type unless shown above -- "
          f"{(entity_alerts['anomaly_type'] == 'unclassified').sum()} of {len(entity_alerts)} landed as "
          f"'unclassified' (flagged for review, not auto-attributed to an attack pattern).")


def main():
    timeline, drift_meta = build_combined_dataset()
    print_timeline(timeline)
    scored_df, classified_df, threshold = run_pipeline()
    print_report(scored_df, classified_df, threshold, drift_meta)
    print(f"\nIsolated demo outputs written to {OUT_DIR}/ -- canonical data_gen/output/, models/, and "
          f"classification/ artifacts were not touched.")


if __name__ == "__main__":
    main()
