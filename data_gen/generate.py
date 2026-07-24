"""
Main entry point: builds the entity population, samples normal events,
injects the three attack patterns, and writes access_logs.csv,
ground_truth_labels.csv, and data_assumptions.md.

Usage:
    python -m data_gen.generate [--out-dir data_gen/output] [--seed 42]
"""

import argparse
import os

import numpy as np
import pandas as pd

from . import config as cfg
from . import docs
from .attacks import ATTACK_GENERATORS
from .entities import generate_entities
from .events import generate_normal_events

COLUMN_ORDER = [
    "event_id", "entity_id", "entity_type", "timestamp", "source_ip",
    "geo_location", "resource_accessed", "auth_method", "auth_result",
    "session_duration", "command_sequence", "device_fingerprint", "label",
]


def build_dataset(seed=cfg.RANDOM_SEED, n_entities=cfg.N_ENTITIES, n_days=cfg.N_DAYS,
                   start_date=cfg.START_DATE):
    rng = np.random.default_rng(seed)

    entities = generate_entities(n_entities=n_entities, rng=rng)
    normal_events = generate_normal_events(entities, n_days=n_days, start_date=start_date, rng=rng)
    n_normal = len(normal_events)

    attack_rate = float(rng.uniform(cfg.ATTACK_RATE_MIN, cfg.ATTACK_RATE_MAX))
    target_attack_events = max(1, int(round(n_normal * attack_rate)))

    attack_types = list(ATTACK_GENERATORS.keys())
    attack_events = []
    i = 0
    while len(attack_events) < target_attack_events:
        atype = attack_types[i % len(attack_types)]
        generator = ATTACK_GENERATORS[atype]
        attack_events.extend(generator(entities, rng, n_days=n_days, start_date=start_date))
        i += 1

    all_events = normal_events + attack_events
    all_events.sort(key=lambda e: e["timestamp"])
    for idx, e in enumerate(all_events, start=1):
        e["event_id"] = f"EVT-{idx:07d}"

    df = pd.DataFrame(all_events)[COLUMN_ORDER]

    # True label counts, not generator-output counts: some attack generators
    # (impossible_travel's legit precursor, device_spoofing's normal lead-in)
    # emit `label = "normal"` rows alongside their anomalous ones, so
    # len(attack_events) overcounts true anomalies.
    n_true_anomaly = int((df["label"] != "normal").sum())
    n_true_normal = len(df) - n_true_anomaly

    stats = {
        "n_entities": n_entities,
        "n_days": n_days,
        "start_date": start_date,
        "seed": seed,
        "total_events": len(df),
        "n_normal": n_true_normal,
        "n_anomaly": n_true_anomaly,
        "anomaly_pct": 100.0 * n_true_anomaly / len(df),
        "attack_rate": attack_rate,
    }
    return df, stats


def write_outputs(df, stats, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    access_logs = df.drop(columns=["label"])
    ground_truth = df[["entity_id", "event_id", "label"]]

    access_logs_path = os.path.join(out_dir, "access_logs.csv")
    ground_truth_path = os.path.join(out_dir, "ground_truth_labels.csv")
    assumptions_path = os.path.join(out_dir, "data_assumptions.md")

    access_logs.to_csv(access_logs_path, index=False)
    ground_truth.to_csv(ground_truth_path, index=False)
    with open(assumptions_path, "w", encoding="utf-8") as f:
        f.write(docs.render(**stats))

    return access_logs_path, ground_truth_path, assumptions_path


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic access-log dataset.")
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "output"))
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_SEED)
    parser.add_argument("--n-entities", type=int, default=cfg.N_ENTITIES)
    parser.add_argument("--n-days", type=int, default=cfg.N_DAYS)
    parser.add_argument("--start-date", default=cfg.START_DATE)
    args = parser.parse_args()

    df, stats = build_dataset(
        seed=args.seed, n_entities=args.n_entities, n_days=args.n_days, start_date=args.start_date,
    )
    paths = write_outputs(df, stats, args.out_dir)

    print(f"Wrote {stats['total_events']} events "
          f"({stats['n_normal']} normal, {stats['n_anomaly']} anomaly, "
          f"{stats['anomaly_pct']:.2f}% anomalous) to:")
    for p in paths:
        print(f"  - {p}")

    print("\nLabel distribution:")
    print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
