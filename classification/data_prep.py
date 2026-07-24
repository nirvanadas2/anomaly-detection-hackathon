"""
Loads models/scored_events.csv and data_gen/output/access_logs.csv, joins
them on entity_id/event_id, and parses the packed JSON columns
(command_sequence, device_fingerprint) from access_logs.

Both source files carry a `timestamp` column; we keep access_logs's copy
(the original) and drop scored_events's duplicate after verifying they
agree, rather than silently suffixing one.
"""

import json

import pandas as pd

from . import config as cfg


def _parse_device_fingerprint(raw):
    try:
        val = json.loads(raw)
        return val if isinstance(val, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def _parse_command_sequence(raw):
    try:
        val = json.loads(raw)
        return val if isinstance(val, list) else []
    except (TypeError, json.JSONDecodeError):
        return []


def load_merged(scored_events_path=None, access_logs_path=None):
    scored = pd.read_csv(scored_events_path or cfg.SCORED_EVENTS_PATH)
    logs = pd.read_csv(access_logs_path or cfg.ACCESS_LOGS_PATH)

    df = logs.merge(
        scored[["event_id", "entity_id", "risk_score", "score_method"]],
        on=["event_id", "entity_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(df) != len(logs):
        raise ValueError(
            f"Join dropped rows: access_logs has {len(logs)}, merged has {len(df)}. "
            "scored_events.csv and access_logs.csv are expected to cover the same events."
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")

    fp = df["device_fingerprint"].apply(_parse_device_fingerprint)
    df["os"] = fp.apply(lambda d: d.get("os", "unknown"))
    df["mac_address"] = fp.apply(lambda d: d.get("mac_address", "unknown"))
    df["protocol"] = fp.apply(lambda d: d.get("protocol", "unknown"))
    df["command_sequence_parsed"] = df["command_sequence"].apply(_parse_command_sequence)

    df = df.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)
    return df
