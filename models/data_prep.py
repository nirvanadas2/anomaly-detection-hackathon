"""
Loads access_logs.csv, parses the packed JSON columns (command_sequence,
device_fingerprint), and engineers the per-event feature set consumed by
both the cold-start baseline and the LSTM autoencoder.

Novelty features (has this entity used this resource/geo/IP/device before)
are computed by walking each entity's events in chronological order and
only ever looking backward -- so no event's features depend on anything
that happens after it in time.
"""

import json

import numpy as np
import pandas as pd

from . import config as cfg

NUMERIC_COLUMNS = [
    "log_session_duration",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "log_command_count", "has_commands",
    "log_minutes_since_last",
    "ip_novelty", "device_novelty", "resource_novelty", "geo_novelty",
]

# Cap used for the "gap since previous event" feature when there is no
# previous event yet (first-ever event for an entity) or the entity has
# been dormant a long time -- keeps the log-transform well-behaved.
NO_PRIOR_EVENT_GAP_MINUTES = 7 * 24 * 60


def _parse_command_sequence(raw):
    try:
        val = json.loads(raw)
        return val if isinstance(val, list) else []
    except (TypeError, json.JSONDecodeError):
        return []


def _parse_device_fingerprint(raw):
    try:
        val = json.loads(raw)
        return val if isinstance(val, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def load_raw(path=cfg.ACCESS_LOGS_PATH):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    df["command_sequence_parsed"] = df["command_sequence"].apply(_parse_command_sequence)
    fp = df["device_fingerprint"].apply(_parse_device_fingerprint)
    df["os"] = fp.apply(lambda d: d.get("os", "unknown"))
    df["mac_address"] = fp.apply(lambda d: d.get("mac_address", "unknown"))
    df["protocol"] = fp.apply(lambda d: d.get("protocol", "unknown"))
    return df


def _novelty_features_for_entity(sub):
    """sub: rows for one entity, already sorted by timestamp. Returns dict of
    new columns computed strictly from that entity's own past events."""
    n = len(sub)
    ip_novelty = np.zeros(n)
    device_novelty = np.zeros(n)
    resource_novelty = np.zeros(n)
    geo_novelty = np.zeros(n)
    minutes_since_last = np.full(n, float(NO_PRIOR_EVENT_GAP_MINUTES))

    seen_ips, seen_macs, seen_resources, seen_geo = set(), set(), set(), set()
    prev_ts = None

    ips = sub["source_ip"].to_numpy()
    macs = sub["mac_address"].to_numpy()
    resources = sub["resource_accessed"].to_numpy()
    geos = sub["geo_location"].to_numpy()
    timestamps = sub["timestamp"].to_numpy()

    for i in range(n):
        ip_novelty[i] = 0.0 if ips[i] in seen_ips else 1.0
        device_novelty[i] = 0.0 if macs[i] in seen_macs else 1.0
        resource_novelty[i] = 0.0 if resources[i] in seen_resources else 1.0
        geo_novelty[i] = 0.0 if geos[i] in seen_geo else 1.0

        if prev_ts is not None:
            gap_minutes = (timestamps[i] - prev_ts) / np.timedelta64(1, "m")
            minutes_since_last[i] = max(0.0, float(gap_minutes))

        seen_ips.add(ips[i])
        seen_macs.add(macs[i])
        seen_resources.add(resources[i])
        seen_geo.add(geos[i])
        prev_ts = timestamps[i]

    return {
        "ip_novelty": ip_novelty,
        "device_novelty": device_novelty,
        "resource_novelty": resource_novelty,
        "geo_novelty": geo_novelty,
        "minutes_since_last": minutes_since_last,
    }


def engineer_features(df, start_date=None):
    df = df.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)

    df["log_session_duration"] = np.log1p(df["session_duration"].clip(lower=0))
    df["hour_float"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_float"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_float"] / 24.0)
    df["dow"] = df["timestamp"].dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7.0)

    df["command_count"] = df["command_sequence_parsed"].apply(len)
    df["log_command_count"] = np.log1p(df["command_count"])
    df["has_commands"] = (df["command_count"] > 0).astype(float)

    novelty_frames = []
    for _, sub in df.groupby("entity_id", sort=False):
        feats = _novelty_features_for_entity(sub)
        novelty_frames.append(pd.DataFrame(feats, index=sub.index))
    novelty_df = pd.concat(novelty_frames).sort_index()
    df = pd.concat([df, novelty_df], axis=1)
    df["log_minutes_since_last"] = np.log1p(df["minutes_since_last"])

    if start_date is None:
        start_date = df["timestamp"].min().normalize()
    else:
        start_date = pd.Timestamp(start_date)
    day_offset = (df["timestamp"] - start_date).dt.total_seconds() / 86400.0
    df["day_offset"] = day_offset
    df["epoch"] = (day_offset // cfg.DRIFT_EPOCH_DAYS).astype(int).clip(lower=0)

    # per-entity running event index (0-based position in that entity's own history)
    df["entity_event_index"] = df.groupby("entity_id").cumcount()
    entity_total_events = df.groupby("entity_id")["entity_id"].transform("count")
    df["entity_total_events"] = entity_total_events
    df["is_cold_start_entity"] = df["entity_total_events"] < cfg.MIN_HISTORY_EVENTS
    df["has_full_window"] = df["entity_event_index"] >= (cfg.WINDOW_SIZE - 1)
    df["use_lstm"] = (~df["is_cold_start_entity"]) & df["has_full_window"]

    return df, start_date
