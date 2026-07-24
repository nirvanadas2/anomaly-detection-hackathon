"""
Rule-based classifiers for the six attack patterns. Each rule takes a
candidate event plus its entity's full event history (from access_logs, not
just other high-risk_score events) and returns a RuleResult: whether it
fired, a numeric "strength" (used only for logging/debugging, not for the
final label), the specific feature values that drove the decision, and a
human-readable explanation string.

Rules only ever look at a candidate's own entity's events at or before/around
its own timestamp -- no rule uses another entity's data or future knowledge
beyond the immediate neighboring events a real-time detector would also have
by the time the next event arrives.
"""

from collections import namedtuple

import numpy as np
import pandas as pd

from . import config as cfg
from . import geo

RuleResult = namedtuple("RuleResult", ["matched", "strength", "features", "explanation"])

NO_MATCH = RuleResult(False, 0.0, {}, "")


def _typical_value_set(value_series, mass):
    """Smallest set of values covering `mass` of a series' frequency -- used for
    both an entity's typical resource set and its typical geo_location set."""
    counts = value_series.value_counts()
    total = counts.sum()
    if total == 0:
        return set()
    chosen, cumulative = set(), 0.0
    for value, count in counts.items():
        chosen.add(value)
        cumulative += count / total
        if cumulative >= mass:
            break
    return chosen


def _typical_resource_set(resource_series, mass=cfg.TOP_RESOURCE_MASS):
    return _typical_value_set(resource_series, mass)


def _hour_probability_histogram(timestamp_series):
    """Laplace-smoothed probability of each hour-of-day, from a series of timestamps."""
    hours = timestamp_series.dt.hour.to_numpy()
    counts = np.bincount(hours, minlength=24).astype(float)
    return (counts + 1.0) / (counts.sum() + 24.0)


def _established_baseline(entity_df, candidate_timestamp, buffer_minutes):
    """Entity history strictly before a buffer window ending at the candidate's
    own timestamp. Used instead of `entity_df.iloc[:idx]` for every "typical
    behavior" / novelty baseline in this file: a multi-event incident's own
    earlier events would otherwise leak into the very baseline being used to
    judge it, making later events in the same burst look progressively less
    novel purely because the first event of the burst is now part of their
    own "history" -- silently gutting recall on any incident with more than
    one event. Excluding the recent window keeps the baseline to genuinely
    established behavior, so an entire burst reads as novel, not just its
    first event."""
    cutoff = candidate_timestamp - pd.Timedelta(minutes=buffer_minutes)
    return entity_df[entity_df["timestamp"] < cutoff]


class RuleContext:
    """Precomputed, reusable structures shared across all candidate evaluations."""

    def __init__(self, df):
        self.df = df
        self.entity_groups = {}
        self.entity_positions = {}
        for entity_id, g in df.groupby("entity_id", sort=False):
            g = g.sort_values("timestamp").reset_index(drop=True)
            self.entity_groups[entity_id] = g
            self.entity_positions[entity_id] = {
                event_id: i for i, event_id in enumerate(g["event_id"])
            }
        self.global_resource_freq = df["resource_accessed"].value_counts(normalize=True)

    def entity_frame_and_position(self, entity_id, event_id):
        entity_df = self.entity_groups[entity_id]
        idx = self.entity_positions[entity_id][event_id]
        return entity_df, idx


def brute_force_check(candidate, entity_df, idx, ctx):
    t = candidate.timestamp
    window = pd.Timedelta(minutes=cfg.BRUTE_FORCE_WINDOW_MINUTES)

    nearby = entity_df[
        (entity_df["timestamp"] >= t - window)
        & (entity_df["timestamp"] <= t + window)
        & (entity_df["source_ip"] == candidate.source_ip)
    ]
    failed = nearby[nearby["auth_result"] == "failure"]
    n_failed = len(failed)

    if n_failed < cfg.BRUTE_FORCE_MIN_FAILURES:
        return RuleResult(False, float(n_failed), {"failed_auth_count_in_window": int(n_failed)}, "")

    prior = _established_baseline(entity_df, candidate.timestamp, cfg.BRUTE_FORCE_WINDOW_MINUTES)
    ip_novel = candidate.source_ip not in set(prior["source_ip"]) if len(prior) else True

    features = {
        "failed_auth_count_in_window": int(n_failed),
        "window_minutes": cfg.BRUTE_FORCE_WINDOW_MINUTES,
        "source_ip": candidate.source_ip,
        "source_ip_novel_for_entity": bool(ip_novel),
    }
    explanation = (
        f"brute_force: {n_failed} failed auth_result attempts from source_ip={candidate.source_ip} "
        f"within {cfg.BRUTE_FORCE_WINDOW_MINUTES} min of this event (threshold={cfg.BRUTE_FORCE_MIN_FAILURES})"
        + (" + source_ip never seen before for this entity" if ip_novel else "")
    )
    return RuleResult(True, float(n_failed), features, explanation)


def impossible_travel_check(candidate, entity_df, idx, ctx):
    neighbors = []
    if idx > 0:
        neighbors.append((entity_df.iloc[idx - 1], "previous"))
    if idx < len(entity_df) - 1:
        neighbors.append((entity_df.iloc[idx + 1], "next"))

    best = None
    for other, direction in neighbors:
        dist = geo.distance_km(candidate.geo_location, other["geo_location"])
        if dist is None:
            continue
        dh = abs((candidate.timestamp - other["timestamp"]).total_seconds()) / 3600.0
        dh = max(dh, cfg.MIN_TIME_DELTA_HOURS)
        speed = dist / dh
        if best is None or speed > best[0]:
            best = (speed, dist, dh, other, direction)

    if best is None:
        return NO_MATCH

    speed, dist, dh, other, direction = best
    if speed <= cfg.MAX_PLAUSIBLE_TRAVEL_KMH:
        return RuleResult(False, speed, {"geo_velocity_kmh": round(speed, 1)}, "")

    device_changed = candidate.mac_address != other["mac_address"]
    features = {
        "geo_velocity_kmh": round(speed, 1),
        "distance_km": round(dist, 1),
        "time_delta_hours": round(dh, 3),
        "compared_to_event": direction,
        "other_geo_location": other["geo_location"],
        "device_fingerprint_changed": bool(device_changed),
    }
    explanation = (
        f"impossible_travel: geo-velocity of {speed:,.0f} km/h between {other['geo_location']} "
        f"and {candidate.geo_location} over {dh:.2f}h (comparing to the {direction} event for this "
        f"entity), exceeding the {cfg.MAX_PLAUSIBLE_TRAVEL_KMH:.0f} km/h plausibility ceiling"
        + (" + new device fingerprint" if device_changed else "")
    )
    return RuleResult(True, speed, features, explanation)


def lateral_movement_check(candidate, entity_df, idx, ctx):
    prior = _established_baseline(entity_df, candidate.timestamp, cfg.LATERAL_MOVEMENT_LOOKBACK_MINUTES)
    if len(prior) < cfg.MIN_PRIOR_EVENTS_FOR_PROFILE:
        return NO_MATCH

    typical_set = _typical_resource_set(prior["resource_accessed"])
    if candidate.resource_accessed in typical_set:
        return NO_MATCH

    window = pd.Timedelta(minutes=cfg.LATERAL_MOVEMENT_LOOKBACK_MINUTES)
    t = candidate.timestamp
    recent = entity_df[(entity_df["timestamp"] >= t - window) & (entity_df["timestamp"] <= t)]
    recent = recent.sort_values("timestamp")
    novel_recent = recent[~recent["resource_accessed"].isin(typical_set)]

    chain = []
    for resource in novel_recent["resource_accessed"]:
        if resource not in chain:
            chain.append(resource)
    if candidate.resource_accessed not in chain:
        chain.append(candidate.resource_accessed)

    chain_length = len(chain)
    if chain_length < cfg.LATERAL_MOVEMENT_MIN_CHAIN:
        return RuleResult(False, float(chain_length), {"chain_length": chain_length}, "")

    freqs = [float(ctx.global_resource_freq.get(r, 0.0)) for r in chain]
    escalating = freqs[-1] < freqs[0]
    if not escalating:
        return RuleResult(False, float(chain_length),
                           {"chain_length": chain_length, "escalating_by_global_frequency": False}, "")

    features = {
        "novel_resource_chain": chain,
        "chain_length": chain_length,
        "lookback_minutes": cfg.LATERAL_MOVEMENT_LOOKBACK_MINUTES,
        "global_frequency_first": round(freqs[0], 4),
        "global_frequency_last": round(freqs[-1], 4),
    }
    explanation = (
        f"lateral_movement: accessed {chain_length} resources outside this entity's normal set within "
        f"{cfg.LATERAL_MOVEMENT_LOOKBACK_MINUTES} min ({' -> '.join(chain)}), escalating toward "
        f"less common resources (global access frequency {freqs[0]:.3f} -> {freqs[-1]:.3f})"
    )
    return RuleResult(True, float(chain_length), features, explanation)


def credential_misuse_check(candidate, entity_df, idx, ctx):
    prior = _established_baseline(entity_df, candidate.timestamp, cfg.CREDENTIAL_MISUSE_CLUSTER_WINDOW_MINUTES)
    if len(prior) < cfg.MIN_PRIOR_EVENTS_FOR_PROFILE:
        return NO_MATCH

    device_novel = candidate.mac_address not in set(prior["mac_address"])
    typical_geo = _typical_value_set(prior["geo_location"], cfg.TOP_GEO_MASS)
    geo_novel = candidate.geo_location not in typical_geo

    hour_hist = _hour_probability_histogram(prior["timestamp"])
    hour = candidate.timestamp.hour
    hour_prob = float(hour_hist[hour])
    off_hours = hour_prob < cfg.CREDENTIAL_MISUSE_OFF_HOURS_MAX_PROB

    if not (device_novel and geo_novel and off_hours):
        return RuleResult(False, 0.0, {
            "device_novel_for_entity": bool(device_novel),
            "geo_novel_for_entity": bool(geo_novel),
            "off_hours": bool(off_hours),
        }, "")

    window = pd.Timedelta(minutes=cfg.CREDENTIAL_MISUSE_CLUSTER_WINDOW_MINUTES)
    t = candidate.timestamp
    cluster = entity_df[
        (entity_df["timestamp"] >= t - window)
        & (entity_df["timestamp"] <= t + window)
        & (entity_df["mac_address"] == candidate.mac_address)
        & (entity_df["geo_location"] == candidate.geo_location)
    ]
    cluster_size = len(cluster)

    features = {
        "device_novel_for_entity": True,
        "mac_address": candidate.mac_address,
        "geo_novel_for_entity": True,
        "geo_location": candidate.geo_location,
        "off_hours": True,
        "hour": int(hour),
        "hour_probability_in_baseline": round(hour_prob, 4),
        "cluster_size_same_device_and_geo": int(cluster_size),
        "cluster_window_minutes": cfg.CREDENTIAL_MISUSE_CLUSTER_WINDOW_MINUTES,
    }
    explanation = (
        f"credential_misuse: new device_fingerprint (mac={candidate.mac_address}) + unusual "
        f"geo_location ({candidate.geo_location}, outside this entity's typical set) + off-hours "
        f"login (hour={hour}, baseline probability={hour_prob:.3f} < {cfg.CREDENTIAL_MISUSE_OFF_HOURS_MAX_PROB:.2f}) "
        f"occurring together"
        + (f" + part of a {cluster_size}-event cluster sharing this device+location within "
           f"{cfg.CREDENTIAL_MISUSE_CLUSTER_WINDOW_MINUTES} min" if cluster_size > 1 else "")
    )
    strength = (1.0 - hour_prob) + float(device_novel) + float(geo_novel)
    return RuleResult(True, strength, features, explanation)


def device_spoofing_check(candidate, entity_df, idx, ctx):
    prior = _established_baseline(entity_df, candidate.timestamp, cfg.DEVICE_SPOOFING_BASELINE_BUFFER_MINUTES)
    if len(prior) < cfg.MIN_PRIOR_EVENTS_FOR_PROFILE:
        return NO_MATCH

    historical_os = set(prior["os"])
    historical_mac = set(prior["mac_address"])
    os_novel = candidate.os not in historical_os
    mac_novel = candidate.mac_address not in historical_mac

    if not (os_novel or mac_novel):
        return NO_MATCH

    window = pd.Timedelta(minutes=cfg.DEVICE_SPOOFING_SESSION_WINDOW_MINUTES)
    t = candidate.timestamp
    neighbors = entity_df[
        (entity_df["timestamp"] >= t - window)
        & (entity_df["timestamp"] <= t + window)
        & (entity_df["event_id"] != candidate.event_id)
    ]
    consistent_neighbors = neighbors[
        neighbors["os"].isin(historical_os) & neighbors["mac_address"].isin(historical_mac)
    ]
    mid_session_change = len(consistent_neighbors) > 0

    no_consistent_combo = os_novel and mac_novel
    if not (no_consistent_combo or mid_session_change):
        return RuleResult(False, 0.0, {
            "os_novel_for_entity": bool(os_novel), "mac_novel_for_entity": bool(mac_novel),
        }, "")

    features = {
        "os": candidate.os,
        "mac_address": candidate.mac_address,
        "protocol": candidate.protocol,
        "os_novel_for_entity": bool(os_novel),
        "mac_novel_for_entity": bool(mac_novel),
        "mid_session_change": bool(mid_session_change),
        "window_minutes": cfg.DEVICE_SPOOFING_SESSION_WINDOW_MINUTES,
        "n_prior_events_checked": len(prior),
    }
    if mid_session_change:
        explanation = (
            f"device_spoofing: device_fingerprint changed mid-session -- os={candidate.os}, "
            f"mac={candidate.mac_address} appears within {cfg.DEVICE_SPOOFING_SESSION_WINDOW_MINUTES} min "
            f"of {len(consistent_neighbors)} event(s) using this entity's historically consistent fingerprint"
        )
    else:
        explanation = (
            f"device_spoofing: os={candidate.os} + mac={candidate.mac_address} matches no historically "
            f"consistent OS/MAC combination observed for this entity across {len(prior)} prior events"
        )
    strength = float(os_novel) + float(mac_novel) + float(mid_session_change)
    return RuleResult(True, strength, features, explanation)


def credential_stuffing_check(candidate, entity_df, idx, ctx):
    """Cross-entity pattern: unlike every other rule here, this one looks past
    the candidate's own entity at source_ip behavior across the WHOLE
    population in a time window -- many different entity_ids, mostly failing,
    from a small set of IPs. That's the entire distinction from brute_force,
    which is many attempts against ONE entity from one IP.

    Deliberately aggregates across a *cluster* of related IPs rather than
    requiring the candidate's own source_ip alone to clear the distinct-
    entity bar: a real incident's targets are split across 2-4 IPs, so any
    one IP typically only accounts for a fraction of the total reach. Judging
    each IP in isolation systematically undercounts every IP that happened
    to get fewer targets in the split."""
    window = pd.Timedelta(minutes=cfg.CREDENTIAL_STUFFING_WINDOW_MINUTES)
    t = candidate.timestamp
    nearby_all = ctx.df[(ctx.df["timestamp"] >= t - window) & (ctx.df["timestamp"] <= t + window)]

    per_ip_entities = nearby_all.groupby("source_ip")["entity_id"].nunique()
    candidate_ip_entities = int(per_ip_entities.get(candidate.source_ip, 0))
    if candidate_ip_entities < cfg.CREDENTIAL_STUFFING_PER_IP_MIN_ENTITIES:
        return RuleResult(False, float(candidate_ip_entities),
                           {"distinct_entities_same_source_ip": candidate_ip_entities}, "")

    active_ips = per_ip_entities[per_ip_entities >= cfg.CREDENTIAL_STUFFING_PER_IP_MIN_ENTITIES]
    if len(active_ips) > cfg.CREDENTIAL_STUFFING_MAX_CLUSTER_IPS:
        # Too many independently-active IPs to read as "a small related set" --
        # e.g. a genuinely busy window, not a coordinated stuffing run.
        return RuleResult(False, float(candidate_ip_entities), {
            "distinct_entities_same_source_ip": candidate_ip_entities,
            "active_ip_cluster_size": int(len(active_ips)),
        }, "")

    cluster_events = nearby_all[nearby_all["source_ip"].isin(active_ips.index)]
    cluster_distinct_entities = int(cluster_events["entity_id"].nunique())
    n_total = len(cluster_events)
    n_failed = int((cluster_events["auth_result"] == "failure").sum())
    failure_rate = n_failed / n_total if n_total else 0.0

    if (cluster_distinct_entities < cfg.CREDENTIAL_STUFFING_MIN_DISTINCT_ENTITIES
            or failure_rate < cfg.CREDENTIAL_STUFFING_MIN_FAILURE_RATE):
        return RuleResult(False, float(cluster_distinct_entities), {
            "active_ip_cluster_size": int(len(active_ips)),
            "cluster_distinct_entities": cluster_distinct_entities,
            "cluster_failure_rate": round(failure_rate, 3),
        }, "")

    features = {
        "active_ip_cluster": list(active_ips.index),
        "active_ip_cluster_size": int(len(active_ips)),
        "cluster_distinct_entities": cluster_distinct_entities,
        "cluster_total_attempts": int(n_total),
        "cluster_failed_attempts": n_failed,
        "cluster_failure_rate": round(failure_rate, 3),
        "window_minutes": cfg.CREDENTIAL_STUFFING_WINDOW_MINUTES,
    }
    explanation = (
        f"credential_stuffing: a cluster of {len(active_ips)} related source_ips "
        f"({', '.join(active_ips.index)}) made {n_total} auth attempts against "
        f"{cluster_distinct_entities} distinct entity_ids within {cfg.CREDENTIAL_STUFFING_WINDOW_MINUTES} min "
        f"({n_failed}/{n_total} = {failure_rate:.0%} failed) -- breadth-across-accounts pattern, not a "
        f"single-entity brute_force"
    )
    strength = float(cluster_distinct_entities) * failure_rate
    return RuleResult(True, strength, features, explanation)


# Evaluated in this order; the first rule to fire wins the final label.
# credential_stuffing runs right after brute_force since both are auth-
# failure-based, but they can never collide in practice: brute_force needs
# >= BRUTE_FORCE_MIN_FAILURES failures against the SAME entity, while
# credential_stuffing needs failures spread across >= 8 DIFFERENT entities
# from the same source_ip -- disjoint by construction.
# credential_misuse is checked before device_spoofing on purpose: a
# credential_misuse event's new device makes both its OS and MAC "novel for
# this entity" too, which would otherwise also satisfy device_spoofing's
# condition. credential_misuse is the strictly more specific pattern (it
# additionally requires novel geo_location AND off-hours timing, both of
# which device_spoofing deliberately leaves normal), so the more specific
# match should win rather than the broader device-only check grabbing it
# first. impossible_travel/lateral_movement run last since their signals
# (geo-velocity, resource sequence) are the least specific.
RULES = [
    ("brute_force", brute_force_check),
    ("credential_stuffing", credential_stuffing_check),
    ("credential_misuse", credential_misuse_check),
    ("device_spoofing", device_spoofing_check),
    ("impossible_travel", impossible_travel_check),
    ("lateral_movement", lateral_movement_check),
]
