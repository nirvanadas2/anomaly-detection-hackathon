"""
Attack-pattern injectors. Each function returns a list of event dicts
(same shape as normal events) already carrying their true `label`.
"""

import datetime as dt
import json

from faker import Faker

from . import config as cfg
from .utils import haversine_km

fake = Faker()


def _protocol_for(resource, rng):
    tier = cfg.RESOURCE_TIER_LOOKUP[resource]
    options = cfg.PROTOCOL_FOR_TIER[tier]
    return options[rng.integers(0, len(options))]


def _fingerprint(os_name, mac, resource, rng):
    return json.dumps({
        "os": os_name,
        "mac_address": mac,
        "protocol": _protocol_for(resource, rng),
    })


def _random_window_start(rng, n_days, start_date, margin_hours=6):
    start = dt.datetime.fromisoformat(start_date)
    max_offset_seconds = (n_days * 24 - margin_hours) * 3600
    offset = rng.integers(0, max_offset_seconds)
    return start + dt.timedelta(seconds=int(offset))


# ---------------------------------------------------------------------------
# 1. Brute force: burst of failed auth attempts from one external source IP,
#    tight time intervals, against one target entity.
# ---------------------------------------------------------------------------
def inject_brute_force(entities, rng, n_days=cfg.N_DAYS, start_date=cfg.START_DATE):
    entity = entities[rng.integers(0, len(entities))]
    n_attempts = int(rng.integers(5, 31))
    attacker_ip = fake.ipv4_public()
    attacker_mac = fake.mac_address()
    attacker_os = "Unknown"
    other_cities = [c for c in cfg.CITIES if c[0] != entity["home_city"][0]]
    attacker_city = other_cities[rng.integers(0, len(other_cities))]

    t = _random_window_start(rng, n_days, start_date)
    breach_idx = n_attempts - 1 if rng.random() < 0.20 else None  # 20% chance of eventual success

    events = []
    for i in range(n_attempts):
        t += dt.timedelta(seconds=float(rng.uniform(1, 15)))
        resource = entity["typical_resources"][rng.integers(0, len(entity["typical_resources"]))]
        auth_method = cfg.AUTH_METHODS[rng.integers(0, len(cfg.AUTH_METHODS))]

        success = (i == breach_idx)
        events.append({
            "entity_id": entity["entity_id"],
            "entity_type": entity["entity_type"],
            "timestamp": t.isoformat(),
            "source_ip": attacker_ip,
            "geo_location": f"{attacker_city[0]}, {attacker_city[1]}",
            "resource_accessed": resource,
            "auth_method": auth_method,
            "auth_result": "success" if success else "failure",
            "session_duration": round(float(max(0.05, rng.normal(3, 1))), 2) if success else 0.0,
            "command_sequence": "[]",
            "device_fingerprint": _fingerprint(attacker_os, attacker_mac, resource, rng),
            "label": "anomaly_brute_force",
        })
    return events


# ---------------------------------------------------------------------------
# 2. Impossible travel: a legitimate-looking login followed shortly after by
#    a login from a location too far away to have been reached in time.
# ---------------------------------------------------------------------------
def inject_impossible_travel(entities, rng, n_days=cfg.N_DAYS, start_date=cfg.START_DATE):
    entity = entities[rng.integers(0, len(entities))]
    home = entity["home_city"]

    delta_hours = float(rng.uniform(0.25, 3.0))
    min_distance_km = cfg.MAX_PLAUSIBLE_TRAVEL_KMH * delta_hours * 1.2  # 20% safety margin
    candidates = [
        c for c in cfg.CITIES
        if haversine_km(home[2], home[3], c[2], c[3]) > min_distance_km
    ]
    if not candidates:
        candidates = [c for c in cfg.CITIES if c[0] != home[0]]
    far_city = candidates[rng.integers(0, len(candidates))]

    t1 = _random_window_start(rng, n_days, start_date, margin_hours=int(delta_hours) + 1)
    t2 = t1 + dt.timedelta(hours=delta_hours)

    resource1 = entity["typical_resources"][rng.integers(0, len(entity["typical_resources"]))]
    resource2 = entity["typical_resources"][rng.integers(0, len(entity["typical_resources"]))]
    auth1 = rng.choice(entity["auth_methods"], p=entity["auth_weights"])
    auth2 = rng.choice(entity["auth_methods"], p=entity["auth_weights"])

    legit_event = {
        "entity_id": entity["entity_id"],
        "entity_type": entity["entity_type"],
        "timestamp": t1.isoformat(),
        "source_ip": entity["home_ips"][rng.integers(0, len(entity["home_ips"]))],
        "geo_location": f"{home[0]}, {home[1]}",
        "resource_accessed": resource1,
        "auth_method": auth1,
        "auth_result": "success",
        "session_duration": round(float(max(0.05, rng.normal(entity["session_mean"], entity["session_std"]))), 2),
        "command_sequence": "[]",
        "device_fingerprint": _fingerprint(entity["os"], entity["mac"], resource1, rng),
        "label": "normal",
    }

    impossible_event = {
        "entity_id": entity["entity_id"],
        "entity_type": entity["entity_type"],
        "timestamp": t2.isoformat(),
        "source_ip": fake.ipv4_public(),
        "geo_location": f"{far_city[0]}, {far_city[1]}",
        "resource_accessed": resource2,
        "auth_method": auth2,
        "auth_result": "success",
        "session_duration": round(float(max(0.05, rng.normal(entity["session_mean"], entity["session_std"]))), 2),
        "command_sequence": "[]",
        "device_fingerprint": _fingerprint(entity["os"], entity["mac"], resource2, rng),
        "label": "anomaly_impossible_travel",
    }
    return [legit_event, impossible_event]


# ---------------------------------------------------------------------------
# 3. Lateral movement: entity steps through resources outside its normal
#    set in a sequential, escalating (tier 2 -> 1 -> 0) pattern.
# ---------------------------------------------------------------------------
def inject_lateral_movement(entities, rng, n_days=cfg.N_DAYS, start_date=cfg.START_DATE):
    entity = entities[rng.integers(0, len(entities))]
    typical = set(entity["typical_resources"])

    pool = {
        tier: [r for r in resources if r not in typical]
        for tier, resources in cfg.RESOURCES_BY_TIER.items()
    }
    for tier in pool:
        if not pool[tier]:
            pool[tier] = cfg.RESOURCES_BY_TIER[tier]  # fallback if everything overlaps

    k = int(rng.integers(4, 11))
    t = _random_window_start(rng, n_days, start_date)

    events = []
    for i in range(k):
        progress = i / max(k - 1, 1)
        tier = 2 if progress < 0.4 else (1 if progress < 0.75 else 0)
        resource = pool[tier][rng.integers(0, len(pool[tier]))]

        t += dt.timedelta(seconds=float(rng.uniform(30, 300)))
        auth_method = rng.choice(entity["auth_methods"], p=entity["auth_weights"])

        if entity["privileged"]:
            stage_idx = min(len(cfg.MALICIOUS_COMMAND_STAGES) - 1,
                             int(progress * len(cfg.MALICIOUS_COMMAND_STAGES)))
            command_sequence = json.dumps(cfg.MALICIOUS_COMMAND_STAGES[stage_idx])
        else:
            command_sequence = "[]"

        session_duration = round(float(max(0.1, entity["session_mean"] * (0.5 + progress) + rng.normal(0, 1))), 2)

        events.append({
            "entity_id": entity["entity_id"],
            "entity_type": entity["entity_type"],
            "timestamp": t.isoformat(),
            "source_ip": entity["home_ips"][rng.integers(0, len(entity["home_ips"]))],
            "geo_location": f"{entity['home_city'][0]}, {entity['home_city'][1]}",
            "resource_accessed": resource,
            "auth_method": auth_method,
            "auth_result": "success",
            "session_duration": session_duration,
            "command_sequence": command_sequence,
            "device_fingerprint": _fingerprint(entity["os"], entity["mac"], resource, rng),
            "label": "anomaly_lateral_movement",
        })
    return events


# ---------------------------------------------------------------------------
# 4. Credential misuse: a stolen-credential login -- new device fingerprint,
#    unfamiliar/off-hours geo and time, all at once, plus unfamiliar
#    enumeration commands if the account is privileged.
# ---------------------------------------------------------------------------
def inject_credential_misuse(entities, rng, n_days=cfg.N_DAYS, start_date=cfg.START_DATE):
    candidates = [e for e in entities if e["entity_type"] in ("user", "service_account")]
    entity = candidates[rng.integers(0, len(candidates))]

    # Off-hours: the opposite side of the clock from this entity's typical window,
    # not just "a bit later" -- someone unfamiliar with its normal rhythm.
    off_hour = (entity["typical_hour_center"] + 12 + float(rng.uniform(-1.5, 1.5))) % 24
    t = _random_window_start(rng, n_days, start_date, margin_hours=30)
    t = t.replace(hour=int(off_hour), minute=int(rng.integers(0, 60)), second=int(rng.integers(0, 60)), microsecond=0)

    known_city_names = {entity["home_city"][0]}
    if entity["secondary_city"] is not None:
        known_city_names.add(entity["secondary_city"][0])
    geo_candidates = [c for c in cfg.CITIES if c[0] not in known_city_names]
    geo = geo_candidates[rng.integers(0, len(geo_candidates))]

    new_mac = fake.mac_address()
    other_os_options = [o for o in cfg.OS_POOL[entity["entity_type"]] if o != entity["os"]]
    new_os = other_os_options[rng.integers(0, len(other_os_options))] if other_os_options else entity["os"]

    n_events = int(rng.integers(1, 4))  # 1-3 events: a short, unfamiliar session
    events = []
    for _ in range(n_events):
        t += dt.timedelta(seconds=float(rng.uniform(30, 240)))
        resource = entity["typical_resources"][rng.integers(0, len(entity["typical_resources"]))]
        auth_method = cfg.AUTH_METHODS[rng.integers(0, len(cfg.AUTH_METHODS))]

        if entity["privileged"]:
            k = int(rng.integers(2, 5))
            command_sequence = json.dumps(list(rng.choice(cfg.CREDENTIAL_MISUSE_COMMANDS, size=k, replace=False)))
        else:
            command_sequence = "[]"

        session_duration = round(float(max(0.05, entity["session_mean"] * rng.uniform(0.2, 0.6))), 2)

        events.append({
            "entity_id": entity["entity_id"],
            "entity_type": entity["entity_type"],
            "timestamp": t.isoformat(),
            "source_ip": fake.ipv4_public(),
            "geo_location": f"{geo[0]}, {geo[1]}",
            "resource_accessed": resource,
            "auth_method": auth_method,
            "auth_result": "success",
            "session_duration": session_duration,
            "command_sequence": command_sequence,
            "device_fingerprint": _fingerprint(new_os, new_mac, resource, rng),
            "label": "anomaly_credential_misuse",
        })
    return events


# ---------------------------------------------------------------------------
# 5. Device spoofing: a session that starts with the entity's normal device
#    fingerprint, then the fingerprint switches mid-session to a MAC/OS
#    combination that isn't even plausible for this entity type.
# ---------------------------------------------------------------------------
def inject_device_spoofing(entities, rng, n_days=cfg.N_DAYS, start_date=cfg.START_DATE):
    entity = entities[rng.integers(0, len(entities))]
    home = entity["home_city"]

    k = int(rng.integers(3, 7))  # 3-6 events forming one session
    split_idx = max(1, k // 2)  # events before this index keep the entity's real fingerprint

    spoof_mac = fake.mac_address()
    foreign_pool = cfg.FOREIGN_OS_POOL[entity["entity_type"]]
    spoof_os = foreign_pool[rng.integers(0, len(foreign_pool))]
    session_auth_method = rng.choice(entity["auth_methods"], p=entity["auth_weights"])

    t = _random_window_start(rng, n_days, start_date)
    events = []
    for i in range(k):
        t += dt.timedelta(seconds=float(rng.uniform(20, 180)))
        resource = entity["typical_resources"][rng.integers(0, len(entity["typical_resources"]))]
        spoofed = i >= split_idx
        os_name = spoof_os if spoofed else entity["os"]
        mac = spoof_mac if spoofed else entity["mac"]

        # Keep geo/time/resource pattern and command flavor normal -- the sole
        # distinguishing signal for this attack is device identity mid-session.
        if entity["privileged"]:
            kcmd = int(rng.integers(2, 7))
            command_sequence = json.dumps(list(rng.choice(cfg.BENIGN_COMMANDS, size=kcmd, replace=False)))
        else:
            command_sequence = "[]"

        events.append({
            "entity_id": entity["entity_id"],
            "entity_type": entity["entity_type"],
            "timestamp": t.isoformat(),
            "source_ip": entity["home_ips"][rng.integers(0, len(entity["home_ips"]))],
            "geo_location": f"{home[0]}, {home[1]}",
            "resource_accessed": resource,
            "auth_method": session_auth_method,
            "auth_result": "success",
            "session_duration": round(float(max(0.05, rng.normal(entity["session_mean"], entity["session_std"]))), 2),
            "command_sequence": command_sequence,
            "device_fingerprint": _fingerprint(os_name, mac, resource, rng),
            "label": "anomaly_device_spoofing" if spoofed else "normal",
        })
    return events


# ---------------------------------------------------------------------------
# 6. Credential stuffing: a small set of source IPs (2-4) each trying stolen
#    credentials against many different entities in a short window -- breadth
#    across accounts, not repeated attempts against one account like
#    brute_force.
# ---------------------------------------------------------------------------
def inject_credential_stuffing(entities, rng, n_days=cfg.N_DAYS, start_date=cfg.START_DATE):
    n_ips = int(rng.integers(2, 5))  # 2-4
    attacker_ips = [fake.ipv4_public() for _ in range(n_ips)]
    ip_city = {ip: cfg.CITIES[rng.integers(0, len(cfg.CITIES))] for ip in attacker_ips}
    ip_mac = {ip: fake.mac_address() for ip in attacker_ips}
    attacker_os = "Unknown"

    n_targets = int(rng.integers(15, 41))
    target_idx = rng.choice(len(entities), size=min(n_targets, len(entities)), replace=False)
    target_entities = [entities[i] for i in target_idx]

    window_minutes = int(rng.integers(10, 31))
    t_start = _random_window_start(rng, n_days, start_date)
    success_prob = float(rng.uniform(0.05, 0.15))  # "a few" succeed -- most fail

    events = []
    for entity in target_entities:
        ip = attacker_ips[rng.integers(0, n_ips)]
        t = t_start + dt.timedelta(seconds=float(rng.uniform(0, window_minutes * 60)))
        resource = entity["typical_resources"][rng.integers(0, len(entity["typical_resources"]))]
        auth_method = cfg.AUTH_METHODS[rng.integers(0, len(cfg.AUTH_METHODS))]
        success = rng.random() < success_prob

        events.append({
            "entity_id": entity["entity_id"],
            "entity_type": entity["entity_type"],
            "timestamp": t.isoformat(),
            "source_ip": ip,
            "geo_location": f"{ip_city[ip][0]}, {ip_city[ip][1]}",
            "resource_accessed": resource,
            "auth_method": auth_method,
            "auth_result": "success" if success else "failure",
            "session_duration": round(float(max(0.05, rng.normal(3, 1))), 2) if success else 0.0,
            "command_sequence": "[]",
            "device_fingerprint": _fingerprint(attacker_os, ip_mac[ip], resource, rng),
            "label": "anomaly_credential_stuffing",
        })
    return events


ATTACK_GENERATORS = {
    "brute_force": inject_brute_force,
    "impossible_travel": inject_impossible_travel,
    "lateral_movement": inject_lateral_movement,
    "credential_misuse": inject_credential_misuse,
    "device_spoofing": inject_device_spoofing,
    "credential_stuffing": inject_credential_stuffing,
}
