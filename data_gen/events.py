"""
Samples normal (non-attack) events from each entity's baseline over the
45-day window. One "day location" is picked per active day (not per event)
so that ordinary dual-office habits never accidentally look like impossible
travel in the ground truth.
"""

import datetime as dt
import json

from faker import Faker

from . import config as cfg

fake = Faker()


def _protocol_for(resource):
    tier = cfg.RESOURCE_TIER_LOOKUP[resource]
    options = cfg.PROTOCOL_FOR_TIER[tier]
    return options[0] if len(options) == 1 else options


def _sample_hour(rng, center, spread):
    h = rng.normal(center, spread) % 24
    return h


def _device_fingerprint(entity, resource, rng):
    tier = cfg.RESOURCE_TIER_LOOKUP[resource]
    protocol_options = cfg.PROTOCOL_FOR_TIER[tier]
    protocol = protocol_options[rng.integers(0, len(protocol_options))]
    return json.dumps({
        "os": entity["os"],
        "mac_address": entity["mac"],
        "protocol": protocol,
    })


def _command_sequence(entity, rng):
    if not entity["privileged"]:
        return "[]"
    k = rng.integers(2, 7)  # 2-6 benign commands
    cmds = list(rng.choice(cfg.BENIGN_COMMANDS, size=k, replace=False))
    return json.dumps(cmds)


def generate_normal_events(entities, n_days=cfg.N_DAYS, start_date=cfg.START_DATE, rng=None):
    rng = rng or cfg.RNG
    start = dt.datetime.fromisoformat(start_date)
    events = []

    for entity in entities:
        for day_offset in range(n_days):
            if rng.random() > entity["active_prob"]:
                continue

            n_events_today = rng.poisson(entity["events_per_day_lambda"])
            if n_events_today <= 0:
                continue

            # Pick a single location for the whole day.
            use_secondary = (
                entity["secondary_city"] is not None and rng.random() < 0.20
            )
            if use_secondary:
                city = entity["secondary_city"]
                if "secondary_ips" not in entity:
                    entity["secondary_ips"] = [fake.ipv4_private()]
                ip_pool = entity["secondary_ips"]
            else:
                city = entity["home_city"]
                ip_pool = entity["home_ips"]

            day_date = start + dt.timedelta(days=day_offset)

            for _ in range(n_events_today):
                hour = _sample_hour(rng, entity["typical_hour_center"], entity["typical_hour_spread"])
                minute = rng.integers(0, 60)
                second = rng.integers(0, 60)
                timestamp = day_date + dt.timedelta(hours=hour, minutes=int(minute), seconds=int(second))

                resource = entity["typical_resources"][rng.integers(0, len(entity["typical_resources"]))]
                auth_method = rng.choice(entity["auth_methods"], p=entity["auth_weights"])
                source_ip = ip_pool[rng.integers(0, len(ip_pool))]

                # Rare, realistic authentication typo/failure (not an attack).
                is_failure = rng.random() < 0.02
                if is_failure:
                    session_duration = 0.0
                    auth_result = "failure"
                else:
                    auth_result = "success"
                    session_duration = max(0.05, rng.normal(entity["session_mean"], entity["session_std"]))

                events.append({
                    "entity_id": entity["entity_id"],
                    "entity_type": entity["entity_type"],
                    "timestamp": timestamp.isoformat(),
                    "source_ip": source_ip,
                    "geo_location": f"{city[0]}, {city[1]}",
                    "resource_accessed": resource,
                    "auth_method": auth_method,
                    "auth_result": auth_result,
                    "session_duration": round(float(session_duration), 2),
                    "command_sequence": _command_sequence(entity, rng),
                    "device_fingerprint": _device_fingerprint(entity, resource, rng),
                    "label": "normal",
                })

    return events
