"""
Builds the population of 300 entities, each with a fixed per-entity
"normal behavior" baseline that event generation later samples from.
"""

from faker import Faker

from . import config as cfg
from .utils import IdCounter

fake = Faker()
Faker.seed(cfg.RANDOM_SEED)

_ENTITY_PREFIX = {"user": "USR", "service_account": "SVC", "edge_device": "DEV"}


def _weighted_choice(rng, options, weights):
    return rng.choice(options, p=weights)


def _pick_typical_resources(rng, entity_type, privileged):
    if entity_type == "edge_device":
        k = rng.integers(1, 3)  # 1-2 resources
        return list(rng.choice(cfg.EDGE_DEVICE_RESOURCES, size=min(k, len(cfg.EDGE_DEVICE_RESOURCES)), replace=False))

    if entity_type == "service_account":
        k = rng.integers(1, 5)  # 1-4 resources, usually tier 1/2
        pool = cfg.RESOURCES_BY_TIER[1] + cfg.RESOURCES_BY_TIER[2]
        if privileged:
            pool = pool + cfg.RESOURCES_BY_TIER[0]
        return list(rng.choice(pool, size=min(k, len(pool)), replace=False))

    # user
    k = rng.integers(3, 9)  # 3-8 resources
    pool = cfg.RESOURCES_BY_TIER[2] + cfg.RESOURCES_BY_TIER[1]
    if privileged:
        pool = pool + cfg.RESOURCES_BY_TIER[0]
    return list(rng.choice(pool, size=min(k, len(pool)), replace=False))


def _auth_weights_for(entity_type):
    w = cfg.AUTH_METHOD_WEIGHTS[entity_type]
    methods = list(w.keys())
    weights = [w[m] for m in methods]
    return methods, weights


def generate_entities(n_entities=cfg.N_ENTITIES, rng=None):
    """Return a list of entity-baseline dicts."""
    rng = rng or cfg.RNG
    counters = {t: IdCounter(_ENTITY_PREFIX[t]) for t in _ENTITY_PREFIX}

    types = list(cfg.ENTITY_TYPE_MIX.keys())
    type_weights = [cfg.ENTITY_TYPE_MIX[t] for t in types]
    assigned_types = rng.choice(types, size=n_entities, p=type_weights)

    entities = []
    for entity_type in assigned_types:
        entity_id = counters[entity_type].next()
        privileged = rng.random() < cfg.PRIVILEGED_FRACTION[entity_type]

        home_city = cfg.CITIES[rng.integers(0, len(cfg.CITIES))]
        has_secondary = rng.random() < 0.10  # e.g. dual-office worker
        secondary_city = None
        if has_secondary and entity_type == "user":
            candidates = [c for c in cfg.CITIES if c[0] != home_city[0]]
            secondary_city = candidates[rng.integers(0, len(candidates))]

        # Typical active hours (24h clock), as a (center, spread) the event
        # sampler draws from with a normal distribution.
        if entity_type == "user":
            night_shift = rng.random() < 0.08
            center = rng.uniform(22, 26) % 24 if night_shift else rng.uniform(8, 17)
            spread = rng.uniform(1.5, 3.0)
            active_prob = rng.uniform(0.45, 0.80)  # not every day is a workday
            events_per_day_lambda = rng.uniform(1.5, 4.5)
        elif entity_type == "service_account":
            center = rng.uniform(0, 24)  # automated jobs run at any scheduled hour
            spread = rng.uniform(0.3, 1.5)  # tight schedule
            active_prob = rng.uniform(0.85, 1.0)
            events_per_day_lambda = rng.uniform(4, 14)
        else:  # edge_device
            center = 12.0
            spread = 24.0  # effectively uniform / round-the-clock heartbeats
            active_prob = rng.uniform(0.95, 1.0)
            events_per_day_lambda = rng.uniform(3, 8)

        session_mean, session_std = {
            "user": (rng.uniform(8, 40), rng.uniform(3, 12)),
            "service_account": (rng.uniform(0.5, 4), rng.uniform(0.2, 1.5)),
            "edge_device": (rng.uniform(0.1, 1.0), rng.uniform(0.05, 0.3)),
        }[entity_type]

        auth_methods, auth_weights = _auth_weights_for(entity_type)
        typical_resources = _pick_typical_resources(rng, entity_type, privileged)

        n_home_ips = rng.integers(1, 3)
        home_ips = [fake.ipv4_private() for _ in range(n_home_ips)]
        os_choice = cfg.OS_POOL[entity_type][rng.integers(0, len(cfg.OS_POOL[entity_type]))]
        mac = fake.mac_address()

        entities.append({
            "entity_id": entity_id,
            "entity_type": entity_type,
            "privileged": bool(privileged),
            "home_city": home_city,
            "secondary_city": secondary_city,
            "typical_hour_center": center,
            "typical_hour_spread": spread,
            "active_prob": active_prob,
            "events_per_day_lambda": events_per_day_lambda,
            "session_mean": session_mean,
            "session_std": session_std,
            "auth_methods": auth_methods,
            "auth_weights": auth_weights,
            "typical_resources": typical_resources,
            "home_ips": home_ips,
            "os": os_choice,
            "mac": mac,
        })

    return entities
