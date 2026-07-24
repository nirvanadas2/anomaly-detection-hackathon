"""Paths and tunable constants for the rule-based classification module."""

import os

_HERE = os.path.dirname(__file__)
_ROOT = os.path.dirname(_HERE)

SCORED_EVENTS_PATH = os.path.join(_ROOT, "models", "scored_events.csv")
ACCESS_LOGS_PATH = os.path.join(_ROOT, "data_gen", "output", "access_logs.csv")
GROUND_TRUTH_PATH = os.path.join(_ROOT, "data_gen", "output", "ground_truth_labels.csv")

OUTPUT_DIR = _HERE
CLASSIFIED_ALERTS_PATH = os.path.join(OUTPUT_DIR, "classified_alerts.csv")

# --- alert budget --------------------------------------------------------
# Only events at/above this risk_score percentile get evaluated by the rules
# at all. This constant is the single knob for "how big is the SOC's queue".
ALERT_BUDGET_FRACTION = 0.01  # top 1%

# --- brute_force -----------------------------------------------------------
# Evaluated directly on auth_result failures near the candidate event, not on
# risk_score, because a successful final "breach" attempt in a brute-force
# burst can itself have a middling risk_score.
BRUTE_FORCE_WINDOW_MINUTES = 10
BRUTE_FORCE_MIN_FAILURES = 4

# --- impossible_travel -----------------------------------------------------
# Same physical ceiling used by the generator (fastest plausible point-to-
# point commercial travel, already padded past cruise speed for connections);
# re-derived independently here from geo_location + timestamp, not imported
# from data_gen.
MAX_PLAUSIBLE_TRAVEL_KMH = 900.0
MIN_TIME_DELTA_HOURS = 1.0 / 60.0  # 1 minute floor to avoid divide-by-zero on same-timestamp events

# --- lateral_movement --------------------------------------------------------
TOP_RESOURCE_MASS = 0.80          # defines an entity's "typical" resource set
MIN_PRIOR_EVENTS_FOR_PROFILE = 5  # need at least this many prior events to trust the typical set
LATERAL_MOVEMENT_LOOKBACK_MINUTES = 20
LATERAL_MOVEMENT_MIN_CHAIN = 3    # candidate + at least 2 other distinct novel resources

# --- credential_misuse -------------------------------------------------------
# Same "typical set" mass used for resources/geo -- TOP_RESOURCE_MASS is reused
# generically for geo_location too (see rules._typical_value_set).
TOP_GEO_MASS = 0.80
CREDENTIAL_MISUSE_OFF_HOURS_MAX_PROB = 0.03   # hour must be rarer than this share of the entity's history
CREDENTIAL_MISUSE_CLUSTER_WINDOW_MINUTES = 15  # window used to report a corroborating event cluster

# --- device_spoofing ----------------------------------------------------------
DEVICE_SPOOFING_SESSION_WINDOW_MINUTES = 5    # how close a "consistent-fingerprint" neighbor must be to call it mid-session
DEVICE_SPOOFING_BASELINE_BUFFER_MINUTES = 30  # excluded from the historical OS/MAC baseline, so a session doesn't self-legitimize

# --- credential_stuffing -------------------------------------------------------
# A cross-entity pattern: unlike every other rule, this one looks at
# source_ip behavior across the WHOLE population in a window, not at one
# entity's own history. Judged as a *cluster* of related IPs rather than any
# single IP in isolation, since a real incident's targets get split across
# 2-4 IPs and no single one may individually clear a high bar.
CREDENTIAL_STUFFING_WINDOW_MINUTES = 30
CREDENTIAL_STUFFING_PER_IP_MIN_ENTITIES = 3     # floor for one IP to count as "active" in the cluster
CREDENTIAL_STUFFING_MAX_CLUSTER_IPS = 6          # upper bound for still calling it "a small set" of IPs
CREDENTIAL_STUFFING_MIN_DISTINCT_ENTITIES = 8   # combined distinct entity_ids across the whole active-IP cluster
CREDENTIAL_STUFFING_MIN_FAILURE_RATE = 0.70     # across the whole cluster's attempts in the window

RANDOM_SEED = 42
