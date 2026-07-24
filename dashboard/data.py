"""
Cached data loading/joining for the dashboard. Reads classified_alerts.csv
and access_logs.csv (joined on entity_id/event_id), and reuses
classification/evaluate.py's own precision/recall/F1 logic rather than
recomputing it independently.
"""

import os
import sys

import pandas as pd
import streamlit as st

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from classification import config as clf_cfg  # noqa: E402
from classification.evaluate import (  # noqa: E402
    budget_false_positive_rates, load_alerts_with_truth, per_type_metrics,
)

ACCESS_LOGS_PATH = clf_cfg.ACCESS_LOGS_PATH
ALERTS_PATH = clf_cfg.CLASSIFIED_ALERTS_PATH
GROUND_TRUTH_PATH = clf_cfg.GROUND_TRUTH_PATH


@st.cache_data
def load_access_logs():
    df = pd.read_csv(ACCESS_LOGS_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    return df


@st.cache_data
def load_alerts():
    return pd.read_csv(ALERTS_PATH)


@st.cache_data
def load_alerts_with_context():
    """Alerts joined with the access-log fields the UI needs (timestamp, entity_type)."""
    alerts = load_alerts()
    logs = load_access_logs()
    merged = alerts.merge(
        logs[["event_id", "entity_id", "timestamp", "entity_type", "resource_accessed"]],
        on=["event_id", "entity_id"],
        how="left",
    )
    return merged


@st.cache_data
def compute_metrics():
    """Thin cached wrapper around classification/evaluate.py's own functions."""
    merged, truth = load_alerts_with_truth()
    truth_label_counts = truth["label"].value_counts()
    metrics_df = per_type_metrics(merged, truth_label_counts)
    fpr = budget_false_positive_rates(merged, truth_label_counts)
    return metrics_df, fpr, merged, truth_label_counts


def typical_resource_set(resource_series, mass=clf_cfg.TOP_RESOURCE_MASS):
    """Smallest set of resources covering `mass` of an entity's access frequency
    -- same definition classification/rules.py uses for its lateral_movement rule."""
    counts = resource_series.value_counts()
    total = counts.sum()
    if total == 0:
        return set()
    chosen, cumulative = set(), 0.0
    for resource, count in counts.items():
        chosen.add(resource)
        cumulative += count / total
        if cumulative >= mass:
            break
    return chosen
