"""
Per-entity-type baseline profiles (the cold-start fallback), plus the
rolling-refresh / calibration machinery that implements concept-drift
handling for both the fallback path and the LSTM path.

Design: every 7 simulated days ("epoch"), we recompute, per entity_type,
using only the *previous* 14 days of data (never the epoch's own events):
  1. a baseline behavioral profile (typical session duration, typical
     resources/geo, typical hour-of-day) used to score cold-start entities
     and events without a full window yet, and
  2. robust calibration statistics (median/MAD) for the fallback score and
     for the LSTM's reconstruction error, so that "how anomalous is this
     raw score" is judged against *recent* normal behavior rather than a
     value fixed once at the start of the run.
The very first epoch has no prior data, so it bootstraps its baseline from
its own events (unavoidable cold start for the whole system).
"""

import numpy as np

from . import config as cfg

EPS = 1e-6


def _typical_set(value_counts, mass_threshold):
    total = value_counts.sum()
    if total == 0:
        return set()
    cumulative = 0.0
    chosen = set()
    for value, count in value_counts.items():
        chosen.add(value)
        cumulative += count / total
        if cumulative >= mass_threshold:
            break
    return chosen


def build_entity_type_profile(sub_df):
    """Baseline profile for one entity_type, from one slice of data."""
    if len(sub_df) == 0:
        return None

    log_dur = sub_df["log_session_duration"]
    hour_counts = np.bincount(sub_df["hour_float"].astype(int).clip(0, 23), minlength=24).astype(float)
    hour_hist = (hour_counts + 1.0) / (hour_counts.sum() + 24.0)  # Laplace smoothing

    return {
        "log_dur_mean": float(log_dur.mean()),
        "log_dur_std": float(log_dur.std(ddof=0)) if len(sub_df) > 1 else 1.0,
        "top_resources": _typical_set(sub_df["resource_accessed"].value_counts(), cfg.TOP_RESOURCE_MASS),
        "top_geo": _typical_set(sub_df["geo_location"].value_counts(), cfg.TOP_GEO_MASS),
        "hour_hist": hour_hist,
        "fail_rate": float((sub_df["auth_result"] == "failure").mean()),
        "n_events": len(sub_df),
    }


def fallback_raw_score(df_slice, profile):
    """Composite heuristic deviation-from-baseline score (higher = more anomalous),
    vectorized over a dataframe slice. Weights are hand-tuned, not learned --
    documented as a simplification in model_notes.md."""
    n = len(df_slice)
    if n == 0:
        return np.zeros(0)
    if profile is None or profile["n_events"] < 5:
        return np.full(n, 3.0)  # no usable baseline yet -> moderately suspicious, not extreme

    z_dur = (df_slice["log_session_duration"] - profile["log_dur_mean"]).abs() / (profile["log_dur_std"] + EPS)
    resource_novel = (~df_slice["resource_accessed"].isin(profile["top_resources"])).astype(float)
    geo_novel = (~df_slice["geo_location"].isin(profile["top_geo"])).astype(float)
    hour_idx = df_slice["hour_float"].astype(int).clip(0, 23).to_numpy()
    hour_unusual = -np.log(profile["hour_hist"][hour_idx] + EPS)
    auth_fail = (df_slice["auth_result"] == "failure").astype(float)

    return (
        0.8 * z_dur.to_numpy()
        + 2.0 * resource_novel.to_numpy()
        + 2.0 * geo_novel.to_numpy()
        + 1.0 * hour_unusual
        + 2.5 * auth_fail.to_numpy()
    )


def robust_stats(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return 0.0, 1.0
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median, mad * 1.4826 + EPS  # scale MAD to be std-equivalent under normality


def calibrate(raw_scores, median, scaled_mad):
    z = (np.asarray(raw_scores, dtype=float) - median) / scaled_mad
    return 1.0 / (1.0 + np.exp(-z))  # sigmoid -> bounded (0, 1) risk score


def trailing_epoch_mask(df, epoch, epoch_col="epoch"):
    """Rows used to (re)build the baseline/calibration that will score `epoch`.
    Epoch 0 bootstraps from itself; later epochs use the previous 2 epochs
    (14 days) only, never the target epoch's own events."""
    if epoch == 0:
        return df[epoch_col] == 0
    lo = max(0, epoch - cfg.TRAILING_EPOCHS_FOR_REFRESH)
    hi = epoch - 1
    return (df[epoch_col] >= lo) & (df[epoch_col] <= hi)
