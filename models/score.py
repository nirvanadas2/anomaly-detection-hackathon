"""
End-to-end pipeline: load -> engineer features -> train LSTM autoencoder
(unsupervised, no ground truth involved) -> build rolling per-epoch
baselines/calibration -> score every event -> write scored_events.csv and
model_notes.md.
"""

import time

import numpy as np
import torch

from . import config as cfg
from .baseline import (
    build_entity_type_profile, calibrate, fallback_raw_score,
    robust_stats, trailing_epoch_mask,
)
from .data_prep import NUMERIC_COLUMNS, engineer_features, load_raw
from .sequence_model import (
    LSTMAutoencoder, WindowDataset, build_windows,
    last_step_reconstruction_errors, train_autoencoder,
)
from .vocab import CATEGORICAL_COLUMNS, build_vocabs

torch.manual_seed(cfg.RANDOM_SEED)
np.random.seed(cfg.RANDOM_SEED)


def run_pipeline(verbose=True, access_logs_path=None):
    t0 = time.time()
    if verbose:
        print("Loading and parsing access_logs.csv ...")
    df_raw = load_raw(path=access_logs_path or cfg.ACCESS_LOGS_PATH)
    df, start_date = engineer_features(df_raw)

    if verbose:
        n_cold = df.loc[df["is_cold_start_entity"], "entity_id"].nunique()
        print(f"  {len(df)} events, {df['entity_id'].nunique()} entities "
              f"({n_cold} cold-start: fewer than {cfg.MIN_HISTORY_EVENTS} events)")

    vocabs = build_vocabs(df)
    cat_arrays = {col: vocabs[col].encode_series(df[col]).to_numpy() for col in CATEGORICAL_COLUMNS}
    numeric_raw = df[NUMERIC_COLUMNS].to_numpy(dtype=np.float32)

    eligible_mask = df["use_lstm"].to_numpy()
    scaler_mean = numeric_raw[eligible_mask].mean(axis=0)
    scaler_std = numeric_raw[eligible_mask].std(axis=0) + 1e-6
    numeric_scaled = ((numeric_raw - scaler_mean) / scaler_std).astype(np.float32)

    eligible_entity_ids = set(df.loc[eligible_mask, "entity_id"].unique())
    if verbose:
        print(f"  {len(eligible_entity_ids)} entities have enough history for the LSTM path "
              f"(window={cfg.WINDOW_SIZE})")

    cat_windows, numeric_windows, target_index = build_windows(
        df, cat_arrays, numeric_scaled, eligible_entity_ids,
    )
    n_windows = numeric_windows.shape[0]
    if verbose:
        print(f"  built {n_windows} training windows")

    model = LSTMAutoencoder(vocabs, cfg.EMBED_DIMS, numeric_dim=numeric_scaled.shape[1])
    dataset = WindowDataset(cat_windows, numeric_windows)

    if verbose:
        print(f"Training LSTM autoencoder for {cfg.N_EPOCHS} epochs ...")
    loss_history = train_autoencoder(model, dataset, verbose=verbose)

    raw_errors = last_step_reconstruction_errors(model, cat_windows, numeric_windows)
    df["lstm_raw_error"] = np.nan
    df.loc[target_index, "lstm_raw_error"] = raw_errors

    if verbose:
        print("Building rolling per-epoch baselines and calibration (drift handling) ...")
    df["risk_score"] = np.nan
    df["score_method"] = ""
    profiles = {}
    fallback_calib, lstm_calib = {}, {}

    for epoch in sorted(df["epoch"].unique()):
        trailing = df[trailing_epoch_mask(df, epoch)]
        for entity_type in df["entity_type"].unique():
            key = (epoch, entity_type)
            trailing_sub = trailing[trailing["entity_type"] == entity_type]
            profile = build_entity_type_profile(trailing_sub)
            profiles[key] = profile

            fb_scores = fallback_raw_score(trailing_sub, profile)
            fallback_calib[key] = robust_stats(fb_scores)

            lstm_errs = trailing_sub.loc[trailing_sub["use_lstm"], "lstm_raw_error"].dropna().to_numpy()
            lstm_calib[key] = robust_stats(lstm_errs) if len(lstm_errs) else (0.0, 1.0)

    for (epoch, entity_type), idx in df.groupby(["epoch", "entity_type"]).groups.items():
        group = df.loc[idx]
        key = (epoch, entity_type)
        profile = profiles[key]

        fb_median, fb_mad = fallback_calib[key]
        fb_raw = fallback_raw_score(group, profile)
        fb_risk = calibrate(fb_raw, fb_median, fb_mad)

        l_median, l_mad = lstm_calib[key]
        l_raw = group["lstm_raw_error"].to_numpy()
        l_risk = calibrate(np.nan_to_num(l_raw, nan=l_median), l_median, l_mad)

        use_lstm = group["use_lstm"].to_numpy()
        final_risk = np.where(use_lstm, l_risk, fb_risk)
        method = np.where(use_lstm, "lstm", "baseline_fallback")

        df.loc[idx, "risk_score"] = final_risk
        df.loc[idx, "score_method"] = method

    stats = {
        "total_events": len(df),
        "n_entities": df["entity_id"].nunique(),
        "n_cold_start_entities": int(df.loc[df["is_cold_start_entity"], "entity_id"].nunique()),
        "n_windows_trained": n_windows,
        "n_epochs_trained": cfg.N_EPOCHS,
        "final_train_loss": loss_history[-1] if loss_history else float("nan"),
        "n_scored_lstm": int((df["score_method"] == "lstm").sum()),
        "n_scored_fallback": int((df["score_method"] == "baseline_fallback").sum()),
        "n_drift_epochs": df["epoch"].nunique(),
        "runtime_seconds": time.time() - t0,
    }
    if verbose:
        print(f"Done in {stats['runtime_seconds']:.1f}s. "
              f"{stats['n_scored_lstm']} events scored via LSTM, "
              f"{stats['n_scored_fallback']} via baseline fallback.")

    return df, stats
