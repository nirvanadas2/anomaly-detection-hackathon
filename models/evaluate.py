"""
Evaluation-only script. Loads scored_events.csv (produced by train.py) and
ground_truth_labels.csv (never touched during training/scoring), joins them
on event_id, and plots how well risk_score separates normal from
attack-labeled events. Saves models/score_separation.png.

Usage:
    python -m models.evaluate
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config as cfg

# --- palette (validated status colors, light surface) ----------------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
COLOR_NORMAL = "#0ca30c"     # status: good
COLOR_ANOMALY = "#d03b3b"    # status: critical


def load_scored_with_labels():
    scored = pd.read_csv(cfg.SCORED_EVENTS_PATH)
    truth = pd.read_csv(cfg.GROUND_TRUTH_PATH)
    df = scored.merge(truth[["event_id", "label"]], on="event_id", how="inner")
    df["is_anomaly"] = df["label"] != "normal"
    return df


def _style_axis(ax):
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)


def plot_separation(df, out_path=cfg.SEPARATION_PLOT_PATH):
    normal_scores = df.loc[~df["is_anomaly"], "risk_score"].to_numpy()
    anomaly_scores = df.loc[df["is_anomaly"], "risk_score"].to_numpy()
    auc = roc_auc_score(df["is_anomaly"], df["risk_score"])

    bins = np.linspace(0, 1, 41)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, facecolor=SURFACE)
    fig.subplots_adjust(hspace=0.35, top=0.86, bottom=0.09, left=0.09, right=0.97)

    ax_n, ax_a = axes
    ax_n.hist(normal_scores, bins=bins, density=True, color=COLOR_NORMAL, alpha=0.85, zorder=2)
    ax_a.hist(anomaly_scores, bins=bins, density=True, color=COLOR_ANOMALY, alpha=0.85, zorder=2)

    ax_n.axvline(np.median(normal_scores), color=INK_PRIMARY, linewidth=1.5, linestyle=(0, (3, 2)), zorder=3)
    ax_a.axvline(np.median(anomaly_scores), color=INK_PRIMARY, linewidth=1.5, linestyle=(0, (3, 2)), zorder=3)

    ax_n.set_title(f"Normal events  (n={len(normal_scores):,}, median={np.median(normal_scores):.2f})",
                    loc="left", color=COLOR_NORMAL, fontsize=11, fontweight="bold")
    ax_a.set_title(f"Ground-truth anomalies  (n={len(anomaly_scores):,}, median={np.median(anomaly_scores):.2f})",
                    loc="left", color=COLOR_ANOMALY, fontsize=11, fontweight="bold")

    for ax in axes:
        _style_axis(ax)
        ax.set_ylabel("density", color=INK_SECONDARY, fontsize=9)
    ax_a.set_xlabel("risk_score", color=INK_SECONDARY, fontsize=10)
    ax_a.set_xlim(0, 1)

    fig.suptitle("Does risk_score separate normal from anomalous events?",
                 x=0.09, ha="left", color=INK_PRIMARY, fontsize=14, fontweight="bold", y=0.97)
    fig.text(0.09, 0.905, f"ROC-AUC = {auc:.3f}  |  dashed line = median  |  evaluated against ground_truth_labels.csv (not used in training)",
              color=INK_SECONDARY, fontsize=9.5)

    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return auc


def main():
    df = load_scored_with_labels()
    auc = plot_separation(df)

    print(f"Joined {len(df)} scored events with ground truth.")
    print(f"ROC-AUC (risk_score vs is_anomaly): {auc:.4f}")
    print()
    print("risk_score summary by label:")
    print(df.groupby("label")["risk_score"].agg(["count", "mean", "median", "std"]).to_string())
    print(f"\nSaved plot to {cfg.SEPARATION_PLOT_PATH}")


if __name__ == "__main__":
    main()
