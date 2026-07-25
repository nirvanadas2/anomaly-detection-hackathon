"""
run_pipeline.py

Single entry point that runs the whole project end to end, in the order the
pieces actually depend on each other:

    1. data_gen.generate    -- synthesize access_logs.csv + ground_truth_labels.csv
    2. models.train          -- LSTM autoencoder + baseline-fallback risk scoring
    3. models.evaluate       -- risk_score vs ground truth (ROC-AUC, separation plot)
    4. classification.main   -- rule-based alert classification + precision/recall/F1

Each stage calls the exact same functions the module's own `python -m
<module>` entry point calls -- this script does not reimplement any of
them, it just sequences them and prints a labeled progress banner before
each one, then rolls the key numbers up into one final summary.

demo_insider_drift.py (concept-drift stress test) and demo_streaming.py
(real-time-feasibility replay) are deliberately NOT part of this run: they
are meant to be run afterward, standalone, once this canonical pipeline's
outputs exist. demo_insider_drift.py writes to its own isolated output
directory; demo_streaming.py only reads models/scored_events.csv and
prints to the console. See the pointer printed at the end of this script.

Usage:
    python run_pipeline.py
"""

import os
import time

import pandas as pd

from classification import classify as clf_classify
from classification import config as clf_cfg
from classification import evaluate as clf_evaluate
from data_gen import generate as dg_generate
from models import config as models_cfg
from models import evaluate as models_evaluate
from models import train as models_train

ROOT = os.path.dirname(os.path.abspath(__file__))


def _banner(title):
    print()
    print("#" * 100)
    print(f"# {title}")
    print("#" * 100)
    print()


# ---------------------------------------------------------------------------
# Stage 1: data_gen.generate
# ---------------------------------------------------------------------------
def stage_generate():
    _banner("STAGE 1/4 -- data_gen.generate: synthesizing the access-log dataset")

    out_dir = os.path.join(os.path.dirname(dg_generate.__file__), "output")
    df, stats = dg_generate.build_dataset()
    paths = dg_generate.write_outputs(df, stats, out_dir)

    print(f"Wrote {stats['total_events']} events "
          f"({stats['n_normal']} normal, {stats['n_anomaly']} anomaly, "
          f"{stats['anomaly_pct']:.2f}% anomalous) to:")
    for p in paths:
        print(f"  - {p}")
    print("\nLabel distribution:")
    print(df["label"].value_counts().to_string())

    return df, stats


# ---------------------------------------------------------------------------
# Stage 2: models.train
# ---------------------------------------------------------------------------
def stage_train():
    _banner("STAGE 2/4 -- models.train: training the LSTM autoencoder + "
            "baseline fallback, scoring every event")
    models_train.main()


# ---------------------------------------------------------------------------
# Stage 3: models.evaluate
# ---------------------------------------------------------------------------
def stage_evaluate_scores():
    _banner("STAGE 3/4 -- models.evaluate: checking risk_score separation "
            "against ground truth")

    df_eval = models_evaluate.load_scored_with_labels()
    auc = models_evaluate.plot_separation(df_eval)

    print(f"Joined {len(df_eval)} scored events with ground truth.")
    print(f"ROC-AUC (risk_score vs is_anomaly): {auc:.4f}")
    print()
    print("risk_score summary by label:")
    print(df_eval.groupby("label")["risk_score"].agg(["count", "mean", "median", "std"]).to_string())
    print(f"\nSaved plot to {models_cfg.SEPARATION_PLOT_PATH}")

    return auc


# ---------------------------------------------------------------------------
# Stage 4: classification.main
# ---------------------------------------------------------------------------
def stage_classify():
    _banner("STAGE 4/4 -- classification.main: rule-based alert classification "
            "+ precision/recall/F1 evaluation")

    clf_classify.run(verbose=True)

    print()
    merged, truth = clf_evaluate.load_alerts_with_truth()
    truth_label_counts = truth["label"].value_counts()
    metrics = clf_evaluate.per_type_metrics(merged, truth_label_counts)
    fpr = clf_evaluate.budget_false_positive_rates(merged, truth_label_counts)

    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(f"Alert budget: top {clf_cfg.ALERT_BUDGET_FRACTION:.0%} of events by risk_score "
          f"({fpr['n_alerts']} alerts)\n")
    print("Precision / recall / F1 per anomaly_type:")
    print(metrics.to_string(index=False))
    print(f"\nUnclassified alerts (flagged by risk_score, no rule matched): "
          f"{(merged['anomaly_type'] == 'unclassified').sum()}")
    print("\nFalse positive rate at the top-1% alert budget:")
    print(f"  alerts that are actually normal: {fpr['n_alerts_that_are_normal']} / {fpr['n_alerts']}"
          f"  = {fpr['alert_false_positive_rate']:.3f}")
    print(f"  of ALL normal events in the dataset: {fpr['statistical_fpr_over_all_normal_events']:.4f}")

    return metrics, fpr


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
def print_final_summary(gen_stats, label_counts, auc, metrics, fpr, elapsed_seconds):
    _banner("PIPELINE COMPLETE -- SUMMARY")

    print(f"Dataset:  {gen_stats['total_events']:,} events across {gen_stats['n_entities']} entities "
          f"over {gen_stats['n_days']} days ({gen_stats['n_normal']:,} normal, "
          f"{gen_stats['n_anomaly']:,} anomaly, {gen_stats['anomaly_pct']:.2f}% anomalous)")

    print("\nLabel distribution (all 6 anomaly types + normal):")
    print(label_counts.to_string())

    print(f"\nROC-AUC (risk_score vs ground-truth anomaly label): {auc:.4f}")

    print("\nPrecision / recall / F1 per anomaly type:")
    print(metrics.to_string(index=False))

    print(f"\nAlert-queue false positive rate (top {clf_cfg.ALERT_BUDGET_FRACTION:.0%} budget, "
          f"{fpr['n_alerts']} alerts):")
    print(f"  {fpr['n_alerts_that_are_normal']} / {fpr['n_alerts']} alerts are false alarms "
          f"= {fpr['alert_false_positive_rate']:.1%} of the alert queue")
    print(f"  {fpr['statistical_fpr_over_all_normal_events']:.2%} of ALL normal events in the "
          f"dataset got swept into the alert queue")

    print(f"\nTotal pipeline wall-clock time: {elapsed_seconds:.1f}s")

    print("\n" + "-" * 100)
    print("Next steps -- two standalone demo scripts, neither of which touches the canonical")
    print("outputs this run just produced (demo_insider_drift.py writes to its own isolated")
    print("output directory; demo_streaming.py only reads and prints, it writes nothing):")
    print("  python demo_insider_drift.py   -- concept-drift stress test: does gradual, genuinely")
    print("                                     normal behavior growth get mistaken for an attack?")
    print("  python demo_streaming.py       -- real-time replay of access_logs.csv with live")
    print("                                     per-event scoring latency, to demonstrate streaming")
    print("                                     deployment feasibility.")
    print("  streamlit run dashboard/app.py -- interactive dashboard over this run's alert output.")
    print("-" * 100)


def main():
    t0 = time.time()

    df, gen_stats = stage_generate()
    stage_train()
    auc = stage_evaluate_scores()
    metrics, fpr = stage_classify()

    elapsed = time.time() - t0
    print_final_summary(gen_stats, df["label"].value_counts(), auc, metrics, fpr, elapsed)


if __name__ == "__main__":
    main()
