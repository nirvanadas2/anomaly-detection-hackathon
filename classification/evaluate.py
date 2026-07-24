"""
Scores classified_alerts.csv against data_gen/output/ground_truth_labels.csv
-- the only place in this module ground truth is read. Prints
precision/recall/F1 per anomaly_type and false positive rate at the top-1%
alert budget.

Usage:
    python -m classification.evaluate
"""

import pandas as pd

from . import config as cfg

TYPE_TO_TRUE_LABEL = {
    "brute_force": "anomaly_brute_force",
    "impossible_travel": "anomaly_impossible_travel",
    "lateral_movement": "anomaly_lateral_movement",
    "credential_misuse": "anomaly_credential_misuse",
    "device_spoofing": "anomaly_device_spoofing",
    "credential_stuffing": "anomaly_credential_stuffing",
}


def load_alerts_with_truth():
    alerts = pd.read_csv(cfg.CLASSIFIED_ALERTS_PATH)
    truth = pd.read_csv(cfg.GROUND_TRUTH_PATH)
    merged = alerts.merge(truth[["entity_id", "event_id", "label"]], on=["entity_id", "event_id"], how="left")
    return merged, truth


def per_type_metrics(merged, truth_label_counts):
    rows = []
    for anomaly_type, true_label in TYPE_TO_TRUE_LABEL.items():
        predicted_mask = merged["anomaly_type"] == anomaly_type
        n_predicted = int(predicted_mask.sum())
        tp = int((predicted_mask & (merged["label"] == true_label)).sum())
        fp = n_predicted - tp
        n_true_total = int(truth_label_counts.get(true_label, 0))

        precision = tp / n_predicted if n_predicted else float("nan")
        recall = tp / n_true_total if n_true_total else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) and precision == precision and recall == recall and (precision + recall) > 0
              else float("nan"))

        rows.append({
            "anomaly_type": anomaly_type,
            "n_predicted": n_predicted,
            "true_positives": tp,
            "false_positives": fp,
            "n_true_total_in_dataset": n_true_total,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    return pd.DataFrame(rows)


def budget_false_positive_rates(merged, truth_label_counts):
    n_alerts = len(merged)
    n_alerts_normal = int((merged["label"] == "normal").sum())
    n_total_normal = int(truth_label_counts.get("normal", 0))

    return {
        "n_alerts": n_alerts,
        "n_alerts_that_are_normal": n_alerts_normal,
        # fraction of the alert queue that is a false alarm -- what a SOC analyst means by
        # "false positive rate" when talking about an alert budget.
        "alert_false_positive_rate": n_alerts_normal / n_alerts if n_alerts else float("nan"),
        # the stricter statistical definition: FP / (FP + TN), i.e. what fraction of ALL
        # normal events in the dataset got swept into this alert budget.
        "statistical_fpr_over_all_normal_events": n_alerts_normal / n_total_normal if n_total_normal else float("nan"),
    }


def main():
    merged, truth = load_alerts_with_truth()
    truth_label_counts = truth["label"].value_counts()

    metrics = per_type_metrics(merged, truth_label_counts)
    fpr = budget_false_positive_rates(merged, truth_label_counts)

    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(f"Alert budget: top {cfg.ALERT_BUDGET_FRACTION:.0%} of events by risk_score "
          f"({fpr['n_alerts']} alerts)\n")
    print("Precision / recall / F1 per anomaly_type (recall is against ALL true events of that")
    print("type in the full dataset, including any the risk_score budget missed entirely):\n")
    print(metrics.to_string(index=False))

    print(f"\nUnclassified alerts (flagged by risk_score, no rule matched): "
          f"{(merged['anomaly_type'] == 'unclassified').sum()}")

    print("\nFalse positive rate at the top-1% alert budget:")
    print(f"  alerts that are actually normal:            {fpr['n_alerts_that_are_normal']} / {fpr['n_alerts']}"
          f"  = {fpr['alert_false_positive_rate']:.3f}  (fraction of the alert queue that's a false alarm)")
    print(f"  of ALL normal events in the dataset:         "
          f"{fpr['statistical_fpr_over_all_normal_events']:.4f}  (fraction of normal traffic that got alerted)")


if __name__ == "__main__":
    main()
