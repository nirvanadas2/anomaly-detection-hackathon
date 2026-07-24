"""
Orchestrates the classification pipeline: pick the top ALERT_BUDGET_FRACTION
of events by risk_score, then let the rule-based logic in rules.py make the
final anomaly_type call for each one. risk_score only decides *which* events
get evaluated -- it plays no role in the label itself.
"""

import json

import pandas as pd

from . import config as cfg
from .data_prep import load_merged
from .rules import RULES, RuleContext

OUTPUT_COLUMNS = [
    "entity_id", "event_id", "risk_score", "anomaly_type", "explanation", "top_contributing_features",
]


def select_candidates(df, budget_fraction=cfg.ALERT_BUDGET_FRACTION):
    threshold = df["risk_score"].quantile(1 - budget_fraction)
    candidates = df[df["risk_score"] >= threshold].copy()
    return candidates.sort_values("risk_score", ascending=False), threshold


def classify_candidates(df, candidates, verbose=True):
    ctx = RuleContext(df)
    rows = []

    for candidate in candidates.itertuples(name="Candidate"):
        entity_df, idx = ctx.entity_frame_and_position(candidate.entity_id, candidate.event_id)

        anomaly_type, explanation, features = "unclassified", "", {}
        for rule_name, rule_fn in RULES:
            result = rule_fn(candidate, entity_df, idx, ctx)
            if result.matched:
                anomaly_type, explanation, features = rule_name, result.explanation, result.features
                break

        if anomaly_type == "unclassified":
            rule_names = ", ".join(name for name, _ in RULES)
            explanation = (
                f"flagged by risk_score in the top {cfg.ALERT_BUDGET_FRACTION:.0%} "
                f"(risk_score={candidate.risk_score:.3f}) but did not match any of the rule "
                f"conditions ({rule_names})"
            )
            features = {"risk_score": round(float(candidate.risk_score), 4), "score_method": candidate.score_method}

        rows.append({
            "entity_id": candidate.entity_id,
            "event_id": candidate.event_id,
            "risk_score": round(float(candidate.risk_score), 6),
            "anomaly_type": anomaly_type,
            "explanation": explanation,
            "top_contributing_features": json.dumps(features),
        })

    result_df = pd.DataFrame(rows)[OUTPUT_COLUMNS].sort_values("risk_score", ascending=False)
    if verbose:
        print("anomaly_type breakdown among alerts:")
        print(result_df["anomaly_type"].value_counts().to_string())
    return result_df


def run(verbose=True, scored_events_path=None, access_logs_path=None, output_path=None):
    df = load_merged(scored_events_path=scored_events_path, access_logs_path=access_logs_path)
    candidates, threshold = select_candidates(df)
    if verbose:
        print(f"Loaded {len(df)} events. Alert budget = top {cfg.ALERT_BUDGET_FRACTION:.0%} "
              f"-> risk_score >= {threshold:.4f} -> {len(candidates)} candidates to classify.")
    classified = classify_candidates(df, candidates, verbose=verbose)
    output_path = output_path or cfg.CLASSIFIED_ALERTS_PATH
    classified.to_csv(output_path, index=False)
    if verbose:
        print(f"Wrote {output_path}")
    return classified
