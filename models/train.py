"""
Entry point: run the full pipeline and write scored_events.csv + model_notes.md.

Usage:
    python -m models.train
"""

from . import config as cfg
from . import docs
from .score import run_pipeline

OUTPUT_COLUMNS = ["entity_id", "event_id", "timestamp", "risk_score", "score_method"]


def main():
    df, stats = run_pipeline(verbose=True)

    scored = df[OUTPUT_COLUMNS].sort_values("timestamp")
    scored.to_csv(cfg.SCORED_EVENTS_PATH, index=False)

    notes = docs.render(
        min_history=cfg.MIN_HISTORY_EVENTS,
        window_minus_one=cfg.WINDOW_SIZE - 1,
        window_size=cfg.WINDOW_SIZE,
        top_resource_mass=cfg.TOP_RESOURCE_MASS,
        top_geo_mass=cfg.TOP_GEO_MASS,
        drift_epoch_days=cfg.DRIFT_EPOCH_DAYS,
        trailing_days=cfg.TRAILING_EPOCHS_FOR_REFRESH * cfg.DRIFT_EPOCH_DAYS,
        **stats,
    )
    with open(cfg.MODEL_NOTES_PATH, "w", encoding="utf-8") as f:
        f.write(notes)

    print(f"\nWrote {cfg.SCORED_EVENTS_PATH}")
    print(f"Wrote {cfg.MODEL_NOTES_PATH}")


if __name__ == "__main__":
    main()
