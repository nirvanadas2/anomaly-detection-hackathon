"""Paths and hyperparameters for the anomaly detection pipeline."""

import os

_HERE = os.path.dirname(__file__)
_ROOT = os.path.dirname(_HERE)

ACCESS_LOGS_PATH = os.path.join(_ROOT, "data_gen", "output", "access_logs.csv")
GROUND_TRUTH_PATH = os.path.join(_ROOT, "data_gen", "output", "ground_truth_labels.csv")

OUTPUT_DIR = _HERE
SCORED_EVENTS_PATH = os.path.join(OUTPUT_DIR, "scored_events.csv")
MODEL_NOTES_PATH = os.path.join(OUTPUT_DIR, "model_notes.md")
SEPARATION_PLOT_PATH = os.path.join(OUTPUT_DIR, "score_separation.png")

RANDOM_SEED = 42

# --- cold-start fallback -----------------------------------------------
MIN_HISTORY_EVENTS = 10   # entities with fewer total events than this always use the fallback profile
TOP_RESOURCE_MASS = 0.80  # baseline "typical resource set" = smallest set covering this much frequency mass
TOP_GEO_MASS = 0.80

# --- sliding-window LSTM autoencoder ------------------------------------
WINDOW_SIZE = 15           # events per sequence (within the requested 10-20 range)
EMBED_DIMS = {
    "resource_accessed": 8,
    "auth_method": 3,
    "auth_result": 2,
    "protocol": 3,
    "geo_location": 6,
    "os": 3,
    "entity_type": 2,
}
HIDDEN_DIM = 64
NUM_LSTM_LAYERS = 1
DROPOUT = 0.1

BATCH_SIZE = 256
N_EPOCHS = 8
LEARNING_RATE = 1e-3

# --- concept drift ("rolling baseline refresh every 7 simulated days") --
DRIFT_EPOCH_DAYS = 7
TRAILING_EPOCHS_FOR_REFRESH = 2  # baseline/calibration for epoch e uses the previous 2 epochs (14 days)
