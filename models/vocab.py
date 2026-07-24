"""Categorical vocabularies. Index 0 is reserved for <UNK>/<PAD>.

Building these from the full dataset up front is treated as "knowing the
environment's schema" (the fixed catalog of resources/cities/OS strings an
org has) rather than as label/temporal leakage -- see model_notes.md. The
statistics that actually drift over time (baseline profiles, error
calibration) are recomputed on a rolling basis in baseline.py instead.
"""

CATEGORICAL_COLUMNS = [
    "resource_accessed", "auth_method", "auth_result", "protocol", "geo_location", "os", "entity_type",
]


class Vocab:
    def __init__(self, values):
        uniques = sorted(set(values))
        self.value_to_idx = {v: i + 1 for i, v in enumerate(uniques)}  # 0 = UNK
        self.size = len(uniques) + 1

    def encode(self, value):
        return self.value_to_idx.get(value, 0)

    def encode_series(self, series):
        return series.map(self.value_to_idx).fillna(0).astype(int)


def build_vocabs(df):
    return {col: Vocab(df[col]) for col in CATEGORICAL_COLUMNS}
