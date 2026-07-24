# Model Notes

Plain-English explanation of the anomaly detection pipeline in `models/`,
which reads `data_gen/output/access_logs.csv` and produces `scored_events.csv`.
It never reads `ground_truth_labels.csv` -- that file is only used afterwards,
by `evaluate.py`, to check how well the unsupervised risk scores line up with
the known attack labels.

## 1. Feature engineering

Each raw row is turned into:

- **Categorical fields** (embedded by the model): `resource_accessed`,
  `auth_method`, `auth_result`, `protocol` and `os` (both pulled out of the
  `device_fingerprint` JSON), `geo_location`, and `entity_type`.
- **Engineered numeric fields**: log-transformed `session_duration`;
  cyclical (`sin`/`cos`) encodings of hour-of-day and day-of-week; the
  length of `command_sequence` (log-transformed) and a `has_commands` flag;
  the log-transformed minutes since that same entity's previous event
  (captures burst/tight-interval behavior, relevant to brute force); and
  four **novelty flags** -- is this the first time this entity has used
  this `source_ip` / `mac_address` / `resource_accessed` / `geo_location`,
  computed by walking each entity's history strictly forward in time so no
  feature ever depends on a future event.

`command_sequence` and `device_fingerprint` are parsed with `json.loads`
from their packed string columns before any of this.

## 2. Cold-start fallback (per-entity-type baseline)

New entities don't have enough personal history to model individually. Any
entity with fewer than 10 total events in the dataset -- and any
event that is among the *first* 14 events of a
longer-lived entity, before it has a full sequence window yet -- is scored
against a **per-entity-type baseline profile** instead of the LSTM:

- typical session duration (mean/std of the log-transformed value),
- a "typical resource set" (the smallest set of resources covering
  80% of that entity type's access frequency),
- a "typical geo set" (same idea, 80% coverage),
- an hour-of-day histogram (Laplace-smoothed),
- and the auth failure rate.

An event's fallback raw score is a hand-weighted sum of: how many standard
deviations its session duration is from typical, whether its resource is
outside the typical set, whether its location is outside the typical set,
how statistically unusual its hour-of-day is, and whether the auth attempt
failed. This is a heuristic, not a learned model -- appropriate for entities
we don't yet have enough data to model any other way.

## 3. LSTM autoencoder (for entities with enough history)

Once an entity has passed the cold-start threshold *and* accumulated a full
window of 15 events, its events are scored by a sequence
autoencoder trained in PyTorch:

1. Each categorical field is embedded; embeddings are concatenated with the
   scaled numeric features at every timestep.
2. An encoder LSTM consumes the 15-event window and its final
   hidden state becomes a single latent vector summarizing the sequence.
3. A decoder LSTM re-expands that latent vector across all
   15 timesteps, and a linear layer projects each timestep back
   into the embedded+numeric feature space.
4. The model is trained (unsupervised -- no labels used) to minimize
   reconstruction error (MSE) across the whole window.
5. At scoring time, only the reconstruction error of the window's *last*
   timestep is kept -- that is the model's "surprise" at the specific event
   the window ends on, and it becomes that event's raw LSTM score. Sliding
   the window forward one event at a time means every eligible event gets
   scored exactly once, as the newest event in its own window.

This run trained on **43808 windows** for
**8 epochs**, reaching a final training reconstruction MSE
of **0.1622**.

Of 48008 total events across 300 entities
(0 of them cold-start), **43808** were
scored by the LSTM and **4200** by the baseline fallback.

## 4. Turning a raw score into `risk_score`

The fallback heuristic and the LSTM's reconstruction error live on
completely different numeric scales, and neither is naturally bounded. Both
are converted to a comparable **risk_score in (0, 1)** the same way: a
robust z-score against a reference distribution
(`(raw - median) / (1.4826 * MAD)`, MAD = median absolute deviation, chosen
over mean/std for its resistance to the very outliers we're trying to
detect), then squashed through a sigmoid. 0.5 means "typical for its
reference distribution"; values pushed toward 1.0 are increasingly
surprising.

## 5. Concept drift: rolling baseline refresh every 7 days

Both the per-entity-type baseline profile (section 2) and the reference
distribution used to calibrate raw scores into `risk_score` (section 4) are
**recomputed every 7 simulated days**, using only the
previous 14 days of data -- never the events they're about to
score. This means "what counts as normal" drifts forward with the data
instead of staying frozen at whatever the first week looked like: if
legitimate behavior gradually shifts (e.g. a resource migration or a new
typical login window), the baseline and the calibration reference both
catch up within about two weeks, instead of the pipeline slowly
accumulating false positives against a stale definition of "normal".

The very first 7-day period has no prior data to draw on,
so it bootstraps its own baseline/calibration from itself -- an unavoidable
cold start for the whole system, separate from the per-entity cold start in
section 2.

The LSTM's *weights* are trained once, offline, on the full history -- a
deliberate scope simplification for this hackathon prototype. Only the
statistics used to interpret its output (and the fallback baseline) are
kept rolling. A production version would likely fine-tune the network
periodically too.

## 6. Output

`scored_events.csv` has `entity_id`, `event_id`, `timestamp`, `risk_score`,
plus one extra diagnostic column, `score_method` (`lstm` or
`baseline_fallback`), so it's possible to tell which scoring path produced
each row.

## 7. Known simplifications (hackathon scope)

- The heuristic fallback score's feature weights are hand-tuned, not fit to
  data.
- Categorical vocabularies (the set of possible resources/cities/OS
  strings/etc.) are built once from the full dataset up front. This is
  treated as "knowing the environment's schema" rather than temporal
  leakage, since it doesn't use any per-event ordering or outcome
  information -- but it is a simplification worth naming.
- The LSTM reconstructs the embedded+numeric feature space directly (a
  regression-style loss) rather than predicting each categorical field
  through its own softmax/cross-entropy head, which is simpler to train
  and score but slightly less precise about which field drove the error.
- No hyperparameter search was performed; window size, hidden size, and
  embedding dimensions were chosen to be small enough to train quickly on
  CPU for a hackathon dataset of this size.
