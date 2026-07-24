"""Holds the (long) text body of data_assumptions.md as a template string."""

TEMPLATE = """# Data Assumptions & Generation Logic

This document explains, in plain English, every distributional assumption,
baseline definition, and attack-injection rule used by `data_gen/generate.py`
to build `access_logs.csv` and `ground_truth_labels.csv`. It's meant to be
read by anyone judging or extending this hackathon submission.

## 1. Scale & reproducibility

- **{n_entities} entities** over a **{n_days}-day window** starting `{start_date}`.
- Random seed fixed at `{seed}` (numpy Generator + Faker), so the run is
  reproducible byte-for-byte given the same code.
- This run produced **{total_events} events** total: **{n_normal} normal**
  and **{n_anomaly} injected attack events** (**{anomaly_pct:.2f}%** of all
  events), within the requested 0.5%-3% injection-rate band
  (actual rate used this run: **{attack_rate:.2%}**, drawn uniformly from
  that range once at the start of the run).

## 2. Entity population

Entities are split into three types by weighted random draw:

| entity_type       | share |
|--------------------|-------|
| user               | 70%   |
| service_account    | 20%   |
| edge_device        | 10%   |

Entity IDs are prefixed by type: `USR-000001`, `SVC-000001`, `DEV-000001`.

A fraction of each type is flagged **privileged** at creation time
(users 15%, service accounts 50%, edge devices 0%). This flag is the single
switch that controls whether an entity's sessions ever carry a
`command_sequence` — non-privileged entities always have `command_sequence = []`.
Privileged entities are also allowed to have tier-0 ("crown jewel") resources
in their *normal* resource set, modeling legitimate admins/automation.

## 3. Per-entity behavioral baseline

Every entity gets a fixed baseline drawn once, which all of its *normal*
events are sampled from:

- **Typical login hours**: modeled as a `(center, spread)` on a 24h clock,
  sampled per event as `normal(center, spread) mod 24`.
  - Users: center drawn uniformly in the 8am-5pm range (plus an 8% chance of
    being a night-shift worker centered ~10pm-2am), spread 1.5-3 hours.
  - Service accounts: center drawn uniformly across the full day (cron-like
    scheduling), tight spread of 0.3-1.5 hours (automation is punctual).
  - Edge devices: effectively uniform across 24h (spread=24h), modeling
    round-the-clock heartbeats/telemetry.
- **Typical geo location**: one "home city" chosen from a 40-city global
  list (with lat/lon for distance math). 10% of *users* additionally get a
  "secondary city" (e.g. a commuter with two offices). On any given active
  day, an entity picks **one** location for the *entire* day (20% chance of
  using the secondary city if one exists) — never two locations on the same
  day — specifically so ordinary dual-office habits can never accidentally
  look like impossible travel in the ground truth.
- **Typical resource set**: a fixed random subset of a 27-resource catalog,
  split into three sensitivity tiers (tier 0 = critical/admin, e.g.
  `domain-controller-01`, `secrets-vault`; tier 1 = internal/sensitive, e.g.
  `source-code-repo`, `hr-portal`; tier 2 = general purpose, e.g. `email`,
  `chat-app`). Users get 3-8 resources, service accounts 1-4, edge devices
  1-2 (drawn only from IoT-relevant endpoints). Normal events sample
  uniformly from this fixed set.
- **Typical session duration**: a per-entity `(mean, std)` in minutes, drawn
  once. Users ~8-40 min mean, service accounts ~0.5-4 min (short automated
  jobs), edge devices ~0.1-1 min (brief heartbeats). Each normal event's
  duration is `max(0.05, normal(mean, std))`.
- **Typical auth method**: a per-entity-type weighted preference (users lean
  password/biometric, service accounts lean token/certificate, edge devices
  lean certificate), sampled per event from those weights.
- **Device fingerprint**: OS/firmware, MAC address are fixed per entity for
  the whole run (representing "their device"). The protocol field varies per
  event based on the sensitivity tier of the resource being accessed (tier 0
  favors SSH/RDP/VPN, tier 1 HTTPS/SSH/LDAP, tier 2 HTTPS/MQTT/SNMPv3),
  reflecting that admin systems are reached over different protocols than
  everyday apps.
- **Source IP(s)**: 1-2 private IPs generated per entity (Faker), standing in
  for a home/office network. A used secondary-city IP is generated lazily the
  first time it's needed and then reused.
- **Event volume**: each entity is "active" on a given day with probability
  `active_prob` (users 45-80%, service accounts 85-100%, edge devices
  95-100%), and if active, the day's event count is drawn from
  `Poisson(events_per_day_lambda)` (users 1.5-4.5/day, service accounts
  4-14/day, edge devices 3-8/day).
- **Command sequence** (privileged entities only): 2-6 commands sampled
  without replacement from a fixed benign command pool (`ls`, `git pull`,
  `kubectl get pods`, etc.), stored as an ordered JSON list.
- **Ordinary auth failures**: 2% of normal events are a plain mistyped
  password / expired token, not an attack — `auth_result = "failure"` with
  `session_duration = 0`. This is separate from the brute-force injection
  below and is labeled `normal`.

An extra field, **`auth_result`** (`success`/`failure`), was added beyond the
minimum requested schema because it's the only way to meaningfully represent
"a burst of *failed* auth attempts" for the brute-force pattern.

## 4. Attack injection

A single attack rate is drawn once per run, uniformly from **0.5%-3%** of
total event volume, then attack *incidents* are generated round-robin across
all **six** attack types (each incident produces a variable number of
events) until that event budget is met or exceeded. This run used
**{attack_rate:.2%}**. Because the round-robin cycles evenly across six
generators instead of three, adding credential_misuse, device_spoofing, and
credential_stuffing automatically shrinks each individual type's share of
the fixed total budget rather than growing the total — the 0.5%-3% ceiling
is enforced on the combined total, never per-type. credential_stuffing
incidents are the largest per-incident of any type (15-40 events each, vs.
1-30 for everything else), so in practice it ends up with a noticeably
larger share of the total budget than its "1 of 6" turn share would
suggest — expected, and still bounded by the same overall ceiling.

### 4.1 brute_force
- Pick one random target entity.
- Pick one attacker source IP (a public/external IP, via Faker) and one
  attacker device fingerprint (unfamiliar OS/MAC) — distinct from the
  entity's own baseline device and IPs.
- Pick one attacker "location" (a city different from the entity's home
  city) — held constant for the whole burst.
- Generate 5-30 auth attempts, each 1-15 seconds apart, auth method
  re-rolled per attempt (attacker trying different credentials/methods),
  target resource drawn from the entity's typical resource set.
- All attempts are `auth_result = "failure"` and `session_duration = 0`,
  **except** a 20% chance that the *final* attempt succeeds (a compromised
  account), which gets a short real session duration. Every event in the
  burst — including a successful final one — is labeled
  `anomaly_brute_force`.

### 4.2 impossible_travel
- Pick one random entity and its home city.
- Draw a time gap `delta_hours` uniformly from 0.25-3 hours.
- Choose a second city such that
  `haversine_distance(home, city) / delta_hours > 900 km/h * 1.2` — i.e. the
  implied travel speed is at least 20% past the fastest plausible
  point-to-point commercial travel speed (900 km/h, itself already generous
  versus a ~880 km/h cruise speed, to account for connections). This
  guarantees genuine physical impossibility rather than "just suspicious."
- Emit two events: a normal, baseline-consistent login at the home city
  (labeled `normal`), then `delta_hours` later a login from the far city
  using a new external IP but the **entity's own device fingerprint and
  usual resource/auth preferences** (labeled `anomaly_impossible_travel`).
  Only geography and timing are anomalous — every other field looks
  legitimate, which is intentional: it isolates travel-speed as the sole
  detection signal for this pattern, rather than stacking multiple giveaways.

### 4.3 lateral_movement
- Pick one random entity and compute its typical-resource set.
- Build a 4-10 step sequence, seconds/minutes apart (30-300s between steps),
  that walks through resources **outside** that entity's typical set, in an
  escalating pattern: the first ~40% of steps hit tier-2 (general) resources
  the entity doesn't normally use, the next ~35% hit tier-1 (internal/
  sensitive), and the final ~25% culminate in tier-0 (critical/admin, e.g.
  `domain-controller-01`, `secrets-vault`).
- Source IP/device fingerprint/location are kept as the entity's own normal
  baseline (models an already-authenticated insider or a session riding on
  stolen-but-valid credentials — the anomaly signal is purely the resource
  *sequence*, not where it's coming from).
- If the entity is privileged, `command_sequence` escalates too, stepping
  through six fixed stages of a recon -> credential-dump -> pivot -> exfil
  chain (`whoami` -> `net user` -> `mimikatz.exe ...` -> `psexec \\\\dc01 cmd`
  -> `net use z: ...` -> `7z a exfil.zip ...`), chosen by how far along the
  sequence the current step is. Non-privileged entities keep
  `command_sequence = []` for these events too, per the rule in section 2.
- All events in the sequence are labeled `anomaly_lateral_movement`.

### 4.4 credential_misuse
- Pick one random target entity, restricted to `user`/`service_account`
  types (an edge device doesn't have a "stolen personal credential" in the
  same sense).
- Pick an **off-hours time**: the opposite side of the 24h clock from the
  entity's typical login-hour center (`center + 12h`, jittered ±1.5h) —
  deliberately the *farthest* point from its normal window, not just
  "a bit late."
- Pick an **unfamiliar location**: a city that is neither the entity's home
  city nor its secondary city (if it has one).
- Pick a **new device fingerprint**: a fresh MAC address, and an OS/firmware
  string drawn from the *same entity type's* OS pool but different from the
  entity's own (a plausible machine, just not this entity's usual one).
- Emit 1-3 events, 30s-4min apart, all sharing that off-hours time / new
  location / new device, auth method re-rolled per event (not the entity's
  usual preference), session durations short (20-60% of the entity's normal
  mean — someone moving quickly and unfamiliar with the environment).
- If the entity is privileged, `command_sequence` is drawn from a dedicated
  enumeration/probing command pool (`whoami /all`, `net user administrator`,
  `netstat -ano`, etc.) — a different flavor from both the benign pool and
  the lateral-movement escalation stages, meant to read as "someone getting
  their bearings," not a multi-stage intrusion.
- All events are labeled `anomaly_credential_misuse`. Unlike
  impossible_travel, this pattern deliberately does **not** check
  geo-velocity against a preceding event — it's a standalone unfamiliar
  login, not a physically-impossible pair.

### 4.5 device_spoofing
- Pick one random entity and build one session of 3-6 events, 20s-3min
  apart, at the entity's **normal** home city and normal auth method (time
  and location are intentionally left alone so the only anomalous signal is
  device identity).
- The first half of the session (at least 1 event) uses the entity's real,
  baseline device fingerprint and is labeled `normal`.
- From the midpoint on, the fingerprint switches: a new MAC address, and an
  OS/firmware string drawn from *another entity type's* OS pool entirely
  (e.g. a user session suddenly reporting IoT firmware, or an edge device
  reporting "macOS 14 Sonoma") — not just unfamiliar but structurally
  inconsistent with anything this entity could legitimately be. These
  events are labeled `anomaly_device_spoofing`.
- Resource pattern and command_sequence (privileged entities: benign pool,
  same as normal sessions) are kept normal throughout, for the same reason
  impossible_travel and lateral_movement isolate their own signal — so the
  device-fingerprint mismatch is the only thing distinguishing the second
  half of the session from the first.

### 4.6 credential_stuffing
- The classic stolen-credential-list attack: breadth across many accounts
  from a small pool of sources, not repeated attempts against one account
  (that's brute_force — see 4.1). Generated as a single incident, not a
  per-entity pattern.
- Pick 2-4 attacker source IPs (Faker public IPs) for the whole incident.
  Each IP gets its own fixed device fingerprint (MAC + `"Unknown"` OS) and
  its own "location" (a random city), modeling a small botnet/proxy pool
  rather than one machine.
- Pick 15-40 distinct target entities at random from the whole population
  (sampled without replacement) — deliberately entity-agnostic; this attack
  doesn't care who it hits, unlike every other pattern in this generator,
  which targets one entity at a time.
- Pick a single short time window, 10-30 minutes.
- Each targeted entity gets exactly **one** login attempt (breadth over
  depth — a real credential-stuffing run tries each leaked credential once
  and moves on, not repeated retries per account), at a random moment in
  the window, from one of the incident's 2-4 IPs (chosen independently per
  target, so IPs get reused across many different entities — that reuse is
  the signal), against a resource drawn from that entity's own typical set,
  with a random auth method.
- One failure rate is drawn once for the whole incident, uniformly from
  5-15% *success* (i.e. 85-95% failure) — "a few" credentials happen to
  still be valid, most don't.
- All events are labeled `anomaly_credential_stuffing`.

## 5. Output files

- **`access_logs.csv`** — every column *except* `label` (this is the file a
  detection model or pipeline should actually run against).
- **`ground_truth_labels.csv`** — `entity_id`, `event_id`, `label`, kept
  separate specifically so labels can be withheld at inference time and
  joined back in only for scoring.
- **`data_assumptions.md`** — this file.

`event_id` is assigned only after all normal and attack events are merged
and sorted by timestamp (`EVT-0000001`, `EVT-0000002`, ...), so IDs are
chronological across the whole dataset regardless of which generator
produced the row.

## 6. Known simplifications (hackathon scope)

- `geo_location` is stored as `"City, Country"` text; the underlying
  lat/lon table used for the impossible-travel physics check is internal to
  the generator and not exposed as output columns.
- Attack incidents are independent of each other (no simulated multi-stage
  campaigns spanning attack types), and target entities are chosen uniformly
  at random rather than being weighted toward "more valuable" targets.
- The 900 km/h impossible-travel threshold is a single global constant; it
  doesn't account for e.g. supersonic travel or account sharing, which real
  systems sometimes special-case.
- credential_misuse's "new device" draws from the entity's own type's OS
  pool (a plausible laptop, just not theirs), while device_spoofing
  deliberately draws from a foreign type's OS pool (a fingerprint with no
  legitimate explanation at all) — a deliberate difference in severity
  between the two patterns, not an inconsistency.
- credential_misuse and device_spoofing can, by chance, land close enough in
  time to another injected incident for the same entity to read as a
  combined event in the raw log; incidents are still generated
  independently of each other (see the point above about no simulated
  multi-stage campaigns).
- credential_stuffing's target entities are sampled uniformly at random
  from the whole population rather than being weighted toward, say, users
  who share a leaked breach corpus; each incident also mints a fresh set of
  attacker IPs/devices, so two separate credential_stuffing incidents in
  the same run never share infrastructure even though real campaigns often
  reuse a botnet across multiple waves.
"""


def render(**kwargs):
    return TEMPLATE.format(**kwargs)
