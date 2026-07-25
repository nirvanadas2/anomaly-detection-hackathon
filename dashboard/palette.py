"""
Single source of truth for anomaly_type -> color, shared by every view so
the same category always renders in the same color across the Alert Queue,
Entity History, and Metrics Summary.

Colors are the project's validated categorical palette (fixed hue order;
checked with the dataviz skill's validate_palette.js -- adjacent-pair CVD
and normal-vision floors both pass in light mode for these 7 hues). `normal`
and `unclassified` deliberately sit outside the 6-slot attack-type sequence:
`normal` is a recessive muted gray (the expected background, not a category
competing for attention), and `unclassified` uses the palette's slot-7
violet rather than the next attack-type slot, specifically to avoid
implying it's a 7th *kind* of attack -- it's "flagged, type unknown," a
different kind of category entirely. (Green, slot 6, is used for a real
confirmed attack type here -- credential_stuffing -- which is fine; the
"green reads as safe" concern only applies to a label that denotes
uncertainty, not to a fully-identified attack.)

Per the palette's own documented limits, only its first 3 slots validate
under an "all distinct colors visible at once" (all-pairs) scatter test --
past 3 it recommends leaning on the legend/tooltip as the reliable identity
channel rather than hue alone. The one scatter-style chart here (Entity
History's timeline) realistically shows at most 2-3 of these categories for
a single entity at once, but every chart in this app also carries a legend
and per-point tooltips regardless, so identity is never carried by color
alone.
"""

ANOMALY_TYPE_COLORS = {
    "brute_force": "#2a78d6",           # categorical slot 1: blue
    "impossible_travel": "#eb6834",     # categorical slot 2: orange
    "lateral_movement": "#1baf7a",      # categorical slot 3: aqua
    "credential_misuse": "#eda100",     # categorical slot 4: yellow
    "device_spoofing": "#e87ba4",       # categorical slot 5: magenta
    "credential_stuffing": "#008300",   # categorical slot 6: green
}

COLOR_NORMAL = "#898781"        # muted ink -- recessive, not a competing category
COLOR_UNCLASSIFIED = "#4a3aa7"  # categorical slot 7: violet -- see module docstring
COLOR_CRITICAL = "#d03b3b"      # status: critical -- non-categorical accent (e.g. tick marks), not part of the type legend

ANOMALY_TYPES = list(ANOMALY_TYPE_COLORS.keys())

# --- dark chrome tokens ------------------------------------------------------
# App-wide dark theme: page/surface/ink/gridline values are the dataviz skill's
# validated dark-mode chart chrome (references/palette.md), reused as-is rather
# than invented, so the CSS theme injected in app.py and every Altair chart's
# background/gridlines are pulled from this one place and can never drift apart.
DARK_PAGE_PLANE = "#0d0d0d"
DARK_SURFACE = "#1a1a19"
DARK_TEXT_PRIMARY = "#ffffff"
DARK_TEXT_SECONDARY = "#c3c2b7"
DARK_TEXT_MUTED = "#898781"
DARK_GRIDLINE = "#2c2c2a"
DARK_BASELINE = "#383835"
DARK_BORDER = "rgba(255,255,255,0.10)"

# Sequential single-hue ramp (blue), dark-surface-appropriate steps only (300->700
# per references/palette.md's ordinal-ramp guidance: no lighter than step 250 on
# light, no darker than step 600 on dark -- 700 is included here since this ramp
# tints a discrete data cell, not a large scale surface). Used for risk_score's
# magnitude gradient: light = lower risk -> dark = higher risk. This is a
# *sequential* (magnitude) encoding, deliberately distinct from the categorical
# ANOMALY_TYPE_COLORS above -- it never doubles as an identity color.
RISK_SEQUENTIAL_RAMP = [
    "#6da7ec",  # step 300
    "#5598e7",  # step 350
    "#3987e5",  # step 400
    "#2a78d6",  # step 450
    "#256abf",  # step 500
    "#1c5cab",  # step 550
    "#184f95",  # step 600
    "#104281",  # step 650
    "#0d366b",  # step 700
]
RISK_ACCENT = "#3987e5"  # step 400 -- the single accent used for risk_score-related UI (KPI top-border, hero rule)

# Every status/anomaly_type value that appears anywhere in the app, in a
# fixed display order: baseline first, the 6 attack types in palette order,
# catch-all last.
FULL_COLOR_MAP = {
    "normal": COLOR_NORMAL,
    "normal / not flagged": COLOR_NORMAL,
    **ANOMALY_TYPE_COLORS,
    "unclassified": COLOR_UNCLASSIFIED,
}


def domain_range(keys):
    """Altair-ready (domain, range) for exactly the given keys, in the fixed
    display order above -- so a legend never reshuffles based on which
    categories happen to be present in a given view's data."""
    present = set(keys)
    ordered = [k for k in FULL_COLOR_MAP if k in present]
    missing = present - set(ordered)
    if missing:
        raise KeyError(f"No palette color defined for: {sorted(missing)}")
    return ordered, [FULL_COLOR_MAP[k] for k in ordered]
