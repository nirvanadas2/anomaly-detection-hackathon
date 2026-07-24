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
