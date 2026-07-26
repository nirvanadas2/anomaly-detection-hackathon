"""
Streamlit dashboard for the anomaly-detection pipeline.

Reads:
  - classification/classified_alerts.csv (entity_id, event_id, risk_score,
    anomaly_type, explanation, top_contributing_features)
  - data_gen/output/access_logs.csv (the raw event log, joined in for
    timestamp / resource / entity_type context the alerts file doesn't carry)
  - data_gen/output/ground_truth_labels.csv (Metrics Summary view only, via
    classification/evaluate.py)

Run with:
    streamlit run dashboard/app.py
"""

import html
import json
import math
import os
import sys

import altair as alt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import ai_triage  # noqa: E402
import data  # noqa: E402
import palette  # noqa: E402
from classification import config as clf_cfg  # noqa: E402
from classification.geo import CITY_COORDS  # noqa: E402

st.set_page_config(page_title="Anomaly Detection Dashboard", page_icon="🛡️", layout="wide")

# Reciprocal link back to the project landing page (docs/index.html), same
# constant pattern as docs/index.html's own DASHBOARD_URL. GitHub Pages isn't
# enabled for this repo yet, so this points at the local docs/index.html via
# a file:// URL instead of a github.io URL that would 404. Once Pages is
# enabled (repo Settings -> Pages -> source: docs/ on master), swap this for
# https://nirvanadas2.github.io/anomaly-detection-hackathon/.
LANDING_PAGE_URL = "file:///C:/Users/Nirvana%20Das/anomaly-detection-hackathon/docs/index.html"

# Avg per-event scoring+classification latency measured by demo_streaming.py's
# live replay (see README's "Key results" -- not computed by this app, since
# that script writes no output file, only a console summary).
STREAMING_AVG_LATENCY_MS = 0.565

# Alert Queue incident grouping: consecutive alerts from the same entity_id
# less than this many minutes apart get merged into one incident. Display-only
# -- purely a re-grouping of classification/classify.py's existing per-event
# output, doesn't change how any individual event is scored or classified.
INCIDENT_GROUPING_WINDOW_MINUTES = 30

# Incident cards are raw HTML <details> elements (a real native expand/collapse,
# not a virtualized grid like st.dataframe), so rendering hundreds of them
# unpaginated is genuinely slow to lay out -- cap how many render per page.
INCIDENT_CARDS_PER_PAGE = 40


# ---------------------------------------------------------------------------
# Visual theme: dark background, blue accent, system sans -- injected once,
# purely cosmetic. Every data binding, filter, and view below is unchanged;
# this only changes how it's painted. Chrome tokens come from
# dashboard/palette.py (the dataviz skill's validated dark chart chrome), so
# this CSS and the Altair chart configs below can never drift apart. Native
# widgets that don't take page CSS (st.dataframe's canvas-rendered grid, most
# notably) get the matching dark theme from .streamlit/config.toml instead.
# ---------------------------------------------------------------------------
def _inject_theme_css():
    st.markdown(
        f"""
        <style>
        :root {{
            --page-plane: {palette.DARK_PAGE_PLANE};
            --surface-1: {palette.DARK_SURFACE};
            --text-primary: {palette.DARK_TEXT_PRIMARY};
            --text-secondary: {palette.DARK_TEXT_SECONDARY};
            --text-muted: {palette.DARK_TEXT_MUTED};
            --border: {palette.DARK_BORDER};
            --accent: {palette.RISK_ACCENT};
        }}

        html, body, [class*="css"] {{
            font-family: system-ui, -apple-system, "Segoe UI", sans-serif !important;
        }}

        .stApp {{ background: var(--page-plane); }}

        [data-testid="stSidebar"] {{
            background: var(--surface-1);
            border-right: 1px solid var(--border);
        }}
        [data-testid="stSidebar"] * {{ color: var(--text-secondary) !important; }}
        [data-testid="stSidebar"] h1 {{ color: var(--text-primary) !important; }}

        h1, h2, h3, h4 {{ color: var(--text-primary) !important; }}
        p, span, label, .stMarkdown, .stCaption {{ color: var(--text-secondary); }}

        /* KPI metric cards */
        div[data-testid="stMetric"] {{
            background: var(--surface-1);
            border: 1px solid var(--border);
            border-top: 3px solid var(--accent);
            border-radius: 12px;
            padding: 18px 20px 14px 20px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.4);
        }}
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] {{
            color: var(--text-muted) !important;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
            color: var(--text-primary) !important;
            font-weight: 600;
        }}

        /* Hero banner */
        .hero-banner {{
            background: linear-gradient(135deg, var(--surface-1) 0%, var(--page-plane) 100%);
            border-bottom: 2px solid var(--accent);
            border-radius: 14px;
            padding: 26px 32px;
            margin-bottom: 20px;
        }}
        .hero-banner h1 {{
            margin: 0;
            font-size: 1.85rem;
            font-weight: 700;
            color: var(--text-primary) !important;
        }}
        .hero-banner p {{
            margin: 6px 0 0 0;
            color: var(--text-secondary) !important;
            font-size: 1rem;
        }}

        [data-testid="stDataFrame"] {{
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_hero():
    st.markdown(
        """
        <div class="hero-banner">
          <h1>🛡️ Anomaly Detection</h1>
          <p>Unsupervised LSTM risk scoring + rule-based classification over 45 days of
          synthetic access logs across 300 entities.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _dark_chart(chart):
    """Apply the same dark chrome tokens used everywhere else (palette.py) to
    an Altair chart's axis/gridlines/background -- cosmetic only, no encoding,
    data, or scale changes."""
    return (
        chart
        .configure_axis(gridColor=palette.DARK_GRIDLINE, grid=True,
                         labelColor=palette.DARK_TEXT_SECONDARY, titleColor=palette.DARK_TEXT_SECONDARY)
        .configure_view(strokeWidth=0, fill=palette.DARK_SURFACE)
        .configure_header(labelColor=palette.DARK_TEXT_SECONDARY, titleColor=palette.DARK_TEXT_SECONDARY)
    )


def _color_legend_html(keys):
    """Small inline swatch+label row so a filter/legend area reflects the
    same fixed anomaly_type -> color mapping used by every chart."""
    ordered, colors = palette.domain_range(keys)
    chips = "".join(
        f'<span style="display:inline-flex;align-items:center;margin-right:14px;font-size:0.85rem;">'
        f'<span style="width:10px;height:10px;border-radius:50%;background:{color};'
        f'display:inline-block;margin-right:6px;"></span>{label}</span>'
        for label, color in zip(ordered, colors)
    )
    return f'<div style="margin:4px 0 12px 0;">{chips}</div>'


def style_anomaly_type_column(df, column="anomaly_type"):
    """Tint each row by its anomaly_type using the shared palette: a light
    wash (not a solid fill, so default cell text stays legible -- text never
    carries the series color, per this project's chart conventions) plus a
    colored left border as the stronger visual cue."""
    def _style(value):
        color = palette.FULL_COLOR_MAP.get(value)
        if color is None:
            return ""
        return f"background-color: {color}26; border-left: 3px solid {color};"

    return df.style.map(_style, subset=[column])


def _risk_ramp_index(value, lo, hi, n):
    """Snap a risk_score to a step index in the n-step sequential ramp, given
    the value range it should scale against. Shared by the table gradient and
    the Attack Map's arc coloring so both read the same magnitude the same way."""
    if hi <= lo:
        return n - 1
    frac = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    return round(frac * (n - 1))


def risk_ramp_color(value, lo, hi):
    """Hex color for a risk_score value from the project's validated sequential
    blue ramp (dashboard/palette.py's RISK_SEQUENTIAL_RAMP)."""
    steps = palette.RISK_SEQUENTIAL_RAMP
    return steps[_risk_ramp_index(value, lo, hi, len(steps))]


def style_risk_score_gradient(styler, series, column="risk_score"):
    """Smooth light->dark gradient on risk_score by magnitude, using only the
    project's validated sequential blue ramp -- a sequential (magnitude)
    encoding, kept deliberately separate from the categorical anomaly_type
    tint above, applied on top of a Styler already built by
    style_anomaly_type_column so both cues coexist on their own columns."""
    lo, hi = float(series.min()), float(series.max())
    steps = palette.RISK_SEQUENTIAL_RAMP
    n = len(steps)

    def _style(value):
        idx = _risk_ramp_index(value, lo, hi, n)
        text = palette.DARK_TEXT_PRIMARY if idx >= n // 2 else "#0b0b0b"
        return f"background-color: {steps[idx]}; color: {text}; font-weight: 600;"

    return styler.map(_style, subset=[column])


# ---------------------------------------------------------------------------
# View 1: Alert Queue
# ---------------------------------------------------------------------------
def group_alerts_into_incidents(alerts_df, window_minutes=INCIDENT_GROUPING_WINDOW_MINUTES):
    """Group flagged alerts from the same entity_id into incidents: consecutive
    alerts (sorted by timestamp) less than `window_minutes` apart get merged
    into one incident -- a session-style gap clustering, not a fixed window
    from the first event, so a slow-building chain still stays one incident
    as long as no single gap between consecutive events exceeds the window.

    Purely a display-layer regrouping of classification/classify.py's existing
    per-event output (`alerts_df` is whatever's already powering the flat
    Alert Queue table) -- doesn't change how any individual event was scored
    or classified.

    Within an incident, the "winning" classification is whichever named
    attack type (i.e. not "unclassified") has the highest average risk_score
    among that incident's events; an incident with no named type stays
    "unclassified". Returns one row per incident with the merged summary plus
    the underlying per-event rows, so the original events are still reachable.
    """
    incidents = []
    for entity_id, entity_alerts in alerts_df.sort_values("timestamp").groupby("entity_id", sort=False):
        entity_alerts = entity_alerts.reset_index(drop=True)
        gap = entity_alerts["timestamp"].diff()
        incident_id = (gap > pd.Timedelta(minutes=window_minutes)).cumsum()
        for _, rows in entity_alerts.groupby(incident_id):
            incidents.append(_summarize_incident(entity_id, rows))
    return pd.DataFrame(incidents)


def _summarize_incident(entity_id, rows):
    rows = rows.sort_values("timestamp")
    named = rows[rows["anomaly_type"] != "unclassified"]
    if named.empty:
        winning_type = "unclassified"
    else:
        winning_type = named.groupby("anomaly_type")["risk_score"].mean().idxmax()

    n_events = len(rows)
    n_absorbed_unclassified = int((rows["anomaly_type"] == "unclassified").sum())
    note = None
    if winning_type != "unclassified" and n_absorbed_unclassified > 0:
        note = (
            f"{n_absorbed_unclassified} of {n_events} events were individually unclassified -- "
            f"grouped here based on temporal proximity to a confirmed {winning_type} incident."
        )

    start_time = rows["timestamp"].min()
    end_time = rows["timestamp"].max()
    return {
        "entity_id": entity_id,
        "start_time": start_time,
        "end_time": end_time,
        "duration_minutes": (end_time - start_time).total_seconds() / 60.0,
        "n_events": n_events,
        "anomaly_type": winning_type,
        "risk_score": float(rows["risk_score"].max()),
        "n_differently_classified": int((rows["anomaly_type"] != winning_type).sum()),
        "note": note,
        "events": rows[["event_id", "timestamp", "resource_accessed", "anomaly_type", "risk_score", "explanation"]].to_dict("records"),
    }


def _incident_card_html(incident):
    """HTML for one incident as an expandable native <details> card --
    entity_id, winning-classification badge, time range, event count, and
    highest risk_score in the summary row; the absorbed individual events
    (with their own per-event classification, when it differs from the
    incident's) inside. Returns a string rather than calling st.markdown
    itself, so a queue of many incidents can be joined into one markdown call
    -- calling st.markdown once per card is the difference between a
    near-instant render and a multi-second one once there are hundreds of
    incidents."""
    color = palette.FULL_COLOR_MAP.get(incident["anomaly_type"], palette.COLOR_UNCLASSIFIED)
    entity_id = html.escape(str(incident["entity_id"]))
    anomaly_type = html.escape(str(incident["anomaly_type"]))
    start_str = incident["start_time"].strftime("%Y-%m-%d %H:%M:%S")
    end_str = incident["end_time"].strftime("%H:%M:%S")
    duration = incident["duration_minutes"]
    duration_str = f"{duration:.0f} min" if duration >= 1 else f"{duration * 60:.0f} sec"
    n_events = incident["n_events"]
    event_word = "event" if n_events == 1 else "events"

    note_html = ""
    if isinstance(incident["note"], str) and incident["note"]:
        note_html = (
            f'<div style="margin-top:10px; padding:9px 14px; background:{color}14; '
            f'border-left:3px solid {color}; border-radius:4px; font-size:0.85rem; color: var(--text-primary);">'
            f'⚠️ {html.escape(incident["note"])}</div>'
        )

    event_rows_html = ""
    for ev in incident["events"]:
        ev_type = str(ev["anomaly_type"])
        ev_color = palette.FULL_COLOR_MAP.get(ev_type, palette.COLOR_UNCLASSIFIED)
        differs_html = (
            f'<span style="color:{ev_color}; font-size:0.72rem; margin-left:8px;">(individually: {html.escape(ev_type)})</span>'
            if ev_type != incident["anomaly_type"] else ""
        )
        event_rows_html += (
            '<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; '
            'padding:8px 12px; border-bottom:1px solid var(--border); font-size:0.85rem;">'
            f'<span style="color:var(--text-muted); font-family:monospace; white-space:nowrap;">{ev["timestamp"].strftime("%Y-%m-%d %H:%M:%S")}</span>'
            f'<span style="color:var(--text-primary); font-weight:600;">{html.escape(str(ev["resource_accessed"]))}</span>'
            f'<span style="color:var(--text-muted); font-family:monospace; font-size:0.8rem;">{html.escape(str(ev["event_id"]))}{differs_html}</span>'
            '</div>'
        )

    return f"""
        <details style="background: var(--surface-1); border: 1px solid var(--border);
                    border-left: 4px solid {color}; border-radius: 10px;
                    padding: 14px 20px; margin-bottom: 10px;">
          <summary style="cursor:pointer;">
            <span style="display:inline-flex; align-items:center; gap:12px; flex-wrap:wrap;">
              <span style="font-weight:700; color:var(--text-primary); font-family:monospace;">{entity_id}</span>
              <span style="background:{color}26; color:{color}; border:1px solid {color}66; border-radius:4px; padding:2px 10px; font-size:0.78rem; font-weight:600;">{anomaly_type}</span>
              <span style="color:var(--text-muted); font-size:0.85rem;">{start_str} &rarr; {end_str}</span>
              <span style="color:var(--text-secondary); font-size:0.85rem;">{n_events} {event_word} over {duration_str}</span>
              <span style="color:var(--text-primary); font-weight:700;">risk={incident['risk_score']:.4f}</span>
            </span>
          </summary>
          {note_html}
          <div style="margin-top:12px;">
            {event_rows_html}
          </div>
        </details>
        """


# Confidence -> color for the AI Triage Copilot card. Not part of palette.py's
# validated categorical/sequential sets (those encode anomaly_type and
# risk_score respectively) -- this is a separate status axis, so it borrows
# COLOR_CRITICAL/RISK_ACCENT/DARK_TEXT_MUTED as standalone accents rather than
# extending either encoded scale.
TRIAGE_CONFIDENCE_COLORS = {
    "high": palette.COLOR_CRITICAL,
    "medium": palette.RISK_ACCENT,
    "low": palette.DARK_TEXT_MUTED,
}


def _render_triage_card(result):
    """Styled card for one Groq triage result. Model output is untrusted
    text, so every field is html.escape()'d before going into unsafe_allow_html
    markdown."""
    confidence = str(result.get("confidence", "")).strip().lower()
    color = TRIAGE_CONFIDENCE_COLORS.get(confidence, palette.DARK_TEXT_MUTED)
    summary = html.escape(str(result.get("summary", "")))
    action = html.escape(str(result.get("recommended_action", "")))
    confidence_label = html.escape(confidence or "unknown")
    model_used = html.escape(str(result.get("_model_used", "groq")))

    st.markdown(
        f"""
        <div style="background: var(--surface-1); border: 1px solid var(--border);
                    border-left: 4px solid {color}; border-radius: 10px;
                    padding: 16px 22px; margin-top: 4px;">
          <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em;
                      color: var(--text-muted); margin-bottom: 8px;">🤖 AI Triage Copilot &mdash; {model_used}</div>
          <p style="color: var(--text-primary); font-size: 1.02rem; margin: 0 0 14px 0; line-height: 1.5;">{summary}</p>
          <div style="display:flex; gap: 32px; flex-wrap: wrap;">
            <div>
              <div style="color: var(--text-muted); font-size:0.78rem; text-transform: uppercase; letter-spacing:0.03em;">Recommended action</div>
              <div style="color: var(--text-primary); font-weight:600; margin-top: 2px;">{action}</div>
            </div>
            <div>
              <div style="color: var(--text-muted); font-size:0.78rem; text-transform: uppercase; letter-spacing:0.03em;">Model confidence</div>
              <div style="color: {color}; font-weight:700; text-transform: capitalize; margin-top: 2px;">{confidence_label}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_alert_queue():
    st.header("Alert Queue")
    st.caption("Events flagged by the model, ranked by risk_score. Click a column header to sort; use the filters below to narrow the queue.")

    df = data.load_alerts_with_context()

    st.markdown(_color_legend_html(df["anomaly_type"].unique()), unsafe_allow_html=True)

    show_ungrouped = st.checkbox(
        "Show ungrouped events",
        value=False,
        help=(
            f"By default, alerts from the same entity_id within {INCIDENT_GROUPING_WINDOW_MINUTES} min of each "
            "other are grouped into a single incident. Check this to see the original flat, one-row-per-event list instead."
        ),
    )

    col1, col2 = st.columns([2, 3])
    with col1:
        types = sorted(df["anomaly_type"].unique())
        selected_types = st.multiselect("anomaly_type", types, default=types)
    with col2:
        lo, hi = float(df["risk_score"].min()), float(df["risk_score"].max())
        if lo == hi:
            st.caption(f"risk_score range: all alerts sit at {lo:.4f} (the top-1% cutoff is this tight)")
            score_range = (lo, hi)
        else:
            score_range = st.slider("risk_score range", min_value=lo, max_value=hi, value=(lo, hi), format="%.4f")

    filtered = df[df["anomaly_type"].isin(selected_types) & df["risk_score"].between(*score_range)]
    filtered = filtered.sort_values("risk_score", ascending=False)

    if show_ungrouped:
        st.caption(f"Showing {len(filtered):,} of {len(df):,} alerts")
        display_cols = ["entity_id", "timestamp", "anomaly_type", "risk_score", "explanation"]
        styled = style_risk_score_gradient(
            style_anomaly_type_column(filtered[display_cols]), filtered["risk_score"],
        )
        st.dataframe(
            styled,
            width="stretch",
            hide_index=True,
            height=560,
            column_config={
                "risk_score": st.column_config.NumberColumn(format="%.4f"),
                "timestamp": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm:ss"),
                "explanation": st.column_config.TextColumn(width="large"),
            },
        )
    else:
        incidents = group_alerts_into_incidents(filtered).sort_values("risk_score", ascending=False).reset_index(drop=True)
        st.caption(
            f"Showing {len(incidents):,} incidents grouped from {len(filtered):,} flagged events "
            f"(consecutive alerts from the same entity_id within {INCIDENT_GROUPING_WINDOW_MINUTES} min of "
            "each other are merged)."
        )
        if incidents.empty:
            st.caption("No alerts match the current filters.")
        else:
            # Each card is raw HTML (a native <details> element, for a real
            # expand/collapse with no JS) rather than a virtualized grid like
            # st.dataframe -- rendering hundreds of these as DOM at once is
            # what actually gets slow, not the grouping computation itself,
            # so page rather than dump the full list every run.
            n_pages = math.ceil(len(incidents) / INCIDENT_CARDS_PER_PAGE)
            if n_pages > 1:
                page_num = st.number_input(
                    "Page", min_value=1, max_value=n_pages, value=1, step=1,
                    help=f"{INCIDENT_CARDS_PER_PAGE} incidents per page, {len(incidents):,} incidents total.",
                )
            else:
                page_num = 1
            start = (page_num - 1) * INCIDENT_CARDS_PER_PAGE
            page_slice = incidents.iloc[start:start + INCIDENT_CARDS_PER_PAGE]
            cards_html = "".join(_incident_card_html(incident) for _, incident in page_slice.iterrows())
            st.markdown(cards_html, unsafe_allow_html=True)
            if n_pages > 1:
                st.caption(f"Page {page_num} of {n_pages} ({start + 1}-{min(start + INCIDENT_CARDS_PER_PAGE, len(incidents))} of {len(incidents):,} incidents)")

    st.subheader("AI Triage Copilot")
    st.caption(
        "Pick an alert below to get a live Groq-drafted incident summary, recommended "
        "response action, and confidence level -- generated fresh on every selection."
    )

    if filtered.empty:
        st.caption("No alerts match the current filters.")
    else:
        selected_event_id = st.selectbox("event_id", filtered["event_id"].tolist(), key="triage_event_id")
        alert_row = filtered.loc[filtered["event_id"] == selected_event_id].iloc[0]

        if not ai_triage.is_configured():
            st.info("Set GROQ_API_KEY environment variable to enable AI Triage Copilot")
        else:
            recent_events = data.entity_recent_events(alert_row["entity_id"]).to_dict("records")
            with st.spinner("Consulting Groq..."):
                try:
                    result = ai_triage.run_triage(alert_row.to_dict(), recent_events)
                except Exception as exc:
                    st.error(f"AI Triage Copilot call failed: {exc}")
                else:
                    _render_triage_card(result)


# ---------------------------------------------------------------------------
# View 2: Entity History
# ---------------------------------------------------------------------------
# Last 10-15 events leading up to (and including) a flagged event, shown by
# the Anomaly Replay expander below.
REPLAY_WINDOW_SIZE = 12


def _replay_baseline(entity_logs_sorted, window_start_ts):
    """Established behavior strictly before a replay window's first event --
    same "don't let the incident's own events leak into its own baseline"
    idea classification/rules.py's _established_baseline uses, kept as a
    small local helper here since this view only needs the resulting
    resource/source_ip/hour sets, not that module's full rule machinery.
    Falls back to the entity's whole history if there's no prior history at
    all (e.g. the flagged event is very early in the entity's timeline), so
    a thin baseline doesn't make every field read as novel by default."""
    prior = entity_logs_sorted[entity_logs_sorted["timestamp"] < window_start_ts]
    if prior.empty:
        prior = entity_logs_sorted
    return {
        "resources": set(prior["resource_accessed"]),
        "source_ips": set(prior["source_ip"]),
        "hours": set(prior["timestamp"].dt.hour),
    }


def _replay_change_notes(row, baseline):
    """One-line, plain-English deviations of a single event from the
    baseline computed by _replay_baseline -- the per-card note in the replay
    below."""
    notes = []
    if row["resource_accessed"] not in baseline["resources"]:
        notes.append("new resource never seen before")
    if baseline["hours"]:
        hour = row["timestamp"].hour
        if hour not in baseline["hours"]:
            typical_hours = sorted(baseline["hours"])
            notes.append(f"unusual hour: {hour:02d}:00 vs typical {typical_hours[0]:02d}:00-{typical_hours[-1]:02d}:00")
    if row["source_ip"] not in baseline["source_ips"]:
        notes.append("new source_ip")
    return notes


def _render_replay_step_card(row, notes, is_flagged, explanation=None):
    """One frame of the replay: a small card for one event, red-bordered and
    labeled if it's the flagged event itself."""
    ts = row["timestamp"]
    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S") if hasattr(ts, "strftime") else str(ts)
    resource = html.escape(str(row["resource_accessed"]))
    note_text = html.escape(" · ".join(notes)) if notes else "consistent with established baseline"

    if is_flagged:
        border = f"2px solid {palette.COLOR_CRITICAL}"
        label_html = (
            f'<div style="color:{palette.COLOR_CRITICAL}; font-weight:700; font-size:0.75rem; '
            f'text-transform:uppercase; letter-spacing:0.04em; margin-bottom:4px;">🚩 Flagged event</div>'
        )
        explanation_html = (
            f'<div style="margin-top:8px; color: var(--text-primary); font-size:0.92rem;">{html.escape(str(explanation))}</div>'
            if explanation else ""
        )
    else:
        border = "1px solid var(--border)"
        label_html = ""
        explanation_html = ""

    st.markdown(
        f"""
        <div style="background: var(--surface-1); border: {border}; border-radius: 8px;
                    padding: 10px 16px; margin-bottom: 6px;">
          {label_html}
          <div style="display:flex; justify-content:space-between; flex-wrap:wrap; gap:12px;">
            <span style="color: var(--text-secondary); font-family: monospace; font-size:0.85rem;">{ts_str}</span>
            <span style="color: var(--text-primary); font-weight:600;">{resource}</span>
          </div>
          <div style="color: var(--text-muted); font-size:0.85rem; margin-top:4px;">{note_text}</div>
          {explanation_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_anomaly_replay(entity_logs_sorted, flagged_alert, window_size=REPLAY_WINDOW_SIZE):
    """Frame-by-frame reconstruction of the events leading up to one flagged
    alert: the last `window_size` events for this entity ending at the
    flagged event, each annotated with how it deviated from the entity's
    established baseline at that point in time."""
    positions = {eid: i for i, eid in enumerate(entity_logs_sorted["event_id"])}
    flagged_event_id = flagged_alert["event_id"]
    if flagged_event_id not in positions:
        st.caption(f"event_id={flagged_event_id} not found in access_logs.csv for this entity.")
        return

    idx = positions[flagged_event_id]
    start = max(0, idx - window_size + 1)
    window = entity_logs_sorted.iloc[start:idx + 1]
    baseline = _replay_baseline(entity_logs_sorted, window.iloc[0]["timestamp"])

    st.markdown(
        f"**{flagged_alert['anomaly_type']}** flagged at risk_score={flagged_alert['risk_score']:.4f} "
        f"(event_id={flagged_event_id}) -- the {len(window)} events leading up to it:"
    )
    for _, row in window.iterrows():
        is_flagged_row = row["event_id"] == flagged_event_id
        notes = _replay_change_notes(row, baseline)
        _render_replay_step_card(
            row, notes, is_flagged_row,
            explanation=flagged_alert["explanation"] if is_flagged_row else None,
        )


def render_entity_history():
    st.header("Entity History")
    st.caption("Recent event timeline for one entity, with baseline behavior (typical resources, typical login hours) compared against any flagged events.")

    logs = data.load_access_logs()
    alerts = data.load_alerts_with_context()

    alert_counts = alerts["entity_id"].value_counts()
    ordered_entities = list(alert_counts.index) + sorted(set(logs["entity_id"]) - set(alert_counts.index))

    def label_entity(eid):
        n = int(alert_counts.get(eid, 0))
        return f"{eid} — {n} alert{'s' if n != 1 else ''}" if n else eid

    entity_id = st.selectbox("entity_id", ordered_entities, format_func=label_entity)

    entity_logs = logs[logs["entity_id"] == entity_id].sort_values("timestamp").copy()
    entity_alert_types = alerts.loc[alerts["entity_id"] == entity_id, ["event_id", "anomaly_type"]]

    entity_logs = entity_logs.merge(entity_alert_types, on="event_id", how="left")
    entity_logs["status"] = entity_logs["anomaly_type"].fillna("normal / not flagged")
    entity_logs["flagged"] = entity_logs["status"] != "normal / not flagged"

    typical_resources = data.typical_resource_set(entity_logs["resource_accessed"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total events", f"{len(entity_logs):,}")
    c2.metric("Flagged events", int(entity_logs["flagged"].sum()))
    c3.metric("entity_type", entity_logs["entity_type"].iloc[0])
    typical_hour_mode = int(entity_logs["timestamp"].dt.hour.mode().iloc[0])
    c4.metric("Most common login hour", f"{typical_hour_mode:02d}:00")

    st.caption(f"**Typical resources** (cover {clf_cfg.TOP_RESOURCE_MASS:.0%} of this entity's normal access): "
               + ", ".join(f"`{r}`" for r in sorted(typical_resources)))

    st.subheader("Event timeline: resource accessed over time")
    status_domain, status_range = palette.domain_range(entity_logs["status"].unique())
    timeline = alt.Chart(entity_logs).mark_circle(size=90, opacity=0.85).encode(
        x=alt.X("timestamp:T", title="time"),
        y=alt.Y("resource_accessed:N", title="resource accessed", sort="-x"),
        color=alt.Color("status:N", title="status",
                         scale=alt.Scale(domain=status_domain, range=status_range)),
        tooltip=[
            alt.Tooltip("timestamp:T", title="time"),
            alt.Tooltip("resource_accessed:N", title="resource"),
            alt.Tooltip("auth_result:N", title="auth_result"),
            alt.Tooltip("status:N", title="status"),
        ],
    ).properties(height=340)
    st.altair_chart(_dark_chart(timeline), width="stretch")

    st.subheader("Login-hour baseline vs flagged events")
    hourly = (
        entity_logs.groupby(entity_logs["timestamp"].dt.hour).size()
        .reindex(range(24), fill_value=0).rename_axis("hour").reset_index(name="count")
    )
    bars = alt.Chart(hourly).mark_bar(color=palette.DARK_TEXT_MUTED, size=14).encode(
        x=alt.X("hour:O", title="hour of day"),
        y=alt.Y("count:Q", title="event count (baseline)"),
        tooltip=["hour", "count"],
    )
    flagged_hours = entity_logs.loc[entity_logs["flagged"], "timestamp"].dt.hour
    if len(flagged_hours):
        ticks_df = pd.DataFrame({"hour": flagged_hours})
        ticks = alt.Chart(ticks_df).mark_tick(color=palette.COLOR_CRITICAL, thickness=3, size=28).encode(
            x=alt.X("hour:O"),
        )
        chart = (bars + ticks)
        st.caption("Gray bars = when this entity normally logs in. Red ticks = the hour of each flagged event.")
    else:
        chart = bars
        st.caption("Gray bars = when this entity normally logs in. No flagged events for this entity.")
    st.altair_chart(
        _dark_chart(chart.properties(height=180)),
        width="stretch",
    )

    if entity_alert_types["anomaly_type"].notna().any():
        st.subheader("Flagged events for this entity")
        flagged_cols = ["event_id", "timestamp", "anomaly_type", "risk_score", "explanation"]
        flagged_rows = alerts[alerts["entity_id"] == entity_id].sort_values("risk_score", ascending=False)
        st.dataframe(
            style_risk_score_gradient(
                style_anomaly_type_column(flagged_rows[flagged_cols]), flagged_rows["risk_score"],
            ),
            width="stretch", hide_index=True,
            column_config={"risk_score": st.column_config.NumberColumn(format="%.4f")},
        )

        with st.expander("Replay: how the risk score evolved", expanded=False):
            replay_source = entity_logs.sort_values("timestamp").reset_index(drop=True)
            replay_alerts = flagged_rows.sort_values("timestamp")
            for i, (_, flagged_alert) in enumerate(replay_alerts.iterrows()):
                _render_anomaly_replay(replay_source, flagged_alert)
                if i < len(replay_alerts) - 1:
                    st.divider()


# ---------------------------------------------------------------------------
# View 3: Metrics Summary
# ---------------------------------------------------------------------------
def render_metrics_summary():
    st.header("Metrics Summary")

    metrics_df, fpr, merged, truth_label_counts = data.compute_metrics()
    auc = data.compute_roc_auc()

    total_events = int(truth_label_counts.sum())
    total_true_anomalies = int(truth_label_counts.drop("normal", errors="ignore").sum())
    n_alerts = fpr["n_alerts"]
    detected_any_type = int((merged["label"] != "normal").sum())
    correctly_typed = int(metrics_df["true_positives"].sum())
    detection_rate = detected_any_type / total_true_anomalies if total_true_anomalies else float("nan")
    budget_ceiling = min(n_alerts, total_true_anomalies) / total_true_anomalies if total_true_anomalies else float("nan")
    n_incidents = len(group_alerts_into_incidents(data.load_alerts_with_context()))

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total alerts", f"{n_alerts:,}")
    k2.metric("Detection rate", f"{detection_rate:.1%}")
    k3.metric("Avg event latency", f"{STREAMING_AVG_LATENCY_MS:.3f} ms",
              help="Measured by demo_streaming.py's live event-by-event replay -- see README.")
    k4.metric("ROC-AUC", f"{auc:.4f}")
    k5.metric("Flagged events → incidents", f"{n_alerts:,} → {n_incidents:,}",
              f"-{n_alerts - n_incidents:,} to review", delta_color="inverse",
              help=(
                  "Alert Queue's incident grouping (Alert Queue tab): consecutive alerts from the same "
                  f"entity_id within {INCIDENT_GROUPING_WINDOW_MINUTES} min of each other are merged into "
                  "one incident, so this is how much less an analyst actually has to review, not just a UI convenience."
              ))

    st.caption(f"{total_true_anomalies:,} true anomalies out of {total_events:,} total events "
               f"({total_true_anomalies / total_events:.2%} of traffic) — a heavily imbalanced label set.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Alert budget", f"{clf_cfg.ALERT_BUDGET_FRACTION:.0%}", f"{n_alerts:,} alerts")
    c2.metric("Detection rate", f"{detection_rate:.1%}", f"{detected_any_type}/{total_true_anomalies} true anomalies in queue")
    c3.metric("Alert false positive rate", f"{fpr['alert_false_positive_rate']:.1%}", "of alerts are false alarms", delta_color="inverse")
    c4.metric("Correctly typed", f"{correctly_typed}/{detected_any_type}", "rule matched the true attack type")

    st.subheader("Precision / recall / F1 per anomaly_type")
    st.caption("Recall's denominator is ALL true events of that type in the full dataset -- including any the alert budget structurally couldn't fit -- not just those inside the queue.")

    display_df = metrics_df.copy()
    for col in ("precision", "recall", "f1"):
        display_df[col] = display_df[col].round(3)
    st.dataframe(style_anomaly_type_column(display_df), width="stretch", hide_index=True)

    st.markdown(_color_legend_html(metrics_df["anomaly_type"].unique()), unsafe_allow_html=True)

    long_df = metrics_df.melt(id_vars="anomaly_type", value_vars=["precision", "recall", "f1"],
                               var_name="metric", value_name="value")
    metric_domain, metric_range = palette.domain_range(metrics_df["anomaly_type"].unique())
    chart = alt.Chart(long_df).mark_bar().encode(
        x=alt.X("metric:N", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("value:Q", title="score", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("anomaly_type:N", title="anomaly_type",
                         scale=alt.Scale(domain=metric_domain, range=metric_range)),
        column=alt.Column("anomaly_type:N", title=None, sort=metric_domain),
        tooltip=["anomaly_type", "metric", alt.Tooltip("value:Q", format=".3f")],
    ).properties(width=110, height=260)
    st.altair_chart(_dark_chart(chart))

    st.subheader("False positive rate at the top-1% alert budget")
    st.markdown(
        f"- **{fpr['n_alerts_that_are_normal']:,} / {fpr['n_alerts']:,}** alerts "
        f"(**{fpr['alert_false_positive_rate']:.1%}**) are actually normal events — "
        f"the false-alarm rate a SOC analyst would experience working this queue.\n"
        f"- **{fpr['statistical_fpr_over_all_normal_events']:.2%}** of *all* normal traffic in the "
        f"dataset got swept into an alert."
    )

    st.info(
        f"**Recall is budget-limited, not a detection failure.** There are **{total_true_anomalies} true anomalies** "
        f"in the dataset, but the top-{clf_cfg.ALERT_BUDGET_FRACTION:.0%} budget only fits **{n_alerts} alerts**. "
        f"Even a perfect rule set could not exceed {n_alerts}/{total_true_anomalies} = **{budget_ceiling:.1%} recall** "
        f"under this budget — today's **{detection_rate:.1%}** overall detection rate reflects that structural "
        f"ceiling as much as it reflects model quality. Raising `ALERT_BUDGET_FRACTION` in "
        f"`classification/config.py` directly raises the recall ceiling (at the cost of more alerts to review)."
    )


# ---------------------------------------------------------------------------
# View 4: Attack Map
# ---------------------------------------------------------------------------
def _great_circle_path(coord_a, coord_b, n=48):
    """Points along the great-circle path between two (lat, lon) coordinates,
    via spherical linear interpolation -- so the arc actually curves on the
    globe instead of Scattergeo's default straight-line interpolation in
    lat/lon space, which reads as a kinked line rather than a geodesic."""
    lat1, lon1 = math.radians(coord_a[0]), math.radians(coord_a[1])
    lat2, lon2 = math.radians(coord_b[0]), math.radians(coord_b[1])
    x1, y1, z1 = math.cos(lat1) * math.cos(lon1), math.cos(lat1) * math.sin(lon1), math.sin(lat1)
    x2, y2, z2 = math.cos(lat2) * math.cos(lon2), math.cos(lat2) * math.sin(lon2), math.sin(lat2)

    dot = max(-1.0, min(1.0, x1 * x2 + y1 * y2 + z1 * z2))
    theta = math.acos(dot)
    if theta < 1e-9:
        return [coord_a[0], coord_b[0]], [coord_a[1], coord_b[1]]

    lats, lons = [], []
    for i in range(n + 1):
        f = i / n
        a = math.sin((1 - f) * theta) / math.sin(theta)
        b = math.sin(f * theta) / math.sin(theta)
        x, y, z = a * x1 + b * x2, a * y1 + b * y2, a * z1 + b * z2
        lats.append(math.degrees(math.atan2(z, math.sqrt(x * x + y * y))))
        lons.append(math.degrees(math.atan2(y, x)))
    return lats, lons


def render_attack_map():
    st.header("Attack Map")
    st.caption(
        "Every classified alert plotted at its geo_location on a rotatable globe, colored by "
        "anomaly_type (same palette as every other view). impossible_travel alerts additionally "
        "draw a great-circle arc between the two locations involved in that incident, shaded by "
        "risk_score. Drag to rotate; scroll or pinch to zoom."
    )

    alerts = data.load_alerts_with_geo()
    mapped = alerts[alerts["geo_location"].isin(CITY_COORDS)].copy()
    n_unmapped = len(alerts) - len(mapped)

    fig = go.Figure()

    types_present, _ = palette.domain_range(mapped["anomaly_type"].unique())
    for atype in types_present:
        sub = mapped[mapped["anomaly_type"] == atype]
        if sub.empty:
            continue
        lats = [CITY_COORDS[g][0] for g in sub["geo_location"]]
        lons = [CITY_COORDS[g][1] for g in sub["geo_location"]]
        fig.add_trace(go.Scattergeo(
            lat=lats, lon=lons, mode="markers", name=atype,
            marker=dict(size=7, color=palette.FULL_COLOR_MAP[atype],
                        line=dict(width=0.6, color=palette.DARK_PAGE_PLANE)),
            customdata=list(zip(sub["entity_id"], sub["risk_score"], sub["geo_location"])),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>" + atype + "<br>%{customdata[2]}<br>"
                "risk_score=%{customdata[1]:.4f}<extra></extra>"
            ),
        ))

    it_rows = mapped[mapped["anomaly_type"] == "impossible_travel"]
    lo, hi = float(alerts["risk_score"].min()), float(alerts["risk_score"].max())
    arc_legend_shown = False
    n_arcs = 0
    for row in it_rows.itertuples():
        try:
            features = json.loads(row.top_contributing_features)
        except (TypeError, ValueError):
            continue
        other_geo = features.get("other_geo_location")
        if other_geo not in CITY_COORDS or row.geo_location not in CITY_COORDS:
            continue

        arc_lats, arc_lons = _great_circle_path(CITY_COORDS[row.geo_location], CITY_COORDS[other_geo])
        fig.add_trace(go.Scattergeo(
            lat=arc_lats, lon=arc_lons, mode="lines",
            line=dict(width=2, color=risk_ramp_color(row.risk_score, lo, hi)),
            opacity=0.85,
            name="impossible_travel arc",
            legendgroup="impossible_travel_arc",
            showlegend=not arc_legend_shown,
            hoverinfo="skip",
        ))
        arc_legend_shown = True
        n_arcs += 1

    fig.update_geos(
        projection_type="orthographic",
        showland=True, landcolor=palette.DARK_SURFACE,
        showocean=True, oceancolor=palette.DARK_PAGE_PLANE,
        showcountries=True, countrycolor=palette.DARK_GRIDLINE,
        showcoastlines=True, coastlinecolor=palette.DARK_GRIDLINE,
        showframe=False,
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        height=640,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color=palette.DARK_TEXT_SECONDARY), bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, width="stretch", theme=None)

    footnote = f"{len(mapped):,} of {len(alerts):,} alerts plotted; {n_arcs} impossible_travel arc(s) drawn."
    if n_unmapped:
        footnote += (f" {n_unmapped} alert(s) reference a geo_location outside the reference lookup "
                     f"table (classification/geo.py) and are not plotted.")
    st.caption(footnote)


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
PAGES = {
    "Alert Queue": render_alert_queue,
    "Entity History": render_entity_history,
    "Metrics Summary": render_metrics_summary,
    "Attack Map": render_attack_map,
}


def main():
    _inject_theme_css()
    _render_hero()

    st.sidebar.title("🛡️ Anomaly Detection")
    st.sidebar.caption("Synthetic access-log anomaly detection: data_gen -> models -> classification")
    st.sidebar.markdown(
        f'<a href="{LANDING_PAGE_URL}" target="_blank" rel="noopener" '
        f'style="font-size:0.85rem;">← Project overview</a>',
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio("View", list(PAGES.keys()), label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.caption(
        f"Alert budget: top {clf_cfg.ALERT_BUDGET_FRACTION:.0%} of events by risk_score "
        f"(configurable in `classification/config.py`)."
    )
    PAGES[page]()


if __name__ == "__main__":
    main()
