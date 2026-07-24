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

import os
import sys

import altair as alt
import pandas as pd
import streamlit as st

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import data  # noqa: E402
import palette  # noqa: E402
from classification import config as clf_cfg  # noqa: E402

st.set_page_config(page_title="Anomaly Detection Dashboard", page_icon="🛡️", layout="wide")


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


# ---------------------------------------------------------------------------
# View 1: Alert Queue
# ---------------------------------------------------------------------------
def render_alert_queue():
    st.header("Alert Queue")
    st.caption("Events flagged by the model, ranked by risk_score. Click a column header to sort; use the filters below to narrow the queue.")

    df = data.load_alerts_with_context()

    st.markdown(_color_legend_html(df["anomaly_type"].unique()), unsafe_allow_html=True)

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

    st.caption(f"Showing {len(filtered):,} of {len(df):,} alerts")
    display_cols = ["entity_id", "timestamp", "anomaly_type", "risk_score", "explanation"]
    styled = style_anomaly_type_column(filtered[display_cols])
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


# ---------------------------------------------------------------------------
# View 2: Entity History
# ---------------------------------------------------------------------------
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
    ).properties(height=340).configure_axis(gridColor="#e1e0d9", grid=True).configure_view(strokeWidth=0)
    st.altair_chart(timeline, width="stretch")

    st.subheader("Login-hour baseline vs flagged events")
    hourly = (
        entity_logs.groupby(entity_logs["timestamp"].dt.hour).size()
        .reindex(range(24), fill_value=0).rename_axis("hour").reset_index(name="count")
    )
    bars = alt.Chart(hourly).mark_bar(color="#c3c2b7", size=14).encode(
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
        chart.properties(height=180).configure_axis(gridColor="#e1e0d9", grid=True).configure_view(strokeWidth=0),
        width="stretch",
    )

    if entity_alert_types["anomaly_type"].notna().any():
        st.subheader("Flagged events for this entity")
        flagged_cols = ["event_id", "timestamp", "anomaly_type", "risk_score", "explanation"]
        flagged_rows = alerts[alerts["entity_id"] == entity_id].sort_values("risk_score", ascending=False)
        st.dataframe(
            style_anomaly_type_column(flagged_rows[flagged_cols]),
            width="stretch", hide_index=True,
            column_config={"risk_score": st.column_config.NumberColumn(format="%.4f")},
        )


# ---------------------------------------------------------------------------
# View 3: Metrics Summary
# ---------------------------------------------------------------------------
def render_metrics_summary():
    st.header("Metrics Summary")

    metrics_df, fpr, merged, truth_label_counts = data.compute_metrics()

    total_events = int(truth_label_counts.sum())
    total_true_anomalies = int(truth_label_counts.drop("normal", errors="ignore").sum())
    n_alerts = fpr["n_alerts"]
    detected_any_type = int((merged["label"] != "normal").sum())
    correctly_typed = int(metrics_df["true_positives"].sum())
    detection_rate = detected_any_type / total_true_anomalies if total_true_anomalies else float("nan")
    budget_ceiling = min(n_alerts, total_true_anomalies) / total_true_anomalies if total_true_anomalies else float("nan")

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
    ).properties(width=110, height=260).configure_axis(gridColor="#e1e0d9", grid=True).configure_view(strokeWidth=0)
    st.altair_chart(chart)

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
# Navigation
# ---------------------------------------------------------------------------
PAGES = {
    "Alert Queue": render_alert_queue,
    "Entity History": render_entity_history,
    "Metrics Summary": render_metrics_summary,
}


def main():
    st.sidebar.title("🛡️ Anomaly Detection")
    st.sidebar.caption("Synthetic access-log anomaly detection: data_gen -> models -> classification")
    page = st.sidebar.radio("View", list(PAGES.keys()), label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.caption(
        f"Alert budget: top {clf_cfg.ALERT_BUDGET_FRACTION:.0%} of events by risk_score "
        f"(configurable in `classification/config.py`)."
    )
    PAGES[page]()


if __name__ == "__main__":
    main()
