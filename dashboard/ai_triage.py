"""
AI Triage Copilot: a live Gemini call that drafts a SOC-analyst-style
assessment for one selected alert (Alert Queue view only).

Deliberately uncached -- dashboard/data.py's @st.cache_data loaders exist
because the underlying CSVs don't change during a session; this module calls
a live external API per alert selection, and every selection is meant to be
a fresh call, so nothing here is wrapped in @st.cache_data.
"""

import json
import os

from google import genai
from google.genai import types

# Tried in order for every call: gemini-2.5-flash returns 404 for accounts
# without legacy access, so gemini-flash-latest (the current default alias)
# is tried first, falling back to the pinned gemini-2.0-flash if that alias
# is ever unavailable for a given key/project.
MODEL_CANDIDATES = ("gemini-flash-latest", "gemini-2.0-flash")

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "2-3 sentence plain-English incident summary a SOC analyst could read in 10 seconds.",
        },
        "recommended_action": {
            "type": "string",
            "description": (
                "A single recommended response action, e.g. 'investigate further', "
                "'likely false positive - first use of new resource', "
                "'escalate immediately - high confidence attack'."
            ),
        },
        "confidence": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": "Confidence level in this assessment.",
        },
    },
    "required": ["summary", "recommended_action", "confidence"],
}


def is_configured():
    """Whether GOOGLE_API_KEY is set -- callers use this to show a setup
    message instead of attempting (and crashing on) a keyless API call."""
    return bool(os.environ.get("GOOGLE_API_KEY"))


def _build_prompt(alert, recent_events):
    events_block = "\n".join(
        f"- {e['timestamp']} | resource={e['resource_accessed']} | "
        f"auth_result={e.get('auth_result', 'n/a')} | geo={e.get('geo_location', 'n/a')}"
        for e in recent_events
    ) or "(no prior event history found for this entity)"

    return f"""You are a SOC (Security Operations Center) triage assistant helping an
analyst work through an alert queue quickly.

ALERT
  entity_id: {alert['entity_id']}
  anomaly_type: {alert['anomaly_type']}
  risk_score: {alert['risk_score']}
  explanation: {alert['explanation']}
  top_contributing_features: {alert['top_contributing_features']}

RECENT EVENT HISTORY for this entity (up to the last 10 events, oldest first):
{events_block}

Based on the alert and this entity's recent behavior, provide:
1. A 2-3 sentence plain-English incident summary a SOC analyst could read in 10 seconds.
2. A recommended response action (e.g. "investigate further", "likely false positive - first use of new resource", "escalate immediately - high confidence attack").
3. Your confidence level (low/medium/high) in this assessment.

Respond with JSON matching the provided schema only."""


def run_triage(alert, recent_events):
    """One live call to Gemini for a single alert. Returns a dict with
    'summary', 'recommended_action', 'confidence', plus '_model_used' (the
    candidate from MODEL_CANDIDATES that actually answered, for display).
    Tries each candidate model in order, moving to the next only if the call
    itself raises (e.g. a 404 for a model unavailable to this key/project);
    raises the last error if every candidate fails. Callers catch and render
    an error card rather than this module failing silently."""
    api_key = os.environ["GOOGLE_API_KEY"]
    client = genai.Client(api_key=api_key)
    contents = _build_prompt(alert, recent_events)
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,
        temperature=0.2,
    )

    last_error = None
    for model_name in MODEL_CANDIDATES:
        try:
            response = client.models.generate_content(model=model_name, contents=contents, config=config)
        except Exception as exc:  # noqa: BLE001 -- try the next candidate on any failure
            last_error = exc
            continue
        result = json.loads(response.text)
        result["_model_used"] = model_name
        return result
    raise last_error
