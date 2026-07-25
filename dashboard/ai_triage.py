"""
AI Triage Copilot: a live Groq call that drafts a SOC-analyst-style
assessment for one selected alert (Alert Queue view only).

Deliberately uncached -- dashboard/data.py's @st.cache_data loaders exist
because the underlying CSVs don't change during a session; this module calls
a live external API per alert selection, and every selection is meant to be
a fresh call, so nothing here is wrapped in @st.cache_data.
"""

import json
import os

from dotenv import load_dotenv
from groq import Groq

# Loads GROQ_API_KEY (and anything else) from a local .env file in the
# project root into os.environ, if one exists -- run at import time, before
# is_configured()/run_triage() ever read the environment, so a developer
# doesn't have to `export GROQ_API_KEY=...` every session. A real .env is
# gitignored; see .env.example for the expected format. No-op (and harmless)
# if the key is already set some other way, e.g. a real environment variable
# or a deployment platform's secrets manager.
load_dotenv()

MODEL_NAME = "llama-3.3-70b-versatile"

# Client is initialized once at import time and reused across calls rather
# than constructed per-request.
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


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
    """Whether GROQ_API_KEY is set -- callers use this to show a setup
    message instead of attempting (and crashing on) a keyless API call."""
    return bool(os.environ.get("GROQ_API_KEY"))


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

Respond with JSON matching this schema only:
{json.dumps(RESPONSE_SCHEMA)}"""


def run_triage(alert, recent_events):
    """One live call to Groq for a single alert. Returns a dict with
    'summary', 'recommended_action', 'confidence', plus '_model_used' (the
    model that answered, for display)."""
    client = _get_client()
    prompt = _build_prompt(alert, recent_events)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    result = json.loads(response.choices[0].message.content)
    result["_model_used"] = MODEL_NAME
    return result
