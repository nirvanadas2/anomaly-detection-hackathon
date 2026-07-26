"""
Standalone Streamlit wrapper around docs/index.html -- renders the exact same
static landing page (unchanged) as its own Streamlit Community Cloud app, so
it's reachable at a streamlit.app URL in addition to GitHub Pages. The page's
"Launch live dashboard" buttons already use target="_blank", which works fine
from inside Streamlit's iframe sandbox (it grants allow-popups); only
same-tab (target="_top") navigation would be blocked, and this page doesn't
use that.

Run with:
    streamlit run landing_app.py
"""

from pathlib import Path

import streamlit as st

INDEX_HTML_PATH = Path(__file__).resolve().parent / "docs" / "index.html"

st.set_page_config(
    page_title="Redlight Greenlight — Behavioral Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
)

# Streamlit's own chrome (header, padding) would otherwise frame the page and
# clash with its full-bleed dark design -- collapse that padding so the
# iframe reads as the whole page, not content inside another app.
st.markdown(
    """
    <style>
    .block-container { padding: 0 !important; max-width: 100% !important; }
    header[data-testid="stHeader"] { background: transparent; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.iframe(INDEX_HTML_PATH, width="stretch")
