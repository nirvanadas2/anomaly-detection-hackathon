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
# iframe reads as the whole page, not content inside another app. Making the
# header merely transparent (background: transparent) left it visually
# invisible but still present and still capturing clicks, silently blocking
# clicks on the embedded page's own sticky nav bar (which sits right under
# it) -- display:none removes it from the layout entirely instead.
st.markdown(
    """
    <style>
    .block-container { padding: 0 !important; max-width: 100% !important; }
    header[data-testid="stHeader"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

# The page's CSS uses vh units (#hero{min-height:92vh}, body{min-height:100vh}),
# which resolve against the iframe's OWN layout height, not the real browser
# window -- an oversized iframe height stretches those sections to match,
# pushing their centered content down into a sea of empty space and blowing up
# the three.js hero particle scene's scale. A realistic viewport-sized height
# makes vh resolve the way it does in a normal browser tab; the iframe scrolls
# internally for the rest of the page, same as scrolling a real page would.
# (height="content", the default, auto-measures instead, but that measurement
# collapses to near-zero for this page in practice.)
st.iframe(INDEX_HTML_PATH, width="stretch", height=1000)
