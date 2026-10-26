"""PlantPilot Streamlit Operator Dashboard.

Evidence-First, Human-Supervised Industrial Operations AI Copilot.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
import streamlit as st

# Configure paths so frontend can import its own modules
frontend_dir = Path(__file__).resolve().parent
if str(frontend_dir) not in sys.path:
    sys.path.insert(0, str(frontend_dir))

from api_client import APIClient
from views.overview import render_overview
from views.incidents import render_incidents
from views.investigations import render_investigations
from views.approvals import render_approvals
from views.fleet import render_fleet
from views.detector_view import render_detector_view
from views.eval_view import render_eval_view
from views.system_health import render_system_health


# Streamlit Page Config
st.set_page_config(
    page_title="PlantPilot — Industrial Operations AI Copilot",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load Custom CSS
css_file = frontend_dir / "styles.css"
if css_file.exists():
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def init_session_state():
    if "api_base_url" not in st.session_state:
        st.session_state["api_base_url"] = os.environ.get("PLANTPILOT_API_URL", "http://127.0.0.1:8000")


init_session_state()

# Sidebar Navigation & Settings
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 0.5rem 0 1rem 0;">
            <div style="font-size: 1.6rem; font-weight: 800; background: linear-gradient(to right, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                PlantPilot
            </div>
            <div style="font-size: 0.78rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em;">
                Industrial AI Copilot
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # API Connection Settings
    backend_url = st.text_input("Backend API URL", value=st.session_state["api_base_url"], key="api_url_input")
    client = APIClient(base_url=backend_url)

    # Live Health Check Pill
    health_check = client.get_health(check_provider=False)
    is_live = health_check.get("success", False)

    if is_live:
        st.markdown(
            """
            <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.4); padding: 0.35rem 0.75rem; border-radius: 9999px; font-size: 0.8rem; color: #34d399; font-weight: 600; text-align: center; margin-bottom: 1rem;">
                ● BACKEND CONNECTED
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="background: rgba(244, 63, 94, 0.15); border: 1px solid rgba(244, 63, 94, 0.4); padding: 0.35rem 0.75rem; border-radius: 9999px; font-size: 0.8rem; color: #fb7185; font-weight: 600; text-align: center; margin-bottom: 1rem;">
                ○ BACKEND DISCONNECTED
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Navigation Menu
    nav_option = st.radio(
        "Navigation",
        [
            "Operations HQ",
            "Incident Management",
            "AI Investigation Center",
            "Human Operator Sign-Off",
            "Fleet & Asset Registry",
            "Dataset & Anomaly Detector",
            "Evaluation & Benchmarks",
            "System Health & Tests",
        ],
        index=0,
    )

    st.markdown("---")
    st.caption("PlantPilot Core Principles:\n- Evidence-First Enforcement\n- Strict Label Quarantine\n- Human-in-the-Loop Sign-Off")


# Route to Selected View
if nav_option == "Operations HQ":
    render_overview(client)
elif nav_option == "Incident Management":
    render_incidents(client)
elif nav_option == "AI Investigation Center":
    render_investigations(client)
elif nav_option == "Human Operator Sign-Off":
    render_approvals(client)
elif nav_option == "Fleet & Asset Registry":
    render_fleet(client)
elif nav_option == "Dataset & Anomaly Detector":
    render_detector_view(client)
elif nav_option == "Evaluation & Benchmarks":
    render_eval_view(client)
elif nav_option == "System Health & Tests":
    render_system_health(client)
