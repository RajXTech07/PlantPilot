"""Overview / Operations HQ view for PlantPilot Streamlit."""

from __future__ import annotations
import streamlit as st
import pandas as pd
from api_client import APIClient


def render_overview(client: APIClient):
    st.markdown(
        """
        <div class="plantpilot-header">
            <div class="plantpilot-title">PlantPilot Operations HQ</div>
            <div class="plantpilot-subtitle">Evidence-First, Human-Supervised Industrial Operations AI Copilot</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Fetch live data from backend
    health_res = client.get_health(check_provider=False)
    incidents_res = client.list_incidents(limit=100)
    machines_res = client.list_machines(limit=100)

    is_online = health_res.get("success", False)
    incidents = incidents_res.get("data", []) if incidents_res.get("success") else []
    machines = machines_res.get("data", []) if machines_res.get("success") else []

    # Calculate operational metrics
    total_incidents = len(incidents)
    critical_incidents = sum(1 for i in incidents if i.get("severity") in ("critical", "high"))
    total_machines = len(machines)

    # Top KPI Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Backend Status</div>
                <div class="stat-value" style="color: {'#10b981' if is_online else '#f43f5e'};">
                    {'ONLINE' if is_online else 'OFFLINE'}
                </div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">FastAPI Port 8000</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Active Incidents</div>
                <div class="stat-value" style="color: #38bdf8;">{total_incidents}</div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">Tracked in SQLite</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">High / Critical</div>
                <div class="stat-value" style="color: {'#fb7185' if critical_incidents > 0 else '#10b981'};">
                    {critical_incidents}
                </div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">Requires attention</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Registered Assets</div>
                <div class="stat-value" style="color: #a78bfa;">{total_machines}</div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">AI4I Machine Fleet</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.write("")

    # Architecture & Evidence-First Workflow diagram
    with st.expander("System Architecture & Evidence-First Guarantee", expanded=True):
        st.markdown(
            """
            PlantPilot guarantees that **no machine diagnosis or recommendation is produced without verified observations**.
            The core architecture separates unverified claims from physical evidence:
            """
        )
        flow_col1, flow_col2, flow_col3, flow_col4, flow_col5 = st.columns(5)
        with flow_col1:
            st.info("**1. Sensor Telemetry**\n\nAI4I 2020 dataset via quarantined DataAdapter")
        with flow_col2:
            st.info("**2. Safety Gate**\n\nPure Python gatekeeper inspects tools before execution")
        with flow_col3:
            st.info("**3. Bounded Agent**\n\nStep 0 forces `tool_choice='required'` for evidence")
        with flow_col4:
            st.info("**4. Evidence Vault**\n\nObserved values persisted with strict provenance")
        with flow_col5:
            st.info("**5. Human Operator**\n\nRecommendations gated: require operator sign-off")

    st.write("")

    # Recent Incidents Table
    st.subheader("Recent Operational Incidents")
    if not incidents:
        st.warning("No incidents found or backend unreachable. Create a new incident from the Incidents tab.")
    else:
        df = pd.DataFrame(incidents)
        display_cols = [c for c in ["id", "description", "severity", "category", "status", "created_at"] if c in df.columns]
        st.dataframe(
            df[display_cols].head(10),
            use_container_width=True,
            hide_index=True,
        )

        st.caption("Tip: Select an incident in the **AI Investigation** tab to launch an evidence-first investigation loop.")
