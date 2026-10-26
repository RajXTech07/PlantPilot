"""Incidents view for PlantPilot Streamlit."""

from __future__ import annotations
import streamlit as st
import pandas as pd
from typing import Any, Dict
from api_client import APIClient


def render_incidents(client: APIClient):
    st.title("Operational Incident Management")
    st.markdown("Log, review, and organize machine anomalies across the industrial facility.")

    tab_list, tab_create = st.tabs(["Browse Incidents", "Log New Incident"])

    with tab_list:
        incidents_res = client.list_incidents(limit=100)
        if not incidents_res.get("success"):
            st.error(f"Failed to fetch incidents: {incidents_res.get('error')}")
            return

        incidents = incidents_res.get("data", [])
        if not incidents:
            st.info("No incidents currently logged. Use the 'Log New Incident' tab to record one.")
            return

        # Filtering controls
        col_search, col_sev, col_cat = st.columns([2, 1, 1])
        with col_search:
            search_query = st.text_input("Search description or ID", placeholder="e.g. UDI 1, wear, temp...")
        with col_sev:
            sev_filter = st.selectbox("Filter Severity", ["All", "critical", "high", "medium", "low"])
        with col_cat:
            cat_filter = st.selectbox("Filter Category", ["All", "operational", "thermal", "mechanical", "electrical"])

        filtered = incidents
        if search_query:
            q = search_query.lower()
            filtered = [i for i in filtered if q in i.get("description", "").lower() or q in i.get("id", "").lower()]
        if sev_filter != "All":
            filtered = [i for i in filtered if i.get("severity") == sev_filter]
        if cat_filter != "All":
            filtered = [i for i in filtered if i.get("category") == cat_filter]

        st.caption(f"Showing {len(filtered)} of {len(incidents)} recorded incidents")

        # Display table
        df = pd.DataFrame(filtered)
        display_cols = [c for c in ["id", "severity", "category", "status", "description", "created_at"] if c in df.columns]
        st.dataframe(df[display_cols], width="stretch", hide_index=True)

        st.divider()
        st.subheader("Incident Quick Inspect")
        selected_id = st.selectbox("Select Incident ID for Detail View", [i["id"] for i in filtered], key="inspect_inc_id")
        if selected_id:
            incident = next((i for i in incidents if i["id"] == selected_id), None)
            if incident:
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.write(f"**ID:** `{incident.get('id')}`")
                    st.write(f"**Severity:** `{incident.get('severity')}`")
                    st.write(f"**Category:** `{incident.get('category')}`")
                    st.write(f"**Status:** `{incident.get('status')}`")
                with col_info2:
                    st.write(f"**Machine ID:** `{incident.get('machine_id') or 'N/A'}`")
                    st.write(f"**Created At:** `{incident.get('created_at')}`")
                    st.write(f"**Updated At:** `{incident.get('updated_at')}`")

                st.write("**Description:**")
                st.info(incident.get("description"))

                if incident.get("metadata"):
                    st.write("**Attached Metadata:**")
                    st.json(incident.get("metadata"))

    with tab_create:
        st.subheader("Record Machine Anomaly / Incident")

        # Preset template buttons
        st.markdown("**Quick Preset Scenarios (AI4I 2020 Dataset Verified):**")
        p_col1, p_col2, p_col3 = st.columns(3)

        preset_desc = ""
        preset_sev = "medium"
        preset_cat = "operational"

        if p_col1.button("Preset: Tool Wear Alert (UDI 1)"):
            st.session_state["form_desc"] = "Elevated tool wear and vibration detected on milling station UDI 1. Operator requests evidence verification."
            st.session_state["form_sev"] = "high"
            st.session_state["form_cat"] = "mechanical"

        if p_col2.button("Preset: Thermal Runaway (UDI 161)"):
            st.session_state["form_desc"] = "Process temperature delta spike observed at station UDI 161. Potential heat dissipation failure."
            st.session_state["form_sev"] = "critical"
            st.session_state["form_cat"] = "thermal"

        if p_col3.button("Preset: Torque Overload (UDI 51)"):
            st.session_state["form_desc"] = "Torque abnormal spike with rotational speed drop on drive unit UDI 51. Inspect operational readings."
            st.session_state["form_sev"] = "high"
            st.session_state["form_cat"] = "electrical"

        with st.form("create_incident_form"):
            desc = st.text_area(
                "Incident Description *",
                value=st.session_state.get("form_desc", "Milling machine UDI 1 reported tool degradation anomaly. Please inspect sensor readings."),
                height=120,
                help="Include reference to dataset UDI (e.g. UDI 1) for automated sensor retrieval.",
            )

            c1, c2 = st.columns(2)
            with c1:
                severity = st.selectbox(
                    "Severity Level *",
                    ["low", "medium", "high", "critical"],
                    index=["low", "medium", "high", "critical"].index(st.session_state.get("form_sev", "high")),
                )
            with c2:
                category = st.selectbox(
                    "Category *",
                    ["operational", "mechanical", "thermal", "electrical"],
                    index=["operational", "mechanical", "thermal", "electrical"].index(st.session_state.get("form_cat", "mechanical")),
                )

            machine_id = st.text_input("Associated Machine ID (Optional)", placeholder="e.g. mach_abc123")

            submitted = st.form_submit_button("Submit Incident to Backend", use_container_width=True)
            if submitted:
                if not desc.strip():
                    st.error("Description cannot be empty.")
                else:
                    with st.spinner("Submitting incident to FastAPI backend..."):
                        res = client.create_incident(
                            description=desc.strip(),
                            severity=severity,
                            category=category,
                            machine_id=machine_id.strip() if machine_id.strip() else None,
                        )
                        if res.get("success"):
                            st.success(f"Incident successfully created! ID: `{res['data']['id']}`")
                            st.json(res["data"])
                        else:
                            st.error(f"Failed to create incident: {res.get('error')}")
