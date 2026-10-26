"""AI Investigation Center for PlantPilot Streamlit."""

from __future__ import annotations
import json
import streamlit as st
import pandas as pd
from typing import Any, Dict
from api_client import APIClient


def render_investigations(client: APIClient):
    st.title("AI Investigation Center")
    st.markdown(
        "Autonomous, evidence-first incident investigations powered by **NVIDIA Nemotron** and bound by strict verification rules."
    )

    tab_run, tab_lookup = st.tabs(["Launch Investigation", "Inspect Past Investigation"])

    with tab_run:
        # Fetch incidents
        incidents_res = client.list_incidents(limit=100)
        incidents = incidents_res.get("data", []) if incidents_res.get("success") else []

        if not incidents:
            st.warning("No incidents available. Please create an incident in the Incidents tab first.")
            return

        incident_choices = {
            f"{i['id']} | {i.get('severity', '').upper()} | {i.get('description', '')[:65]}...": i['id']
            for i in incidents
        }

        selected_label = st.selectbox("Select Incident to Investigate", list(incident_choices.keys()))
        selected_incident_id = incident_choices[selected_label]

        # Show incident summary
        selected_inc = next((i for i in incidents if i["id"] == selected_incident_id), None)
        if selected_inc:
            with st.container():
                st.info(f"**Target Incident:** {selected_inc.get('description')}\n\n**Severity:** `{selected_inc.get('severity')}` | **Category:** `{selected_inc.get('category')}`")

        notes = st.text_input(
            "Investigation Operator Notes (Optional)",
            value="Standard operational diagnostic with sensor reading verification.",
            help="Notes appended to investigation record for audit purposes."
        )

        col_btn, col_info = st.columns([1, 2])
        with col_btn:
            run_btn = st.button("Start Investigation", type="primary", use_container_width=True)

        with col_info:
            st.caption("Synchronous bounded agent loop: Step 0 forces tool invocation to collect verified sensor telemetry.")

        if run_btn:
            with st.spinner("PlantPilot AI agent investigating... Invoking tools and validating telemetry..."):
                result = client.start_investigation(
                    incident_id=selected_incident_id,
                    notes=notes.strip() if notes else None,
                    timeout=90.0,
                )

            if not result.get("success"):
                st.error(f"Investigation request failed: {result.get('error')}")
                if "raw" in result:
                    with st.expander("Raw Server Response"):
                        st.code(result.get("raw"))
            else:
                inv_data = result.get("data", {})
                st.success("AI Investigation Completed!")
                st.session_state["latest_inv_id"] = inv_data.get("investigation_id")
                render_investigation_details(client, inv_data.get("investigation_id"), result_summary=inv_data)

        elif st.session_state.get("latest_inv_id"):
            st.divider()
            st.subheader(f"Latest Investigation: `{st.session_state['latest_inv_id']}`")
            render_investigation_details(client, st.session_state["latest_inv_id"])

    with tab_lookup:
        lookup_id = st.text_input("Enter Investigation ID (e.g., inv_abc123...)", value=st.session_state.get("latest_inv_id", ""))
        if st.button("Load Investigation Details", use_container_width=True):
            if lookup_id.strip():
                render_investigation_details(client, lookup_id.strip())
            else:
                st.warning("Please enter a valid Investigation ID.")


def render_investigation_details(client: APIClient, investigation_id: str, result_summary: Dict[str, Any] = None):
    res = client.get_investigation(investigation_id)
    if not res.get("success"):
        st.error(f"Failed to load investigation details: {res.get('error')}")
        if result_summary:
            st.json(result_summary)
        return

    inv = res.get("data", {})

    status_val = inv.get("status", "unknown").upper()
    status_color = "#10b981" if status_val == "COMPLETED" else ("#f59e0b" if status_val == "INCOMPLETE" else "#ef4444")

    # Status header
    st.markdown(
        f"""
        <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 8px; padding: 1rem 1.5rem; margin: 1rem 0;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 1.25rem; font-weight: 700; color: #f8fafc;">Investigation: {investigation_id}</span>
                    <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">Incident: {inv.get('incident_id')} | Started: {inv.get('started_at')}</div>
                </div>
                <div style="background: {status_color}22; color: {status_color}; border: 1px solid {status_color}55; padding: 0.4rem 1rem; border-radius: 9999px; font-weight: 700; font-size: 0.95rem;">
                    {status_val}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Findings section
    findings = inv.get("findings", [])
    if findings:
        st.subheader("Key Findings & Agent Reasoning")
        for f in findings:
            st.markdown(f"> {f}")

    # Tabs for detailed inspection
    tab_ev, tab_hyp, tab_rec, tab_tools, tab_raw = st.tabs([
        f"Verified Evidence ({len(inv.get('evidence', []))})",
        f"Model Hypotheses ({len(inv.get('hypotheses', []))})",
        f"Gated Recommendations ({len(inv.get('recommendations', []))})",
        f"Tool Audit Trail ({len(inv.get('tool_executions', []))})",
        "Raw JSON Audit"
    ])

    with tab_ev:
        evidence = inv.get("evidence", [])
        if not evidence:
            st.warning("No physical evidence was verified for this investigation.")
        else:
            st.markdown(
                """
                <span class="badge-evidence">EVIDENCE-FIRST PRINCIPLE</span>
                All readings below were retrieved via verified tool execution against dataset sensors, strictly quarantined from failure labels.
                """,
                unsafe_allow_html=True,
            )
            ev_records = []
            for e in evidence:
                ev_records.append({
                    "Parameter": e.get("parameter_name") or "N/A",
                    "Value": f"{e.get('numeric_value')} {e.get('unit') or ''}".strip(),
                    "Type": e.get("evidence_type"),
                    "Verified": "VERIFIED" if e.get("verified") else "UNVERIFIED",
                    "Source": e.get("source"),
                    "Description": e.get("description"),
                    "Tool Execution FK": e.get("tool_execution_id") or "N/A",
                })
            st.dataframe(pd.DataFrame(ev_records), width="stretch", hide_index=True)

    with tab_hyp:
        hypotheses = inv.get("hypotheses", [])
        if not hypotheses:
            st.info("No hypotheses recorded.")
        else:
            st.markdown(
                """
                <span class="badge-hypothesis">UNVERIFIED MODEL REASONING</span>
                Hypotheses represent generative model claims. Under PlantPilot safety doctrine, hypotheses are NEVER treated as physical facts.
                """,
                unsafe_allow_html=True,
            )
            for h in hypotheses:
                with st.expander(f"Hypothesis: {h.get('title')}", expanded=True):
                    st.write(f"**Confidence:** `{h.get('confidence') or 'unspecified'}`")
                    st.write(f"**Explanation:**\n{h.get('explanation')}")
                    st.write(f"**Supporting Evidence IDs:** `{h.get('supporting_evidence_ids', [])}`")
                    st.write(f"**Contradicting Evidence IDs:** `{h.get('contradicting_evidence_ids', [])}`")

    with tab_rec:
        recs = inv.get("recommendations", [])
        if not recs:
            st.info("No recommendations generated.")
        else:
            st.markdown(
                """
                <span class="badge-safety">HUMAN-IN-THE-LOOP SAFETY GATE</span>
                The AI cannot trigger physical actions autonomously. All proposals require human operator review.
                """,
                unsafe_allow_html=True,
            )
            for r in recs:
                st.markdown(
                    f"""
                    <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 8px; padding: 1rem; margin-bottom: 0.75rem;">
                        <div style="display: flex; justify-content: space-between;">
                            <span style="font-weight: 700; color: #f1f5f9;">{r.get('title')}</span>
                            <span style="font-size: 0.8rem; background: #f59e0b22; color: #fbbf24; border: 1px solid #f59e0b44; padding: 2px 8px; border-radius: 4px;">
                                {r.get('approval_status', '').upper()}
                            </span>
                        </div>
                        <p style="color: #cbd5e1; font-size: 0.95rem; margin-top: 6px;">{r.get('description')}</p>
                        <div style="font-size: 0.8rem; color: #94a3b8;">
                            Requires Approval: <b>{r.get('requires_approval')}</b> | Priority: <b>{r.get('priority')}</b> | ID: <code>{r.get('id')}</code>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with tab_tools:
        tools = inv.get("tool_executions", [])
        if not tools:
            st.info("No tool executions recorded.")
        else:
            for t in tools:
                status_icon = "PASS" if t.get("status") == "completed" else "FAIL"
                with st.expander(f"Step {t.get('step_index')}: {t.get('tool_name')} [{status_icon}]"):
                    st.write(f"**Execution ID:** `{t.get('id')}`")
                    st.write(f"**Executed At:** `{t.get('executed_at')}`")
                    st.write(f"**Status:** `{t.get('status')}`")
                    st.write(f"**Input Size:** `{t.get('input_size_bytes')} bytes` | **Output Size:** `{t.get('output_size_bytes')} bytes`")
                    if t.get("error_message"):
                        st.error(f"Error: {t.get('error_message')}")
                    st.markdown("**Arguments:**")
                    st.json(t.get("arguments", {}))
                    st.markdown("**Tool Output Result:**")
                    st.code(t.get("result") or "(empty)", language="json")

    with tab_raw:
        st.json(inv)
