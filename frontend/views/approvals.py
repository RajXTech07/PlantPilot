"""Human Operator Sign-Off view for PlantPilot Streamlit."""

from __future__ import annotations
import streamlit as st
import pandas as pd
from typing import Any, Dict
from api_client import APIClient


def render_approvals(client: APIClient):
    st.title("Human Operator Sign-Off")
    st.markdown(
        "PlantPilot enforces a strict **Human-in-the-Loop** policy. The AI copilot can propose maintenance or corrective actions, but cannot execute them without human operator sign-off."
    )

    # Fetch recommendations
    tab_pending, tab_history = st.tabs(["Pending Review Queue", "Reviewed Recommendations Audit"])

    with tab_pending:
        res = client.list_recommendations(approval_status="pending_approval", limit=100)
        recs = res.get("data", []) if res.get("success") else []

        if not recs:
            st.success("No recommendations are currently pending approval. All clear!")
            return

        st.info(f"**{len(recs)} recommendation(s)** requiring operator inspection and sign-off.")

        for r in recs:
            rec_id = r.get("id")
            with st.container():
                st.markdown(
                    f"""
                    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(245, 158, 11, 0.4); border-radius: 8px; padding: 1.2rem; margin-bottom: 1rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-size: 1.15rem; font-weight: 700; color: #f1f5f9;">{r.get('title')}</span>
                            <span class="badge-hypothesis">PENDING APPROVAL</span>
                        </div>
                        <p style="color: #cbd5e1; margin-top: 8px; font-size: 0.95rem;">{r.get('description')}</p>
                        <div style="font-size: 0.8rem; color: #94a3b8;">
                            Priority: <b>{r.get('priority').upper()}</b> | Created: <b>{r.get('created_at')}</b> | ID: <code>{rec_id}</code>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Action form
                with st.form(key=f"review_form_{rec_id}"):
                    col_op, col_choice = st.columns([1, 1])
                    with col_op:
                        operator_name = st.text_input(
                            "Operator Name *",
                            value="Operator Chief (Plant 01)",
                            key=f"op_{rec_id}",
                        )
                    with col_choice:
                        decision = st.radio(
                            "Decision *",
                            ["approved", "rejected"],
                            horizontal=True,
                            key=f"dec_{rec_id}",
                        )

                    review_notes = st.text_input(
                        "Operator Rationale / Notes",
                        placeholder="e.g. Verified vibration readings; approved spindle maintenance window.",
                        key=f"notes_{rec_id}",
                    )

                    btn_col1, btn_col2 = st.columns([1, 2])
                    with btn_col1:
                        submit_review = st.form_submit_button(
                            "Submit Sign-Off",
                            type="primary",
                            use_container_width=True,
                        )

                    if submit_review:
                        if not operator_name.strip():
                            st.error("Operator Name is required for safety audit compliance.")
                        else:
                            with st.spinner("Recording operator sign-off..."):
                                review_res = client.review_recommendation(
                                    recommendation_id=rec_id,
                                    status_choice=decision,
                                    reviewed_by=operator_name.strip(),
                                    notes=review_notes.strip() if review_notes else None,
                                )
                                if review_res.get("success"):
                                    st.success(f"Recommendation `{rec_id}` marked as {decision.upper()} by {operator_name}!")
                                    st.rerun()
                                else:
                                    st.error(f"Sign-off failed: {review_res.get('error')}")

    with tab_history:
        all_res = client.list_recommendations(limit=100)
        all_recs = all_res.get("data", []) if all_res.get("success") else []
        reviewed = [r for r in all_recs if r.get("approval_status") in ("approved", "rejected")]

        if not reviewed:
            st.info("No historically reviewed recommendations found.")
        else:
            df = pd.DataFrame(reviewed)
            cols = [c for c in ["id", "title", "approval_status", "reviewed_by", "review_notes", "reviewed_at"] if c in df.columns]
            st.dataframe(df[cols], width="stretch", hide_index=True)
