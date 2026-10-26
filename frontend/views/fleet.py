"""Fleet & Asset Management view for PlantPilot Streamlit."""

from __future__ import annotations
import streamlit as st
import pandas as pd
from typing import Any, Dict
from api_client import APIClient


def render_fleet(client: APIClient):
    st.title("Industrial Machine Fleet")
    st.markdown("Track and register physical manufacturing equipment and machining centers.")

    tab_fleet, tab_register = st.tabs(["Active Assets Inventory", "Register Industrial Asset"])

    with tab_fleet:
        machines_res = client.list_machines(limit=100)
        if not machines_res.get("success"):
            st.error(f"Failed to fetch fleet assets: {machines_res.get('error')}")
            return

        machines = machines_res.get("data", [])
        if not machines:
            st.info("No machines registered yet. Use the 'Register Industrial Asset' tab to add equipment.")
            return

        st.caption(f"Total Registered Equipment: {len(machines)}")
        df = pd.DataFrame(machines)
        display_cols = [c for c in ["id", "product_id", "type", "status", "created_at"] if c in df.columns]
        st.dataframe(df[display_cols], width="stretch", hide_index=True)

        st.divider()
        st.subheader("Asset Telemetry Quick Inspect")
        selected_m = st.selectbox("Select Machine", [f"{m['id']} ({m['product_id']})" for m in machines])
        if selected_m:
            mach_id = selected_m.split(" ")[0]
            mach = next((m for m in machines if m["id"] == mach_id), None)
            if mach:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Asset ID", mach.get("id"))
                c2.metric("Product ID", mach.get("product_id"))
                c3.metric("Type Variant", mach.get("type"))
                c4.metric("Status", mach.get("status").upper())

    with tab_register:
        st.subheader("Register Machine into Operations Network")
        with st.form("register_machine_form"):
            col_p, col_t = st.columns(2)
            with col_p:
                product_id = st.text_input("Product ID (AI4I Format) *", value="M14860", help="e.g. L47181, M14860, H39401")
            with col_t:
                asset_type = st.selectbox("Quality Variant Type *", ["L", "M", "H"], index=1, help="L (Low 50%), M (Medium 30%), H (High 20%)")

            status = st.selectbox("Initial Operational Status", ["active", "maintenance", "degraded", "offline"])

            submit_mach = st.form_submit_button("Register Machine Asset", type="primary", use_container_width=True)
            if submit_mach:
                if not product_id.strip():
                    st.error("Product ID is required.")
                else:
                    with st.spinner("Registering asset in SQLite registry..."):
                        res = client.create_machine(
                            product_id=product_id.strip(),
                            asset_type=asset_type,
                            status=status,
                        )
                        if res.get("success"):
                            st.success(f"Asset successfully registered! ID: `{res['data']['id']}`")
                            st.json(res["data"])
                        else:
                            st.error(f"Failed to register asset: {res.get('error')}")
