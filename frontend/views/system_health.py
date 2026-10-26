"""System Health & Live API Test Suite for PlantPilot Streamlit."""

from __future__ import annotations
import time
import uuid
import streamlit as st
import pandas as pd
from typing import Any, Dict, List
from api_client import APIClient


def render_system_health(client: APIClient):
    st.title("System Health & API Diagnostics")
    st.markdown("Live telemetry on backend services, provider connectivity, and automated endpoint verification.")

    # 1. Health Checks
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.subheader("FastAPI Service Status")
        root_res = client.get_root()
        if root_res.get("success"):
            data = root_res.get("data", {})
            st.success(f"Backend Online: **{data.get('application')} v{data.get('version')}**")
            st.json(data)
        else:
            st.error(f"Backend Unreachable: {root_res.get('error')}")

    with col_h2:
        st.subheader("NVIDIA Provider Connectivity")
        check_provider = st.checkbox("Ping NVIDIA API endpoint", value=False)
        if check_provider:
            with st.spinner("Checking NVIDIA provider reachability..."):
                health_res = client.get_health(check_provider=True)
                if health_res.get("success"):
                    h_data = health_res.get("data", {})
                    if h_data.get("status") == "healthy":
                        st.success(f"NVIDIA API Reachable: `{h_data.get('provider')}`")
                    else:
                        st.warning(f"Degraded: {h_data.get('provider')}")
                    st.json(h_data)
                else:
                    st.error(f"Health check failed: {health_res.get('error')}")
        else:
            health_res = client.get_health(check_provider=False)
            if health_res.get("success"):
                st.info("Provider check skipped. Check box above to test NVIDIA connectivity.")
            else:
                st.error("Backend health probe failed.")

    st.divider()

    # 2. Automated Live API Test Suite
    st.subheader("Automated Live End-to-End Test Suite")
    st.markdown(
        "Run real-time automated verification across all PlantPilot backend endpoints and database schemas."
    )

    if st.button("Run Full System Diagnostic Suite", type="primary", use_container_width=True):
        results = run_live_diagnostics(client)
        render_diagnostic_results(results)


def run_live_diagnostics(client: APIClient) -> List[Dict[str, Any]]:
    suite = []
    test_uid = uuid.uuid4().hex[:6]

    # Test 1: Root
    t0 = time.time()
    res = client.get_root()
    dt = (time.time() - t0) * 1000
    suite.append({
        "Test Name": "Root Endpoint (GET /)",
        "Endpoint": "GET /",
        "Status": "PASSED" if res.get("success") and res.get("data", {}).get("status") == "running" else "FAILED",
        "Latency": f"{dt:.1f} ms",
        "Details": f"Application: {res.get('data', {}).get('application')}",
    })

    # Test 2: Health
    t0 = time.time()
    res = client.get_health(check_provider=False)
    dt = (time.time() - t0) * 1000
    suite.append({
        "Test Name": "Health Check (GET /health)",
        "Endpoint": "GET /health",
        "Status": "PASSED" if res.get("success") and res.get("data", {}).get("status") == "healthy" else "FAILED",
        "Latency": f"{dt:.1f} ms",
        "Details": f"Status: {res.get('data', {}).get('status')}",
    })

    # Test 3: Create Machine
    t0 = time.time()
    prod_id = f"T_{test_uid}"
    res = client.create_machine(product_id=prod_id, asset_type="M", status="active")
    dt = (time.time() - t0) * 1000
    created_mach_id = res.get("data", {}).get("id") if res.get("success") else None
    suite.append({
        "Test Name": "Asset Registration (POST /api/v1/machines)",
        "Endpoint": "POST /api/v1/machines",
        "Status": "PASSED" if res.get("success") and created_mach_id else "FAILED",
        "Latency": f"{dt:.1f} ms",
        "Details": f"Created Machine ID: {created_mach_id}",
    })

    # Test 4: Retrieve Machine
    if created_mach_id:
        t0 = time.time()
        res = client.get_machine(created_mach_id)
        dt = (time.time() - t0) * 1000
        suite.append({
            "Test Name": "Asset Retrieval (GET /api/v1/machines/{id})",
            "Endpoint": f"GET /api/v1/machines/{created_mach_id}",
            "Status": "PASSED" if res.get("success") and res.get("data", {}).get("product_id") == prod_id else "FAILED",
            "Latency": f"{dt:.1f} ms",
            "Details": f"Retrieved Product ID: {prod_id}",
        })

    # Test 5: List Machines
    t0 = time.time()
    res = client.list_machines(limit=5)
    dt = (time.time() - t0) * 1000
    suite.append({
        "Test Name": "List Fleet Assets (GET /api/v1/machines)",
        "Endpoint": "GET /api/v1/machines",
        "Status": "PASSED" if res.get("success") and isinstance(res.get("data"), list) else "FAILED",
        "Latency": f"{dt:.1f} ms",
        "Details": f"Total retrieved in page: {len(res.get('data', []))}",
    })

    # Test 6: Create Incident
    t0 = time.time()
    res = client.create_incident(
        description=f"Automated Diagnostic Incident {test_uid} on UDI 1",
        severity="medium",
        category="operational",
        machine_id=created_mach_id,
        metadata={"diagnostic": True, "test_id": test_uid},
    )
    dt = (time.time() - t0) * 1000
    created_inc_id = res.get("data", {}).get("id") if res.get("success") else None
    suite.append({
        "Test Name": "Create Incident (POST /api/v1/incidents)",
        "Endpoint": "POST /api/v1/incidents",
        "Status": "PASSED" if res.get("success") and created_inc_id else "FAILED",
        "Latency": f"{dt:.1f} ms",
        "Details": f"Created Incident ID: {created_inc_id}",
    })

    # Test 7: Retrieve Incident
    if created_inc_id:
        t0 = time.time()
        res = client.get_incident(created_inc_id)
        dt = (time.time() - t0) * 1000
        suite.append({
            "Test Name": "Retrieve Incident (GET /api/v1/incidents/{id})",
            "Endpoint": f"GET /api/v1/incidents/{created_inc_id}",
            "Status": "PASSED" if res.get("success") and res.get("data", {}).get("id") == created_inc_id else "FAILED",
            "Latency": f"{dt:.1f} ms",
            "Details": f"Verified Incident Status: {res.get('data', {}).get('status')}",
        })

    # Test 8: List Incidents
    t0 = time.time()
    res = client.list_incidents(limit=10)
    dt = (time.time() - t0) * 1000
    suite.append({
        "Test Name": "List Incidents (GET /api/v1/incidents)",
        "Endpoint": "GET /api/v1/incidents",
        "Status": "PASSED" if res.get("success") and len(res.get("data", [])) > 0 else "FAILED",
        "Latency": f"{dt:.1f} ms",
        "Details": f"Records returned: {len(res.get('data', []))}",
    })

    # Test 9: List Recommendations
    t0 = time.time()
    res = client.list_recommendations(limit=10)
    dt = (time.time() - t0) * 1000
    suite.append({
        "Test Name": "List Recommendations (GET /api/v1/recommendations)",
        "Endpoint": "GET /api/v1/recommendations",
        "Status": "PASSED" if res.get("success") and isinstance(res.get("data"), list) else "FAILED",
        "Latency": f"{dt:.1f} ms",
        "Details": f"Recommendations retrieved: {len(res.get('data', []))}",
    })

    # Test 10: List Investigations
    t0 = time.time()
    res = client.list_investigations(limit=10)
    dt = (time.time() - t0) * 1000
    suite.append({
        "Test Name": "List Investigations (GET /api/v1/investigations)",
        "Endpoint": "GET /api/v1/investigations",
        "Status": "PASSED" if res.get("success") and isinstance(res.get("data"), list) else "FAILED",
        "Latency": f"{dt:.1f} ms",
        "Details": f"Investigations in DB: {len(res.get('data', []))}",
    })

    return suite


def render_diagnostic_results(results: List[Dict[str, Any]]):
    passed_count = sum(1 for r in results if r["Status"] == "PASSED")
    total_count = len(results)

    if passed_count == total_count:
        st.success(f"ALL CLEAR: {passed_count}/{total_count} Automated Diagnostic Tests Passed Successfully!")
    else:
        st.warning(f"Diagnostic Complete: {passed_count}/{total_count} Passed.")

    st.dataframe(pd.DataFrame(results), width="stretch", hide_index=True)
