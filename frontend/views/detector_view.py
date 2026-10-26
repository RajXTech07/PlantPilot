"""Dataset Explorer & Anomaly Detector view for PlantPilot Streamlit."""

from __future__ import annotations
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
from api_client import APIClient

# Ensure backend modules can be imported
backend_path = Path(__file__).resolve().parents[2] / "backend" / "app"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

try:
    from plantpilot.data_adapter import get_sensor_readings, get_evaluation_only
    from plantpilot.detector import detect, fit_references
    from plantpilot.eval.split import stratified_split
    DETECTOR_AVAILABLE = True
except Exception as e:
    DETECTOR_AVAILABLE = False
    IMPORT_ERR = str(e)


@st.cache_data
def load_dataset_references():
    """Load stratified training split and compute reference statistics."""
    csv_path = Path(__file__).resolve().parents[2] / "backend" / "data" / "ai4i2020.csv"
    if not csv_path.exists():
        csv_path = Path(__file__).resolve().parents[2] / "backend" / "ai4i2020.csv"
    train_rows, dev_rows, eval_rows = stratified_split(csv_path=str(csv_path))
    references = fit_references(train_rows)
    return references, csv_path


def render_detector_view(client: APIClient):
    st.title("Dataset Explorer & Statistical Anomaly Detector")
    st.markdown(
        "Interactive telemetry analysis over the **AI4I 2020 Predictive Maintenance Dataset** (10,000 industrial records). "
        "Evaluates sensor readings using dataset-derived reference statistics (IQR bounds fit on training split) and verified heuristics."
    )

    if not DETECTOR_AVAILABLE:
        st.error(f"Could not load backend detector modules: {IMPORT_ERR}")
        return

    try:
        references, csv_path = load_dataset_references()
    except Exception as exc:
        st.error(f"Error computing reference bounds: {exc}")
        return

    col_l1, col_l2 = st.columns([1, 2])
    with col_l1:
        udi_input = st.number_input(
            "Enter Dataset Record UDI (1 - 10,000)",
            min_value=1,
            max_value=10000,
            value=1,
            step=1,
            help="Unique Dataset Identifier from AI4I 2020 dataset.",
        )
    with col_l2:
        st.markdown(
            """
            **Quick Interesting Records:**
            - **UDI 1**: Tool Wear = 216 min (*Experimental Heuristic wear > 200 triggered*)
            - **UDI 51**: Rotational Speed Anomaly (*Multi-signal IQR*)
            - **UDI 161**: Heat Dissipation Temperature Delta anomaly
            """
        )

    # Fetch readings via quarantined data adapter
    try:
        sensor_data = get_sensor_readings(station_id=int(udi_input), csv_path=csv_path)
    except Exception as exc:
        st.error(f"Failed to retrieve sensor readings for UDI {udi_input}: {exc}")
        return

    st.subheader(f"Observed Telemetry for UDI {udi_input} ({sensor_data.get('product_id', 'Unknown')})")

    measurements = sensor_data.get("measurements", {})
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)

    with m_col1:
        st.metric("Air Temperature", f"{measurements.get('air_temperature', {}).get('value')} K")
    with m_col2:
        st.metric("Process Temperature", f"{measurements.get('process_temperature', {}).get('value')} K")
    with m_col3:
        st.metric("Rotational Speed", f"{measurements.get('rotational_speed', {}).get('value')} rpm")
    with m_col4:
        st.metric("Torque", f"{measurements.get('torque', {}).get('value')} Nm")
    with m_col5:
        wear_val = measurements.get('tool_wear', {}).get('value', 0)
        st.metric(
            "Tool Wear",
            f"{wear_val} min",
            delta="> 200 min Threshold" if wear_val > 200 else "Normal Range",
            delta_color="inverse" if wear_val > 200 else "normal",
        )

    st.divider()

    # Run multi-signal anomaly detector
    raw_row = {
        "UDI": str(udi_input),
        "Air temperature [K]": measurements.get("air_temperature", {}).get("value"),
        "Process temperature [K]": measurements.get("process_temperature", {}).get("value"),
        "Rotational speed [rpm]": measurements.get("rotational_speed", {}).get("value"),
        "Torque [Nm]": measurements.get("torque", {}).get("value"),
        "Tool wear [min]": measurements.get("tool_wear", {}).get("value"),
    }

    result = detect(row=raw_row, references=references)
    res_dict = result.to_dict()

    st.subheader("Statistical Anomaly Detector Analysis")
    flagged = res_dict.get("flagged_parameters", [])

    if not flagged:
        st.success("All sensor telemetry is within expected IQR training bounds. No anomaly detected.")
    else:
        st.error(f"Anomaly Detected! Flagged Signals: {', '.join(flagged)}")

    # Details table
    detail_records = []
    param_keys = [
        ("air_temperature_k", "Air Temperature", "K"),
        ("process_temperature_k", "Process Temperature", "K"),
        ("rotational_speed_rpm", "Rotational Speed", "rpm"),
        ("torque_nm", "Torque", "Nm"),
        ("tool_wear_min", "Tool Wear", "min"),
    ]

    for p_key, p_label, p_unit in param_keys:
        val = res_dict["observed_values"].get(p_key)
        ref = references.get(p_key, {})
        lower = ref.get("lower", 0.0)
        upper = ref.get("upper", 0.0)
        mean = ref.get("mean", 0.0)
        is_flagged = p_key in flagged

        detail_records.append({
            "Signal": p_label,
            "Observed Value": f"{val:.1f} {p_unit}",
            "Reference Range (IQR)": f"{lower:.1f} - {upper:.1f} {p_unit}",
            "Training Mean": f"{mean:.1f} {p_unit}",
            "Status": "FLAGGED ANOMALY" if is_flagged else "NORMAL",
            "Detection Method": res_dict["detection_methods"].get(p_key, "In-bounds"),
        })

    st.dataframe(pd.DataFrame(detail_records), width="stretch", hide_index=True)

    st.write(f"**Detector Explanation:** {res_dict['explanation']}")

    st.divider()

    # Label Quarantine Principle Demonstration
    st.subheader("Ground-Truth Label Quarantine Principle")
    st.info(
        "Under PlantPilot safety doctrine, **the AI agent and anomaly detector are strictly quarantined from ground-truth failure labels** "
        "(Machine failure, TWF, HDF, PWF, OSF, RNF). The agent only reasons over sensor measurements."
    )

    show_ground_truth = st.checkbox("Operator Supervisor Only: Inspect Quarantined Ground Truth for Evaluation")
    if show_ground_truth:
        eval_data = get_evaluation_only(station_id=int(udi_input), csv_path=csv_path)
        labels = eval_data.get("ground_truth_labels", {})
        st.warning("EVALUATION-ONLY LABELS (HIDDEN FROM AI AGENT):")
        st.json(labels)
