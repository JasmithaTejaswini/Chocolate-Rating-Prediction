
from pathlib import Path
import json

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]

MONITOR_REPORT = ROOT / "outputs" / "monitoring" / "monitoring_report.json"
EXECUTION_REPORT = ROOT / "outputs" / "lab13_execution_report.json"
PREDICTION_LOG = ROOT / "outputs" / "monitoring" / "prediction_log.csv"

st.set_page_config(
    page_title="Chocolate Rating MLOps Dashboard",
    page_icon="🍫",
    layout="wide"
)


def load_json(path):
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return {}


st.title("🍫 Chocolate Rating Prediction")
st.subheader("MLOps Monitoring Dashboard")

if st.button("🔄 Refresh dashboard"):
    st.rerun()

monitor = load_json(MONITOR_REPORT)
execution = load_json(EXECUTION_REPORT)

if not monitor:
    st.warning("Monitoring report not found. Run Lab 12 first.")
else:
    st.markdown("## Lab 12 – Model Monitoring")

    status = monitor.get("status", "unknown")
    drift = monitor.get("drift_detected", False)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Monitoring Status", status.upper())
    col2.metric(
        "Drift Status",
        "DETECTED" if drift else "NOT DETECTED"
    )
    col3.metric("Reference Records", monitor.get("reference_rows", 0))
    col4.metric("Incoming Records", monitor.get("incoming_rows", 0))

    st.markdown("### Prediction Statistics")

    p1, p2, p3 = st.columns(3)
    p1.metric("Mean Prediction", f"{monitor.get('prediction_mean', 0):.3f}")
    p2.metric("Minimum Prediction", f"{monitor.get('prediction_min', 0):.3f}")
    p3.metric("Maximum Prediction", f"{monitor.get('prediction_max', 0):.3f}")

    st.markdown("### Feature Drift Analysis")

    feature_drift = monitor.get("feature_drift", {})

    if feature_drift:
        drift_df = pd.DataFrame([
            {
                "Feature": feature,
                "PSI Score": result.get("psi"),
                "Drift Detected": result.get("drift_detected", False)
            }
            for feature, result in feature_drift.items()
        ])

        st.dataframe(drift_df, use_container_width=True)

        chart_df = drift_df.dropna(subset=["PSI Score"]).set_index("Feature")
        if not chart_df.empty:
            st.bar_chart(chart_df["PSI Score"])

        st.caption("PSI threshold: " + str(monitor.get("drift_threshold", 0.2)))

    st.markdown("### Model Performance")

    metrics = monitor.get("performance_metrics")

    if metrics:
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", f"{metrics.get('MAE', 0):.4f}")
        m2.metric("RMSE", f"{metrics.get('RMSE', 0):.4f}")
        r2 = metrics.get("R2")
        m3.metric("R²", f"{r2:.4f}" if r2 is not None else "Not available")
        st.caption(f"Evaluated on {metrics.get('labelled_rows', 0)} labelled records.")
    else:
        st.info("Actual ratings are not available. Performance metrics cannot be calculated.")

    st.markdown("### Retraining Recommendation")

    if monitor.get("retraining_recommended", False):
        st.warning(
            "Retraining is recommended for review. "
            "This dashboard does not automatically retrain or replace the model."
        )
    else:
        st.success("No retraining recommendation was generated.")


st.divider()
st.markdown("## Lab 13 – End-to-End Automation")

if execution:
    e1, e2, e3 = st.columns(3)

    e1.metric(
        "Overall Status",
        execution.get("overall_status", "unknown").upper()
    )
    e2.metric(
        "Monitoring Status",
        execution.get("monitoring_status", "unknown").upper()
    )
    e3.metric(
        "Monitoring Exit Code",
        execution.get("monitoring_exit_code", "N/A")
    )

    st.write("**Started:**", execution.get("started_at", "N/A"))
    st.write("**Completed:**", execution.get("completed_at", "N/A"))

    if execution.get("error"):
        st.error(execution["error"])
    else:
        st.success("Lab 13 automation completed successfully.")

    with st.expander("View complete Lab 13 execution report"):
        st.json(execution)
else:
    st.info("Lab 13 execution report not found. Run Lab 13 first.")


st.divider()
st.markdown("## Prediction Logs")

if PREDICTION_LOG.exists():
    try:
        logs = pd.read_csv(PREDICTION_LOG)
        st.metric("Total Logged Predictions", len(logs))
        st.dataframe(logs.tail(50), use_container_width=True)
    except Exception as error:
        st.error(f"Could not read prediction log: {error}")
else:
    st.info("No prediction log found. Run Lab 12 first.")

st.caption("Chocolate Rating MLOps Project | Model Monitoring and Automation")
