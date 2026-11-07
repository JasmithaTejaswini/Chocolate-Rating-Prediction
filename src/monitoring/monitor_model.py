
from pathlib import Path
from datetime import datetime, timezone
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]

REFERENCE_DATA = ROOT / "data" / "processed" / "chocolate_cleaned.csv"
NEW_DATA = ROOT / "data" / "monitoring" / "new_data.csv"
MODEL_PATH = ROOT / "models" / "best_model.joblib"

OUTPUT_DIR = ROOT / "outputs" / "monitoring"
PREDICTION_LOG = OUTPUT_DIR / "prediction_log.csv"
REPORT_PATH = OUTPUT_DIR / "monitoring_report.json"

PSI_THRESHOLD = 0.20


def calculate_psi(reference, current, bins=10):
    """Calculate PSI for numeric or categorical columns."""
    ref = pd.Series(reference).dropna()
    cur = pd.Series(current).dropna()

    if len(ref) == 0 or len(cur) == 0:
        return None

    if pd.api.types.is_numeric_dtype(ref):
        edges = np.unique(
            np.quantile(ref.astype(float), np.linspace(0, 1, bins + 1))
        )

        if len(edges) < 2:
            return 0.0

        edges[0] = -np.inf
        edges[-1] = np.inf

        ref_counts = np.histogram(ref.astype(float), bins=edges)[0]
        cur_counts = np.histogram(cur.astype(float), bins=edges)[0]
    else:
        categories = set(ref.astype(str)) | set(cur.astype(str))
        ref_counts = np.array([
            (ref.astype(str) == category).sum()
            for category in categories
        ])
        cur_counts = np.array([
            (cur.astype(str) == category).sum()
            for category in categories
        ])

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)

    epsilon = 0.0001
    ref_pct = np.clip(ref_pct, epsilon, None)
    cur_pct = np.clip(cur_pct, epsilon, None)

    return float(np.sum(
        (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
    ))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not REFERENCE_DATA.exists():
        raise FileNotFoundError(f"Reference dataset not found: {REFERENCE_DATA}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not NEW_DATA.exists():
        report = {
            "status": "no_new_data",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": "Add incoming records to data/monitoring/new_data.csv."
        }
        REPORT_PATH.write_text(json.dumps(report, indent=4), encoding="utf-8")
        print(json.dumps(report, indent=4))
        return

    reference = pd.read_csv(REFERENCE_DATA)
    incoming = pd.read_csv(NEW_DATA)

    if "Rating" not in reference.columns:
        raise ValueError("Reference dataset must contain the Rating column.")

    if incoming.empty:
        raise ValueError("The incoming dataset is empty.")

    feature_columns = [
        column for column in reference.columns if column != "Rating"
    ]

    missing = [column for column in feature_columns if column not in incoming.columns]
    if missing:
        raise ValueError(
            "Incoming data is missing required model features: "
            + ", ".join(missing)
        )

    # Keep exactly the feature columns used by the existing model.
    X_reference = reference[feature_columns]
    X_current = incoming[feature_columns]

    model = joblib.load(MODEL_PATH)

    drift_results = {}
    for column in feature_columns:
        psi = calculate_psi(
            X_reference[column],
            X_current[column]
        )
        drift_results[column] = {
            "psi": psi,
            "drift_detected": (
                psi is not None and psi >= PSI_THRESHOLD
            )
        }

    predictions = model.predict(X_current)
    predictions = np.asarray(predictions, dtype=float)

    timestamp = datetime.now(timezone.utc).isoformat()

    log = X_current.copy()
    log["predicted_rating"] = predictions
    log["timestamp"] = timestamp

    if "Rating" in incoming.columns:
        log["actual_rating"] = incoming["Rating"].to_numpy()

    if PREDICTION_LOG.exists():
        log.to_csv(PREDICTION_LOG, mode="a", header=False, index=False)
    else:
        log.to_csv(PREDICTION_LOG, index=False)

    metrics = None
    if "Rating" in incoming.columns:
        labelled = incoming["Rating"].notna().to_numpy()

        if labelled.any():
            actual = incoming.loc[labelled, "Rating"].astype(float)
            predicted = predictions[labelled]

            metrics = {
                "labelled_rows": int(labelled.sum()),
                "MAE": float(mean_absolute_error(actual, predicted)),
                "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
                "R2": (
                    float(r2_score(actual, predicted))
                    if len(actual) >= 2 else None
                )
            }

    drift_detected = any(
        item["drift_detected"]
        for item in drift_results.values()
    )

    report = {
        "status": "completed",
        "timestamp": timestamp,
        "reference_rows": int(len(reference)),
        "incoming_rows": int(len(incoming)),
        "drift_threshold": PSI_THRESHOLD,
        "drift_detected": bool(drift_detected),
        "drifted_features": [
            name for name, result in drift_results.items()
            if result["drift_detected"]
        ],
        "feature_drift": drift_results,
        "prediction_mean": float(np.mean(predictions)),
        "prediction_min": float(np.min(predictions)),
        "prediction_max": float(np.max(predictions)),
        "performance_metrics": metrics,
        "retraining_recommended": bool(
            drift_detected and metrics is not None
        )
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    print("LAB 12 - MODEL MONITORING COMPLETED")
    print(json.dumps(report, indent=4))


if __name__ == "__main__":
    main()
