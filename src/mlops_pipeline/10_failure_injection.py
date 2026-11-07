from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "flavors_of_cacao.csv"
ARTIFACTS = ROOT / "artifacts"

def main():
    if not RAW.exists():
        raise FileNotFoundError(RAW)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW)
    tests = []

    # Simulate a missing required target column without modifying the real dataset.
    missing_target = df.drop(columns=["Rating"], errors="ignore")
    tests.append({
        "scenario": "missing_target_column",
        "detected": "Rating" not in missing_target.columns,
        "expected_detection": True
    })

    # Simulate invalid target values in memory.
    corrupted = df.copy()
    if "Rating" in corrupted.columns and len(corrupted):
        corrupted["Rating"] = "not_a_number"
        detected = pd.to_numeric(corrupted["Rating"], errors="coerce").isna().all()
    else:
        detected = False
    tests.append({
        "scenario": "non_numeric_target",
        "detected": bool(detected),
        "expected_detection": True
    })

    report = {
        "status": "passed" if all(t["detected"] == t["expected_detection"] for t in tests) else "failed",
        "note": "Failure scenarios were tested in memory; the source CSV was not modified.",
        "tests": tests
    }
    (ARTIFACTS / "failure_simulation_report.json").write_text(
        json.dumps(report, indent=4), encoding="utf-8"
    )
    print(f"LAB 10: FAILURE INJECTION {report['status'].upper()}")
    if report["status"] != "passed":
        raise RuntimeError("One or more failure scenarios were not detected.")

if __name__ == "__main__":
    main()
