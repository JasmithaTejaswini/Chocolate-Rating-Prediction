from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed" / "chocolate_cleaned.csv"
META = ROOT / "data" / "processed" / "dataset_metadata.json"
ARTIFACTS = ROOT / "artifacts"

def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    checks = {
        "cleaned_dataset_exists": PROCESSED.exists(),
        "metadata_exists": META.exists()
    }
    if checks["cleaned_dataset_exists"]:
        df = pd.read_csv(PROCESSED)
        checks["cleaned_dataset_not_empty"] = len(df) > 0
        checks["rating_column_exists"] = "Rating" in df.columns
        checks["rating_has_no_missing_values"] = (
            "Rating" in df.columns and not df["Rating"].isna().any()
        )
    else:
        df = None
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "processed_file": str(PROCESSED.relative_to(ROOT))
    }
    (ARTIFACTS / "preprocessing_summary_report.json").write_text(
        json.dumps(report, indent=4), encoding="utf-8"
    )
    print("PREPROCESSING OUTPUT VALIDATION", report["status"].upper())
    if not all(checks.values()):
        raise ValueError(f"Preprocessing output validation failed: {checks}")

if __name__ == "__main__":
    main()
