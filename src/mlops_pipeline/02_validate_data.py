from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "flavors_of_cacao.csv"
ARTIFACTS = ROOT / "artifacts"

def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_PATH}")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW_PATH)
    df.columns = (df.columns.astype(str).str.replace("\\xa0", " ", regex=False)
                  .str.replace("\\n", " ", regex=False).str.strip())
    checks = {
        "dataset_not_empty": len(df) > 0,
        "rating_column_exists": "Rating" in df.columns,
        "rating_has_numeric_values": (
            "Rating" in df.columns and
            pd.to_numeric(df["Rating"], errors="coerce").notna().any()
        ),
        "no_fully_empty_columns": not bool(df.isna().all().any())
    }
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "rows": int(len(df)),
        "columns": df.columns.tolist()
    }
    pd.DataFrame(
        [{"check": k, "passed": bool(v)} for k, v in checks.items()]
    ).to_csv(ARTIFACTS / "baseline_validation.csv", index=False)
    (ARTIFACTS / "data_validation_report.json").write_text(
        json.dumps(report, indent=4, default=str), encoding="utf-8"
    )
    print("DATA VALIDATION", report["status"].upper())
    if not all(checks.values()):
        raise ValueError(f"Data validation failed: {checks}")

if __name__ == "__main__":
    main()
