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
    report = {
        "status": "success",
        "source_file": str(RAW_PATH.relative_to(ROOT)),
        "rows": int(len(df)),
        "columns_count": int(len(df.columns)),
        "columns": [str(c) for c in df.columns],
        "missing_values": {str(k): int(v) for k, v in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "data_types": {str(k): str(v) for k, v in df.dtypes.items()}
    }
    (ARTIFACTS / "data_load_report.json").write_text(
        json.dumps(report, indent=4), encoding="utf-8"
    )
    print("DATA LOADING COMPLETED")
    print(f"Dataset shape: {df.shape}")
    print(f"Report: {ARTIFACTS / 'data_load_report.json'}")

if __name__ == "__main__":
    main()
