from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "flavors_of_cacao.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "chocolate_cleaned.csv"
METADATA_PATH = PROCESSED_DIR / "dataset_metadata.json"

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_PATH}\n"
            "Place flavors_of_cacao.csv inside data/raw/."
        )

    df = pd.read_csv(RAW_PATH)
    df.columns = (df.columns.astype(str).str.replace("\\xa0", " ", regex=False)
                  .str.replace("\\n", " ", regex=False).str.strip())
    if "Rating" not in df.columns:
        raise ValueError(f"Expected target column 'Rating'. Found: {df.columns.tolist()}")

    before = len(df)
    df = df.drop_duplicates().copy()

    if "Cocoa Percent" in df.columns:
        df["Cocoa Percent"] = pd.to_numeric(
            df["Cocoa Percent"].astype(str).str.replace("%", "", regex=False),
            errors="coerce"
        )

    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df = df.dropna(subset=["Rating"])

    # REF is an identifier, not a model input.
    if "REF" in df.columns:
        df = df.drop(columns=["REF"])

    df.to_csv(OUTPUT_PATH, index=False)
    metadata = {
        "source_file": str(RAW_PATH.relative_to(ROOT)),
        "processed_file": str(OUTPUT_PATH.relative_to(ROOT)),
        "rows_before_cleaning": int(before),
        "rows_after_cleaning": int(len(df)),
        "columns": df.columns.tolist(),
        "target": "Rating",
        "duplicate_rows_removed": int(before - len(pd.read_csv(RAW_PATH).drop_duplicates()))
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=4), encoding="utf-8")
    print(f"Saved cleaned data: {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")
    print(f"Saved metadata: {METADATA_PATH}")

if __name__ == "__main__":
    main()
