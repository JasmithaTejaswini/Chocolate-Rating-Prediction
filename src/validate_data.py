
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "chocolate_cleaned.csv"


def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")

    df = pd.read_csv(DATA)

    if df.empty:
        raise ValueError("Dataset is empty.")

    if "Rating" not in df.columns:
        raise ValueError("Target column 'Rating' is missing.")

    if df["Rating"].isna().any():
        raise ValueError("Rating contains missing values.")

    if not pd.api.types.is_numeric_dtype(df["Rating"]):
        raise ValueError("Rating must be numeric.")

    print("Data validation passed.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Target: Rating")


if __name__ == "__main__":
    main()