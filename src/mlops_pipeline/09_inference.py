from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "chocolate_cleaned.csv"
MODEL = ROOT / "models" / "best_model.joblib"
OUTPUTS = ROOT / "outputs"

def main():
    if not DATA.exists() or not MODEL.exists():
        raise FileNotFoundError("Cleaned data or trained model is missing.")
    df = pd.read_csv(DATA)
    X = df.drop(columns=["Rating"])
    model = joblib.load(MODEL)
    predictions = model.predict(X)
    result = pd.DataFrame({"predicted_rating": predictions})
    if "Rating" in df:
        result.insert(0, "actual_rating", df["Rating"].to_numpy())
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUTS / "inference_predictions.csv", index=False)
    print(f"INFERENCE COMPLETED. Rows predicted: {len(result)}")

if __name__ == "__main__":
    main()
