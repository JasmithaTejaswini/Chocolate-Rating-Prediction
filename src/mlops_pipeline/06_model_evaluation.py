from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "chocolate_cleaned.csv"
MODEL = ROOT / "models" / "best_model.joblib"
OUTPUTS = ROOT / "outputs"
ARTIFACTS = ROOT / "artifacts"

def main():
    if not DATA.exists() or not MODEL.exists():
        raise FileNotFoundError("Run preprocessing and training before evaluation.")
    df = pd.read_csv(DATA)
    if "Rating" not in df:
        raise ValueError("Target column 'Rating' is missing.")
    X, y = df.drop(columns=["Rating"]), df["Rating"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    model = joblib.load(MODEL)
    predictions = model.predict(X_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "MSE": float(mean_squared_error(y_test, predictions)),
        "RMSE": float(mean_squared_error(y_test, predictions) ** 0.5),
        "R2": float(r2_score(y_test, predictions)),
        "test_rows": int(len(y_test))
    }
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=4), encoding="utf-8"
    )
    pd.DataFrame({"actual_rating": y_test.to_numpy(),
                  "predicted_rating": predictions}).to_csv(
        OUTPUTS / "predictions.csv", index=False
    )
    (ARTIFACTS / "evaluation_report.json").write_text(
        json.dumps(metrics, indent=4), encoding="utf-8"
    )
    print("MODEL EVALUATION COMPLETED")
    print(json.dumps(metrics, indent=4))

if __name__ == "__main__":
    main()
