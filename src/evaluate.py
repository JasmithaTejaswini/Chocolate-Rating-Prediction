from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]

def main():
    data_path = ROOT / "data" / "processed" / "chocolate_cleaned.csv"
    model_path = ROOT / "models" / "best_model.joblib"
    output_dir = ROOT / "outputs"
    if not data_path.exists() or not model_path.exists():
        raise FileNotFoundError("Run preprocess.py and train.py first.")

    df = pd.read_csv(data_path)
    X, y = df.drop(columns=["Rating"]), df["Rating"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    model = joblib.load(model_path)
    predictions = model.predict(X_test)

    metrics = {
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "MSE": float(mean_squared_error(y_test, predictions)),
        "RMSE": float(mean_squared_error(y_test, predictions) ** 0.5),
        "R2": float(r2_score(y_test, predictions))
    }
    output_dir.mkdir(exist_ok=True)
    (output_dir / "evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=4), encoding="utf-8"
    )
    pd.DataFrame({"Actual_Rating": y_test, "Predicted_Rating": predictions}).to_csv(
        output_dir / "predictions.csv", index=False
    )
    print(json.dumps(metrics, indent=4))

if __name__ == "__main__":
    main()
