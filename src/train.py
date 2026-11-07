from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "chocolate_cleaned.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run: python src/preprocess.py")
    df = pd.read_csv(DATA_PATH)
    X, y = df.drop(columns=["Rating"]), df["Rating"]
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    prep = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.01, max_iter=10000),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42)
    }

    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    rows, fitted = [], {}
    for name, estimator in models.items():
        model = Pipeline([("preprocessing", prep), ("model", estimator)])
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "MSE": mean_squared_error(y_test, pred),
            "RMSE": mean_squared_error(y_test, pred) ** 0.5,
            "R2": r2_score(y_test, pred)
        })
        fitted[name] = model

    results = pd.DataFrame(rows).sort_values("R2", ascending=False)
    results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    best_name = results.iloc[0]["Model"]
    joblib.dump(fitted[best_name], MODEL_DIR / "best_model.joblib")
    (OUTPUT_DIR / "best_model.json").write_text(
        json.dumps({"selected_model": best_name}, indent=4), encoding="utf-8"
    )
    print(results.to_string(index=False))
    print(f"\nSaved best model: {best_name}")

if __name__ == "__main__":
    main()
