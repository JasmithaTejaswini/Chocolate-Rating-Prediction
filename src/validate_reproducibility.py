
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "chocolate_cleaned.csv"


def build_model(X):
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()

    prep = ColumnTransformer([
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler())
            ]),
            numeric
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical
        )
    ])

    return Pipeline([
        ("preprocessing", prep),
        ("model", RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=1
        ))
    ])


def main():
    df = pd.read_csv(DATA)
    X = df.drop(columns=["Rating"])
    y = df["Rating"]

    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_one = build_model(X)
    model_two = build_model(X)

    model_one.fit(X_train, y_train)
    model_two.fit(X_train, y_train)

    pred_one = model_one.predict(X_test)
    pred_two = model_two.predict(X_test)

    if np.allclose(pred_one, pred_two):
        print("Reproducibility validation passed.")
        print("Both runs produced identical predictions.")
    else:
        raise AssertionError("Predictions differ between repeated runs.")


if __name__ == "__main__":
    main()