
from pathlib import Path
import sys
import subprocess

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "chocolate_cleaned.csv"


def main():
    # Run preprocessing
    subprocess.run(
        [sys.executable, str(ROOT / "src" / "preprocess.py")],
        check=True
    )

    # Load cleaned dataset
    df = pd.read_csv(DATA)

    X = df.drop(columns=["Rating"])
    y = df["Rating"]

    # Identify numeric and categorical columns
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    # Preprocessing pipelines
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
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore")
                )
            ]),
            categorical
        )
    ])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Define model parameters
    params = {
        "n_estimators": 200,
        "random_state": 42
    }

    # Build complete model pipeline
    model = Pipeline([
        ("preprocessing", prep),
        (
            "model",
            RandomForestRegressor(**params, n_jobs=-1)
        )
    ])

    # Configure MLflow experiment
    mlflow.set_experiment("Chocolate_Rating_Prediction")

    with mlflow.start_run(run_name="RandomForest_Baseline"):

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        pred = model.predict(X_test)

        # Calculate evaluation metrics
        metrics = {
            "mae": mean_absolute_error(y_test, pred),
            "mse": mean_squared_error(y_test, pred),
            "rmse": mean_squared_error(y_test, pred) ** 0.5,
            "r2": r2_score(y_test, pred)
        }

        # Log parameters and metrics
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        # Log the trained model
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree"
            ]
        )

        print("Logged MLflow run. Metrics:", metrics)

    print("Lab 4 MLflow tracking pipeline completed successfully.")


if __name__ == "__main__":
    main()