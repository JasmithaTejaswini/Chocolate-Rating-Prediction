
from pathlib import Path
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "model_registry_report.csv"
MODEL_NAME = "ChocolateRatingModel"


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()

    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )

    if not versions:
        raise RuntimeError(
            "No registered model versions found. Register a model first."
        )

    rows = []

    for version in versions:
        rows.append({
            "model_name": version.name,
            "version": version.version,
            "status": version.status,
            "run_id": version.run_id,
            "source": version.source,
            "creation_timestamp": version.creation_timestamp,
            "current_stage": version.current_stage
        })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT, index=False)

    print("Model registry report generated.")
    print(f"Saved to: {OUTPUT}")


if __name__ == "__main__":
    main()