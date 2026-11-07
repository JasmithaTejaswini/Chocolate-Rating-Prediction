
import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "ChocolateRatingModel"
MODEL_ALIAS = "champion"


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()

    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )

    if not versions:
        raise RuntimeError(
            "No registered model version found. Run train_registry.py first."
        )

    latest_version = max(
        versions,
        key=lambda version: int(version.version)
    )

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=MODEL_ALIAS,
        version=latest_version.version
    )

    print("Model lifecycle update completed.")
    print(f"Model: {MODEL_NAME}")
    print(f"Alias: {MODEL_ALIAS}")
    print(f"Version: {latest_version.version}")


if __name__ == "__main__":
    main()