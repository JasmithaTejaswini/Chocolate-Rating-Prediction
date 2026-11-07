
import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "Chocolate_Rating_Prediction"
MODEL_NAME = "ChocolateRatingModel"


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise RuntimeError(
            f"Experiment '{EXPERIMENT_NAME}' was not found. Run Lab 4 first."
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=20
    )

    selected_run = None

    for run in runs:
        if run.data.tags.get("mlflow.runName") == "RandomForest_Baseline":
            selected_run = run
            break

    if selected_run is None:
        raise RuntimeError("No completed RandomForest_Baseline run found.")

    model_uri = f"runs:/{selected_run.info.run_id}/model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    print("Model registered successfully.")
    print(f"Model name: {MODEL_NAME}")
    print(f"Version: {registered_model.version}")
    print(f"Run ID: {selected_run.info.run_id}")


if __name__ == "__main__":
    main()