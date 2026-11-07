from pathlib import Path
import subprocess
import sys
import json
import joblib
import mlflow
import mlflow.sklearn

ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = ROOT / "src" / "train.py"
MODEL_PATH = ROOT / "models" / "best_model.joblib"
BEST_INFO = ROOT / "outputs" / "best_model.json"

def main():
    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(TRAIN_SCRIPT)
    subprocess.run([sys.executable, str(TRAIN_SCRIPT)], cwd=ROOT, check=True)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Training did not create {MODEL_PATH}")
    model_name = "ChocolateRatingModel"
    info = json.loads(BEST_INFO.read_text(encoding="utf-8")) if BEST_INFO.exists() else {}
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("Chocolate_Rating_MLOps")
    with mlflow.start_run(run_name="lab7_model_training") as run:
        mlflow.log_param("target", "Rating")
        mlflow.log_param("selected_model", info.get("selected_model", "unknown"))
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")
        mlflow.set_tag("project", "Chocolate Bar Rating")
        mlflow.set_tag("model_name", model_name)
        print(f"MLflow run logged: {run.info.run_id}")
    print(f"MODEL TRAINING AND TRACKING COMPLETED: {info.get('selected_model', 'best model')}")

if __name__ == "__main__":
    main()
