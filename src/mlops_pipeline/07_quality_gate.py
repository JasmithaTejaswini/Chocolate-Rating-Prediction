from pathlib import Path
import json
import math

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "outputs" / "evaluation_metrics.json"
MODEL = ROOT / "models" / "best_model.joblib"
REPORTS = ROOT / "reports"

def main():
    if not METRICS.exists() or not MODEL.exists():
        raise FileNotFoundError("Evaluation metrics or trained model are missing.")
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    checks = {
        "model_exists": MODEL.exists(),
        "mae_is_finite": math.isfinite(float(metrics["MAE"])),
        "rmse_is_finite": math.isfinite(float(metrics["RMSE"])),
        "r2_is_finite": math.isfinite(float(metrics["R2"]))
    }
    passed = all(checks.values())
    report = {"status": "passed" if passed else "failed",
              "checks": checks, "metrics": metrics}
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "quality_gate_report.json").write_text(
        json.dumps(report, indent=4), encoding="utf-8"
    )
    print(f"QUALITY GATE {report['status'].upper()}")
    if not passed:
        raise RuntimeError("Model quality gate failed.")

if __name__ == "__main__":
    main()
