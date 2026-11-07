from pathlib import Path
import subprocess
import sys
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "chocolate_rating_pipeline_run_log.json"

STEPS = [
    "01_load_data.py",
    "02_validate_data.py",
    "03_preprocessing.py",
    "04_validate_outputs.py",
    "05_train_registry.py",
    "06_model_evaluation.py",
    "07_quality_gate.py",
    "08_automate_lifecycle.py",
    "09_inference.py",
    "10_failure_injection.py",
    "13_end_to_end_automation.py",
]

def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for name in STEPS:
        script = PIPELINE_DIR / name
        started = datetime.now(timezone.utc).isoformat()
        try:
            subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)
            results.append({"script": name, "status": "success", "started_at": started})
        except subprocess.CalledProcessError as exc:
            results.append({"script": name, "status": "failed",
                            "return_code": exc.returncode, "started_at": started})
            break
    report = {
        "status": "success" if len(results) == len(STEPS) and all(x["status"] == "success" for x in results) else "failed",
        "completed_steps": len(results),
        "total_steps": len(STEPS),
        "steps": results,
        "finished_at_utc": datetime.now(timezone.utc).isoformat()
    }
    LOG_FILE.write_text(json.dumps(report, indent=4), encoding="utf-8")
    print(f"Chocolate Rating pipeline: {report['status'].upper()}")
    print(f"Run log: {LOG_FILE}")
    if report["status"] != "success":
        raise SystemExit(1)

if __name__ == "__main__":
    main()
