
from pathlib import Path
from datetime import datetime, timezone
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]

MONITOR_SCRIPT = ROOT / "src" / "monitoring" / "monitor_model.py"
MONITOR_REPORT = ROOT / "outputs" / "monitoring" / "monitoring_report.json"

OUTPUT_DIR = ROOT / "outputs"
EXECUTION_REPORT = OUTPUT_DIR / "lab13_execution_report.json"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("LAB 13 - END-TO-END MLOPS AUTOMATION")
    print("=" * 65)

    execution = {
        "lab": "Lab 13 - End-to-End MLOps Automation",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "monitoring_status": "not_started",
        "overall_status": "running"
    }

    try:
        if not MONITOR_SCRIPT.exists():
            raise FileNotFoundError(MONITOR_SCRIPT)

        result = subprocess.run(
            [sys.executable, str(MONITOR_SCRIPT)],
            cwd=ROOT,
            capture_output=True,
            text=True
        )

        execution["monitoring_exit_code"] = result.returncode
        execution["monitoring_stdout"] = result.stdout[-4000:]
        execution["monitoring_stderr"] = result.stderr[-4000:]

        if result.returncode != 0:
            raise RuntimeError("Lab 12 monitoring failed.")

        if not MONITOR_REPORT.exists():
            raise FileNotFoundError(MONITOR_REPORT)

        with MONITOR_REPORT.open("r", encoding="utf-8") as file:
            monitoring = json.load(file)

        if monitoring.get("status") != "completed":
            raise RuntimeError("Monitoring report does not show completion.")

        execution["monitoring_status"] = monitoring["status"]
        execution["drift_detected"] = monitoring.get("drift_detected", False)
        execution["incoming_rows"] = monitoring.get("incoming_rows", 0)
        execution["drifted_features"] = monitoring.get("drifted_features", [])
        execution["performance_metrics"] = monitoring.get(
            "performance_metrics"
        )
        execution["retraining_recommended"] = monitoring.get(
            "retraining_recommended", False
        )

        execution["overall_status"] = "completed"

    except Exception as error:
        execution["error"] = str(error)
        execution["overall_status"] = "failed"

    finally:
        execution["completed_at"] = datetime.now(
            timezone.utc
        ).isoformat()

        EXECUTION_REPORT.write_text(
            json.dumps(execution, indent=4),
            encoding="utf-8"
        )

    print("\nLAB 13 EXECUTION REPORT")
    print(json.dumps(execution, indent=4))
    print(f"\nReport saved to: {EXECUTION_REPORT}")

    if execution["overall_status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
