from pathlib import Path
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
QUALITY = ROOT / "reports" / "quality_gate_report.json"
MODEL_INFO = ROOT / "outputs" / "best_model.json"
REPORTS = ROOT / "reports"

def main():
    if not QUALITY.exists():
        raise FileNotFoundError("Run 07_quality_gate.py first.")
    quality = json.loads(QUALITY.read_text(encoding="utf-8"))
    selected = json.loads(MODEL_INFO.read_text(encoding="utf-8")) if MODEL_INFO.exists() else {}
    status = "Production" if quality.get("status") == "passed" else "Rejected"
    report = {
        "model_name": "ChocolateRatingModel",
        "selected_model": selected.get("selected_model", "unknown"),
        "lifecycle_stage": status,
        "quality_gate": quality.get("status"),
        "updated_at_utc": datetime.now(timezone.utc).isoformat()
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "lifecycle_report.json").write_text(
        json.dumps(report, indent=4), encoding="utf-8"
    )
    print(f"MODEL LIFECYCLE STATUS: {status}")

if __name__ == "__main__":
    main()
