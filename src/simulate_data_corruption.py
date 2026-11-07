from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "mlops_pipeline" / "10_failure_injection.py"

if __name__ == "__main__":
    subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT, check=True)
