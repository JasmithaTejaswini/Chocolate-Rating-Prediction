import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for script in ("src/preprocess.py", "src/train.py", "src/evaluate.py"):
    print(f"\nRunning {script} ...")
    result = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT)
    if result.returncode:
        raise SystemExit(f"Pipeline stopped: {script} failed.")
print("\nLab 3 baseline pipeline completed.")
