from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "src" / "preprocess.py"

def main():
    if not SCRIPT.exists():
        raise FileNotFoundError(f"Existing preprocessing script not found: {SCRIPT}")
    subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT, check=True)
    print("PREPROCESSING COMPLETED using existing src/preprocess.py")

if __name__ == "__main__":
    main()
