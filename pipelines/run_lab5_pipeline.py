
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run_script(script_name):
    script = ROOT / "src" / script_name
    subprocess.run([sys.executable, str(script)], check=True)


def main():
    run_script("preprocess.py")
    run_script("validate_data.py")
    run_script("validate_outputs.py")
    run_script("validate_reproducibility.py")

    print("Lab 5 pipeline completed successfully.")


if __name__ == "__main__":
    main()