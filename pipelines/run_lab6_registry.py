
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run_script(script_name):
    script = ROOT / "src" / script_name
    subprocess.run([sys.executable, str(script)], check=True)


def main():
    run_script("train_registry.py")
    run_script("automate_lifecycle.py")
    run_script("generate_registry_report.py")

    print("Lab 6 model registry pipeline completed successfully.")


if __name__ == "__main__":
    main()