
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    ROOT / "data" / "processed" / "chocolate_cleaned.csv",
    ROOT / "data" / "processed" / "dataset_metadata.json",
]


def main():
    missing = [str(path) for path in REQUIRED_FILES if not path.exists()]

    if missing:
        print("Output validation failed. Missing files:")
        for path in missing:
            print(f"- {path}")
        raise FileNotFoundError("Required preprocessing outputs are missing.")

    metadata_path = ROOT / "data" / "processed" / "dataset_metadata.json"

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    print("Output validation passed.")
    print("Required preprocessing files exist.")
    print("Metadata file is valid JSON.")


if __name__ == "__main__":
    main()