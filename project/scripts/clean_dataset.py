"""Run from project/: .venv/Scripts/python.exe -X utf8 scripts/clean_dataset.py."""

import argparse
from pathlib import Path

from movie_rating.data.clean import clean_dataset


def main():
    workspace = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Create a cleaned copy of MovieLens CSV files")
    parser.add_argument("--source", type=Path, default=workspace / "dataset")
    parser.add_argument("--output", type=Path, default=workspace / "dataset_cleaned")
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()
    try:
        clean_dataset(args.source, args.output, args.chunksize)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"Cleaning failed: {error}\n")


if __name__ == "__main__":
    main()
