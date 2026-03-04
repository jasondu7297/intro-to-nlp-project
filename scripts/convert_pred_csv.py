#!/usr/bin/env python3
"""Convert line-based predictions (pred.txt) to Kaggle submission CSV format."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred_txt",
        default="pred.txt",
        help="Path to line-based prediction file (default: pred.txt)",
    )
    parser.add_argument(
        "--output_csv",
        default="pred.csv",
        help="Path to output submission CSV (default: pred.csv)",
    )
    return parser.parse_args()


def read_predictions(path: Path) -> list[str]:
    with path.open("rt", encoding="utf-8", errors="ignore") as f:
        return [line.rstrip("\n\r") for line in f]


def write_submission(path: Path, predictions: list[str]) -> None:
    with path.open("wt", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "prediction"])
        writer.writeheader()
        for sample_id, pred in enumerate(predictions):
            writer.writerow({"id": sample_id, "prediction": pred})


def main() -> None:
    args = parse_args()
    pred_txt = Path(args.pred_txt)
    output_csv = Path(args.output_csv)

    predictions = read_predictions(pred_txt)
    write_submission(output_csv, predictions)
    print(f"Wrote {len(predictions)} rows to {output_csv}")


if __name__ == "__main__":
    main()
