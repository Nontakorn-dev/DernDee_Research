#!/usr/bin/env python3
"""Preprocess NONAN GaitPrint CSVs for TinyTCN gait phase classification.

Keeps DernDee-compatible dual-shank IMU features (accel + gyro) and
foot contact ground-truth labels. Contact is NOT a model input — use
build_labels.py to derive HS/TO and 4-phase labels from contact.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TIME_COLUMN = "time"

LEFT_SHANK_COLUMNS = [
    "Noraxon MyoMotion-Segments-Shank LT-Acceleration-x (mG)",
    "Noraxon MyoMotion-Segments-Shank LT-Acceleration-y (mG)",
    "Noraxon MyoMotion-Segments-Shank LT-Acceleration-z (mG)",
    "Noraxon MyoMotion-Segments-Shank LT-Gyroscope-x (deg/s)",
    "Noraxon MyoMotion-Segments-Shank LT-Gyroscope-y (deg/s)",
    "Noraxon MyoMotion-Segments-Shank LT-Gyroscope-z (deg/s)",
]

RIGHT_SHANK_COLUMNS = [
    "Noraxon MyoMotion-Segments-Shank RT-Acceleration-x (mG)",
    "Noraxon MyoMotion-Segments-Shank RT-Acceleration-y (mG)",
    "Noraxon MyoMotion-Segments-Shank RT-Acceleration-z (mG)",
    "Noraxon MyoMotion-Segments-Shank RT-Gyroscope-x (deg/s)",
    "Noraxon MyoMotion-Segments-Shank RT-Gyroscope-y (deg/s)",
    "Noraxon MyoMotion-Segments-Shank RT-Gyroscope-z (deg/s)",
]

CONTACT_COLUMNS = [
    "Contact LT",
    "Contact RT",
]

IMU_COLUMNS = LEFT_SHANK_COLUMNS + RIGHT_SHANK_COLUMNS
OUTPUT_COLUMNS = [TIME_COLUMN, *IMU_COLUMNS, *CONTACT_COLUMNS]


def preprocess_file(input_path: Path, output_path: Path) -> dict:
    """Extract TinyTCN columns from one raw GaitPrint CSV."""
    df = pd.read_csv(input_path, usecols=OUTPUT_COLUMNS)[OUTPUT_COLUMNS]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return {
        "file": input_path.name,
        "rows": len(df),
        "columns": len(df.columns),
        "input_mb": round(input_path.stat().st_size / 1024**2, 2),
        "output_mb": round(output_path.stat().st_size / 1024**2, 2),
    }


def discover_csv_files(input_dir: Path) -> list[Path]:
    return sorted(input_dir.rglob("*.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Root directory containing subject folders (default: script directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: overwrite input files in place)",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="Process only one subject folder, e.g. S034",
    )
    args = parser.parse_args()

    if args.subject:
        search_root = args.input_dir / args.subject
    else:
        search_root = args.input_dir

    csv_files = discover_csv_files(search_root)
    if not csv_files:
        raise SystemExit(f"No CSV files found under {search_root}")

    print(f"Processing {len(csv_files)} file(s)")
    print(f"Output columns ({len(OUTPUT_COLUMNS)}): {OUTPUT_COLUMNS}\n")

    summaries = []
    for input_path in csv_files:
        if args.output_dir is None:
            output_path = input_path
            temp_path = input_path.with_suffix(".csv.tmp")
            summary = preprocess_file(input_path, temp_path)
            temp_path.replace(output_path)
        else:
            rel = input_path.relative_to(args.input_dir)
            output_path = args.output_dir / rel
            summary = preprocess_file(input_path, output_path)

        summaries.append(summary)
        print(
            f"  {summary['file']}: {summary['rows']} rows, "
            f"{summary['input_mb']} MB -> {summary['output_mb']} MB"
        )

    total_in = sum(s["input_mb"] for s in summaries)
    total_out = sum(s["output_mb"] for s in summaries)
    print(f"\nDone. Total size: {total_in:.1f} MB -> {total_out:.1f} MB")


if __name__ == "__main__":
    main()
