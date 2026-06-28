#!/usr/bin/env python3
"""Check processed/Xy has the label columns needed for training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

RESEARCH_ROOT = Path(__file__).resolve().parents[2]
SHARED = RESEARCH_ROOT / "shared"
sys.path.insert(0, str(SHARED))

from gait_labels import resolve_label_column  # noqa: E402
from paths import DATA_XY  # noqa: E402

CHANNELS_NEEDED = {
    "left": "left (phase_lt or legacy phase)",
    "right": "right (phase_rt)",
    "bilateral": "bilateral (phase_lt or legacy phase)",
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xy-root", type=Path, default=DATA_XY)
    p.add_argument("--channels", choices=["left", "right", "bilateral"], default="right")
    args = p.parse_args()

    trials = sorted(args.xy_root.rglob("S*.csv"))
    if not trials:
        raise SystemExit(f"No trials under {args.xy_root}")

    sample_cols = pd.read_csv(trials[0], nrows=0).columns.tolist()
    try:
        col = resolve_label_column(sample_cols, args.channels)
    except KeyError as exc:
        print(f"NOT READY for {CHANNELS_NEEDED[args.channels]}")
        print(exc)
        print("\nFix: restore NONAN_Dataset/ and run:")
        print("  python shared/preprocess_gaitprint.py --input-dir NONAN_Dataset --output-dir dataset/Xy")
        print("  python shared/scripts/upgrade_xy_labels.py --xy-root dataset/Xy")
        print("Then labels live in dataset/Xy/ permanently.")
        raise SystemExit(1) from exc

    print(f"OK — {len(trials)} trials, label column: {col!r}")
    print(f"Ready to train: --channels {args.channels}")


if __name__ == "__main__":
    main()
