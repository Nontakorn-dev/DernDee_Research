#!/usr/bin/env python3
"""Create subject-level train/val/test split file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RESEARCH_ROOT = Path(__file__).resolve().parents[2]
SHARED = RESEARCH_ROOT / "shared"
sys.path.insert(0, str(SHARED))

from data.splits import discover_subjects, make_subject_split, save_split  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=RESEARCH_ROOT / "dataset" / "Xy")
    p.add_argument("--output", type=Path, default=SHARED / "splits" / "subject_split.csv")
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    subjects = discover_subjects(args.data_root)
    df = make_subject_split(
        subjects,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    save_split(df, args.output)

    for split in ("train", "val", "test"):
        subs = df.loc[df["split"] == split, "subject"].tolist()
        print(f"{split:5s}: {len(subs):2d} subjects  {subs}")
    print(f"\nSaved: {args.output}")
    print(f"Saved: {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
