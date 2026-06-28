#!/usr/bin/env python3
"""Compute phase-specific degradation relative to the FP32 TinyTCN baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH_ROOT))

from shared.paths import COMPRESSION_RESULTS

PHASES = ("LR", "LS", "PSw", "Sw")


def percent(value: float | None) -> float | None:
    return None if value is None else round(value * 100, 2)


def delta_percent(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return round((value - baseline) * 100, 2)


def build_rows(manifest_path: Path, metric: str) -> list[dict]:
    data = json.loads(manifest_path.read_text())
    points = data["points"]
    fp32 = next((p for p in points if p["name"] == "FP32"), None)
    if fp32 is None:
        raise ValueError("Manifest does not contain an FP32 point.")

    baseline = fp32.get(metric) or {}
    rows = []
    for point in points:
        values = point.get(metric) or {}
        row = {
            "name": point["name"],
            "macro_f1": percent(point.get("macro_f1")),
            "accuracy": percent(point.get("accuracy")),
        }
        for phase in PHASES:
            row[f"{phase}_{metric}"] = percent(values.get(phase))
            row[f"{phase}_delta_pct"] = delta_percent(values.get(phase), baseline.get(phase))
        rows.append(row)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=COMPRESSION_RESULTS / "manifest.json")
    p.add_argument("--metric", choices=["phase_f1", "phase_accuracy"], default="phase_accuracy")
    p.add_argument("--out", type=Path, default=COMPRESSION_RESULTS / "phase_degradation.json")
    args = p.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"Manifest not found: {args.manifest}. Run analysis/collect_pareto.py first.")
    rows = build_rows(args.manifest, args.metric)
    args.out.write_text(json.dumps({"metric": args.metric, "rows": rows}, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
