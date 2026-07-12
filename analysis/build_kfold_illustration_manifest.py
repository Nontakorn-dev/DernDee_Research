#!/usr/bin/env python3
"""Build a compression Pareto/phase-degradation manifest from one k-fold run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH_ROOT))

from shared.paths import COMPRESSION_RESULTS  # noqa: E402

DEFAULT_CONFIGS = ("FP32", "INT8", "Prune25", "Prune50", "Prune75", "INT8+Prune50")


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(RESEARCH_ROOT))
    except ValueError:
        return str(path)


def load_run_metrics(metrics_path: Path) -> dict[str, Any]:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics: {metrics_path}")
    data = json.loads(metrics_path.read_text())
    name = str(data.get("name") or metrics_path.parent.name)
    return {
        "name": name,
        "size_kb": float(data["size_kb"]),
        "macro_f1": float(data["macro_f1"]),
        "accuracy": float(data.get("accuracy", 0.0)),
        "phase_f1": data.get("phase_f1", {}),
        "phase_accuracy": data.get("phase_accuracy", {}),
        "phase_support": data.get("phase_support", {}),
        "params": int(data.get("params", 0)),
        "note": str(data.get("method_note", "")),
        "artifact": _relative(metrics_path.parent / "model.pt"),
        "measured": True,
    }


def build_manifest(
    *,
    model: str,
    fold: int,
    seed: int,
    configs: list[str],
    runs_root: Path | None = None,
) -> dict[str, Any]:
    runs_root = runs_root or (RESEARCH_ROOT / "experiments" / "compression" / "runs" / model)
    run_dir = runs_root / f"fold{fold}_seed{seed}"
    if not run_dir.is_dir():
        raise FileNotFoundError(f"K-fold run directory not found: {run_dir}")

    points: list[dict[str, Any]] = []
    for name in configs:
        points.append(load_run_metrics(run_dir / name / "metrics.json"))

    fp32 = next((p for p in points if p["name"] == "FP32"), points[0])
    return {
        "source": f"{model}/fold{fold}_seed{seed}",
        "run_dir": _relative(run_dir),
        "params": fp32["params"],
        "points": points,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="tcn")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--configs",
        nargs="+",
        default=list(DEFAULT_CONFIGS),
        help="Configs to include (default: paper illustration set).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=COMPRESSION_RESULTS / "manifest_fold0_seed42.json",
    )
    args = p.parse_args()

    manifest = build_manifest(
        model=args.model,
        fold=args.fold,
        seed=args.seed,
        configs=args.configs,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {args.out} ({manifest['source']})")
    for pt in manifest["points"]:
        print(f"  {pt['name']:14}  {pt['size_kb']:6.2f} KB  F1={pt['macro_f1'] * 100:.2f}%")


if __name__ == "__main__":
    main()
