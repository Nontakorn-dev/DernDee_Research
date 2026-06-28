#!/usr/bin/env python3
"""Build Pareto manifest from TinyTCN FP32 run + compression overrides."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH_ROOT))

from analysis.model_profile import count_parameters, estimate_sizes, load_tinytcn_from_checkpoint  # noqa: E402
from shared.paths import COMPRESSION_RESULTS, TINYTCN_RUNS  # noqa: E402

PHASES = ("LR", "LS", "PSw", "Sw")


def parse_macro_f1(report_path: Path) -> float | None:
    if not report_path.exists():
        return None
    m = re.search(r"macro avg\s+[\d.]+\s+[\d.]+\s+([\d.]+)", report_path.read_text())
    return float(m.group(1)) if m else None


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(RESEARCH_ROOT))
    except ValueError:
        return str(path)


def parse_report_metrics(report_path: Path, confusion_path: Path | None = None) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if not report_path.exists():
        return metrics

    phase_f1: dict[str, float] = {}
    phase_accuracy: dict[str, float] = {}
    phase_support: dict[str, int] = {}
    for line in report_path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] in PHASES and len(parts) >= 5:
            phase = parts[0]
            phase_accuracy[phase] = float(parts[2])
            phase_f1[phase] = float(parts[3])
            phase_support[phase] = int(parts[4])
        elif parts[0] == "accuracy" and len(parts) >= 3:
            metrics["accuracy"] = float(parts[1])
        elif parts[0] == "macro" and len(parts) >= 5 and parts[1] == "avg":
            metrics["macro_f1"] = float(parts[4])

    if confusion_path is not None and confusion_path.exists():
        cm = json.loads(confusion_path.read_text())
        for idx, phase in enumerate(PHASES):
            support = sum(cm[idx])
            phase_support[phase] = int(support)
            phase_accuracy[phase] = float(cm[idx][idx] / support) if support else 0.0
        total = sum(sum(row) for row in cm)
        correct = sum(cm[i][i] for i in range(min(len(cm), len(PHASES))))
        if total:
            metrics["accuracy"] = float(correct / total)
        metrics["confusion_matrix"] = cm

    if phase_f1:
        metrics["phase_f1"] = phase_f1
    if phase_accuracy:
        metrics["phase_accuracy"] = phase_accuracy
    if phase_support:
        metrics["phase_support"] = phase_support
    metrics["report"] = str(report_path)
    return metrics


def load_metric_sources(metrics_path: Path, overrides_path: Path) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if metrics_path.exists():
        raw = json.loads(metrics_path.read_text())
        raw_results = raw.get("results", raw)
        if isinstance(raw_results, dict):
            merged.update(raw_results)
    if overrides_path.exists():
        raw = json.loads(overrides_path.read_text())
        for name, value in raw.items():
            if value is None:
                continue
            if isinstance(value, dict):
                merged[name] = {**merged.get(name, {}), **value}
            else:
                merged[name] = {**merged.get(name, {}), "macro_f1": float(value)}
    return merged


def estimates_from_previous_manifest(path: Path):
    if not path.exists():
        return None, None
    raw = json.loads(path.read_text())
    params = raw.get("params")
    points = raw.get("points")
    if params is None or not points:
        return None, None
    return int(params), points


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", type=Path, default=TINYTCN_RUNS / "fp32_100hz")
    p.add_argument("--overrides", type=Path, default=COMPRESSION_RESULTS / "overrides.json")
    p.add_argument("--metrics", type=Path, default=COMPRESSION_RESULTS / "results" / "metrics.json")
    p.add_argument("--out", type=Path, default=COMPRESSION_RESULTS / "manifest.json")
    args = p.parse_args()

    ckpt = args.checkpoint_dir / "best_model.pt"
    previous_params, previous_points = estimates_from_previous_manifest(args.out)
    if ckpt.exists():
        model = load_tinytcn_from_checkpoint(ckpt)
        params = count_parameters(model)
        estimates = [est.to_dict() for est in estimate_sizes(params)]
    elif previous_params is not None and previous_points is not None:
        params = previous_params
        estimates = previous_points
        print(f"Checkpoint not found ({ckpt}); reusing size estimates from {args.out}")
    else:
        raise SystemExit(f"Checkpoint not found and no reusable manifest exists: {ckpt}")

    fp32_metrics = parse_report_metrics(
        args.checkpoint_dir / "test_report.txt",
        args.checkpoint_dir / "test_confusion_matrix.json",
    )
    if not fp32_metrics:
        val_metrics = parse_report_metrics(
            args.checkpoint_dir / "val_report.txt",
            args.checkpoint_dir / "val_confusion_matrix.json",
        )
        fp32_metrics = val_metrics
    if "macro_f1" not in fp32_metrics:
        fallback_f1 = parse_macro_f1(args.checkpoint_dir / "test_report.txt") or parse_macro_f1(
            args.checkpoint_dir / "val_report.txt"
        )
        if fallback_f1 is not None:
            fp32_metrics["macro_f1"] = fallback_f1

    metric_sources = load_metric_sources(args.metrics, args.overrides)

    points = []
    for est in estimates:
        name = est["name"]
        source = dict(metric_sources.get(name, {}))
        if name == "FP32":
            source = {**fp32_metrics, **source}

        size_kb = float(source.get("size_kb", est["size_kb"]))
        point = {
            "name": name,
            "size_kb": size_kb,
            "macro_f1": source.get("macro_f1"),
            "accuracy": source.get("accuracy"),
            "phase_f1": source.get("phase_f1"),
            "phase_accuracy": source.get("phase_accuracy"),
            "phase_support": source.get("phase_support"),
            "params": int(source.get("params", est["params"])),
            "note": source.get("method_note", est.get("note", "")),
            "artifact": source.get("artifact"),
            "measured": source.get("macro_f1") is not None,
        }
        points.append(point)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "checkpoint_dir": _relative(args.checkpoint_dir),
        "params": params,
        "points": points,
    }, indent=2))
    print(f"Wrote {args.out}")
    for pt in points:
        f1s = f"{pt['macro_f1']:.4f}" if pt["macro_f1"] is not None else "—"
        print(f"  {pt['name']:12}  {pt['size_kb']:6.1f} KB  F1={f1s}")


if __name__ == "__main__":
    main()
