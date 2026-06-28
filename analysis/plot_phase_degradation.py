#!/usr/bin/env python3
"""Plot phase-specific accuracy degradation across compression configs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH_ROOT))

from analysis.plot_style import (  # noqa: E402
    CONFIG_COLORS,
    PHASES,
    apply_paper_style,
    ordered_configs,
    save_figure,
)
from shared.paths import COMPRESSION_RESULTS, PAPER_FIGURES, SHARED_SPLITS  # noqa: E402

METRIC_LABELS = {
    "phase_accuracy": "Phase accuracy (%)",
    "phase_f1": "Phase F1 (%)",
}


def load_split_counts() -> dict[str, int]:
    split_file = SHARED_SPLITS / "subject_split.json"
    if not split_file.exists():
        return {}
    data = json.loads(split_file.read_text())
    return {split: len(subjects) for split, subjects in data.items()}


def measured_points(manifest: dict) -> list[dict]:
    points = [p for p in manifest["points"] if p.get("macro_f1") is not None]
    if not points:
        raise ValueError("No measured configs in manifest.")
    return points


def phase_values(point: dict, metric: str) -> list[float | None]:
    values = point.get(metric) or {}
    return [None if values.get(phase) is None else float(values[phase]) * 100 for phase in PHASES]


def phase_deltas(point: dict, baseline: dict, metric: str) -> list[float | None]:
    values = point.get(metric) or {}
    base = baseline.get(metric) or {}
    deltas: list[float | None] = []
    for phase in PHASES:
        value = values.get(phase)
        base_value = base.get(phase)
        if value is None or base_value is None:
            deltas.append(None)
        else:
            deltas.append((float(value) - float(base_value)) * 100)
    return deltas


def plot_phase_degradation(
    manifest_path: Path,
    *,
    metric: str = "phase_accuracy",
    out_pdf: Path,
    out_png: Path | None = None,
) -> None:
    manifest = json.loads(manifest_path.read_text())
    points = measured_points(manifest)
    configs = ordered_configs([p["name"] for p in points])
    by_name = {p["name"]: p for p in points}
    baseline = by_name["FP32"]
    compressed = [name for name in configs if name != "FP32"]

    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharex=True)

    x = np.arange(len(PHASES))
    width = 0.8 / max(len(configs), 1)

    ax_abs = axes[0]
    for idx, name in enumerate(configs):
        offset = (idx - (len(configs) - 1) / 2) * width
        values = phase_values(by_name[name], metric)
        heights = [0.0 if v is None else v for v in values]
        bars = ax_abs.bar(
            x + offset,
            heights,
            width=width,
            label=name,
            color=CONFIG_COLORS.get(name, "#64748b"),
            edgecolor="white",
            linewidth=0.6,
        )
        for bar, value in zip(bars, values):
            if value is not None:
                ax_abs.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.4,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax_abs.set_xticks(x)
    ax_abs.set_xticklabels(PHASES)
    ax_abs.set_ylabel(METRIC_LABELS[metric])
    ax_abs.set_title("(a) Absolute phase performance (test split)")
    ax_abs.set_ylim(0, 105)
    ax_abs.legend(loc="lower right", ncol=2, frameon=False)

    ax_delta = axes[1]
    if compressed:
        width_delta = 0.8 / max(len(compressed), 1)
        for idx, name in enumerate(compressed):
            offset = (idx - (len(compressed) - 1) / 2) * width_delta
            deltas = phase_deltas(by_name[name], baseline, metric)
            heights = [0.0 if d is None else d for d in deltas]
            bars = ax_delta.bar(
                x + offset,
                heights,
                width=width_delta,
                label=name,
                color=CONFIG_COLORS.get(name, "#64748b"),
                edgecolor="white",
                linewidth=0.6,
            )
            for bar, delta in zip(bars, deltas):
                if delta is not None:
                    ax_delta.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (0.15 if delta >= 0 else -0.45),
                        f"{delta:+.1f}",
                        ha="center",
                        va="bottom" if delta >= 0 else "top",
                        fontsize=7,
                    )
    else:
        ax_delta.text(
            0.5,
            0.5,
            "Run compression evaluation to populate\ndegradation bars for INT8 / INT4 / Prune50.",
            ha="center",
            va="center",
            transform=ax_delta.transAxes,
            fontsize=10,
            color="#64748b",
        )

    ax_delta.axhline(0.0, color="#334155", linewidth=0.8)
    ax_delta.set_xticks(x)
    ax_delta.set_xticklabels(PHASES)
    ax_delta.set_ylabel("Change vs FP32 (percentage points)")
    ax_delta.set_title("(b) Phase degradation relative to FP32 (test split)")
    if compressed:
        ax_delta.legend(loc="lower right", ncol=2, frameon=False)

    split_counts = load_split_counts()
    test_n = split_counts.get("test")
    suffix = f" | test subjects: {test_n}" if test_n else ""
    fig.suptitle(f"Phase-specific compression impact{suffix}", y=1.02, fontsize=12)

    save_figure(fig, out_pdf, out_png)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=COMPRESSION_RESULTS / "manifest.json")
    p.add_argument("--metric", choices=list(METRIC_LABELS), default="phase_accuracy")
    p.add_argument("--out-pdf", type=Path, default=PAPER_FIGURES / "phase_degradation.pdf")
    p.add_argument("--out-png", type=Path, default=PAPER_FIGURES / "phase_degradation.png")
    args = p.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"Manifest not found: {args.manifest}. Run analysis/collect_pareto.py first.")
    plot_phase_degradation(args.manifest, metric=args.metric, out_pdf=args.out_pdf, out_png=args.out_png)
    print(f"Saved {args.out_pdf}")


if __name__ == "__main__":
    main()
