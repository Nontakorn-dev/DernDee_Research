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

# Offsets (in x-axis units) of each config within a phase group, in
# CONFIG_ORDER sequence, so the connecting line reads left-to-right as
# "increasing compression".
GROUP_OFFSETS = (-0.27, -0.09, 0.09, 0.27)


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


def _offsets(n: int) -> list[float]:
    if n == len(GROUP_OFFSETS):
        return list(GROUP_OFFSETS)
    span = GROUP_OFFSETS[-1] - GROUP_OFFSETS[0] if n > 1 else 0.0
    return [GROUP_OFFSETS[0] + span * (i / (n - 1)) if n > 1 else 0.0 for i in range(n)]


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
    fig, (ax_abs, ax_delta) = plt.subplots(1, 2, figsize=(7.0, 2.85), layout="constrained")

    x = np.arange(len(PHASES))
    offsets_abs = _offsets(len(configs))

    # Light vertical separators between phase groups for visual grouping
    # without relying on hatch fills or dense bar packing.
    for ax in (ax_abs, ax_delta):
        for boundary in range(len(PHASES) - 1):
            ax.axvline(boundary + 0.5, color="#e5e7eb", linewidth=0.8, zorder=0)

    legend_handles: dict[str, plt.Artist] = {}

    all_abs_values: list[float] = []
    for phase_idx in range(len(PHASES)):
        xs = [phase_idx + off for off in offsets_abs]
        ys = [phase_values(by_name[name], metric)[phase_idx] for name in configs]
        clean = [(xv, yv) for xv, yv in zip(xs, ys) if yv is not None]
        if clean:
            ax_abs.plot(*zip(*clean), "-", color="#b8bec6", linewidth=1.1, zorder=1)
        for name, xv, yv in zip(configs, xs, ys):
            if yv is None:
                continue
            all_abs_values.append(yv)
            handle = ax_abs.scatter(
                xv,
                yv,
                s=40,
                color=CONFIG_COLORS.get(name, "#64748b"),
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            legend_handles.setdefault(name, handle)

    ax_abs.set_xticks(x)
    ax_abs.set_xticklabels(PHASES)
    ax_abs.set_xlim(-0.5, len(PHASES) - 0.5)
    ax_abs.set_ylabel(METRIC_LABELS[metric])
    ax_abs.set_title("(a) Absolute phase performance", fontsize=9)
    if all_abs_values:
        pad = max(2.0, 0.15 * (max(all_abs_values) - min(all_abs_values)))
        ax_abs.set_ylim(min(all_abs_values) - pad, max(all_abs_values) + pad)

    offsets_delta = _offsets(len(compressed)) if compressed else []
    if compressed:
        for phase_idx in range(len(PHASES)):
            for name, off in zip(compressed, offsets_delta):
                delta = phase_deltas(by_name[name], baseline, metric)[phase_idx]
                if delta is None:
                    continue
                xv = phase_idx + off
                ax_delta.plot(
                    [xv, xv],
                    [0.0, delta],
                    "-",
                    color=CONFIG_COLORS.get(name, "#64748b"),
                    linewidth=2.0,
                    solid_capstyle="round",
                    zorder=2,
                    alpha=0.85,
                )
                handle = ax_delta.scatter(
                    xv,
                    delta,
                    s=34,
                    color=CONFIG_COLORS.get(name, "#64748b"),
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=3,
                )
                legend_handles.setdefault(name, handle)
                ax_delta.text(
                    xv,
                    delta + (0.22 if delta >= 0 else -0.22),
                    f"{delta:+.1f}",
                    ha="center",
                    va="bottom" if delta >= 0 else "top",
                    fontsize=6.5,
                )
    else:
        ax_delta.text(
            0.5,
            0.5,
            "Run compression evaluation to populate\ndegradation markers for INT8 / Prune50 / INT8+Prune50.",
            ha="center",
            va="center",
            transform=ax_delta.transAxes,
            fontsize=9,
            color="#64748b",
        )

    ax_delta.axhline(0.0, color="#333333", linewidth=0.8, zorder=1)
    ax_delta.set_xticks(x)
    ax_delta.set_xticklabels(PHASES)
    ax_delta.set_xlim(-0.5, len(PHASES) - 0.5)
    ax_delta.set_ylabel("\u0394 vs. FP32 (pp)")
    ax_delta.set_title("(b) Degradation relative to FP32", fontsize=9)
    deltas_all = [
        d
        for name in compressed
        for d in phase_deltas(by_name[name], baseline, metric)
        if d is not None
    ]
    if deltas_all:
        pad = max(0.7, 0.3 * (max(deltas_all) - min(deltas_all)))
        ax_delta.set_ylim(min(deltas_all) - pad, max(deltas_all) + pad)

    ordered_names = [name for name in configs if name in legend_handles]
    fig.legend(
        [legend_handles[name] for name in ordered_names],
        ordered_names,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=len(ordered_names),
        frameon=False,
        fontsize=8,
        handletextpad=0.4,
        columnspacing=1.3,
    )

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
