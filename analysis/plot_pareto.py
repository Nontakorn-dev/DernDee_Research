#!/usr/bin/env python3
"""Plot Pareto front (macro F1 vs. model size) for the paper."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH_ROOT))

from analysis.plot_style import CONFIG_COLORS, apply_paper_style, ordered_configs, save_figure  # noqa: E402
from shared.paths import COMPRESSION_RESULTS, PAPER_FIGURES  # noqa: E402

# Tiny horizontal dodge (KB) so INT8 and Prune50 (~13 KB) remain visually separable.
DISPLAY_X_DODGE_KB = {
    "INT8": -0.35,
    "Prune50": 0.35,
}


def display_x(point: dict) -> float:
    return float(point["size_kb"]) + DISPLAY_X_DODGE_KB.get(point["name"], 0.0)


def pareto_frontier(points: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return Pareto-optimal and dominated points (minimize size, maximize F1)."""
    ordered = sorted(points, key=lambda p: p["size_kb"])
    frontier: list[dict] = []
    dominated: list[dict] = []
    best_f1_so_far = float("-inf")
    for pt in ordered:
        if pt["macro_f1"] > best_f1_so_far:
            frontier.append(pt)
            best_f1_so_far = pt["macro_f1"]
        else:
            dominated.append(pt)
    return frontier, dominated


def plot_pareto(
    manifest_path: Path,
    *,
    out_pdf: Path,
    out_png: Path | None = None,
) -> None:
    data = json.loads(manifest_path.read_text())
    measured = [pt for pt in data["points"] if pt.get("macro_f1") is not None]
    if not measured:
        raise SystemExit("No measured F1 points in manifest.")

    measured = sorted(measured, key=lambda p: ordered_configs([p["name"]]).index(p["name"]))
    frontier, dominated = pareto_frontier(measured)
    frontier_names = {pt["name"] for pt in frontier}

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(3.45, 2.65), layout="constrained")

    if len(frontier) >= 2:
        ax.plot(
            [display_x(pt) for pt in frontier],
            [pt["macro_f1"] for pt in frontier],
            "-",
            color="#4b5563",
            linewidth=1.2,
            zorder=1,
            alpha=0.9,
        )

    legend_handles: list[Line2D] = []
    for pt in measured:
        color = CONFIG_COLORS.get(pt["name"], "#374151")
        on_frontier = pt["name"] in frontier_names
        ax.scatter(
            display_x(pt),
            pt["macro_f1"],
            s=52 if on_frontier else 40,
            color=color if on_frontier else "white",
            edgecolor=color,
            linewidth=1.2 if on_frontier else 1.0,
            zorder=3 if on_frontier else 2,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color if on_frontier else "white",
                markeredgecolor=color,
                markeredgewidth=1.1,
                markersize=6,
                label=pt["name"],
            )
        )

    all_f1 = [pt["macro_f1"] for pt in measured]
    y_lo = max(0.0, min(all_f1) - 0.012)
    y_hi = min(1.0, max(all_f1) + 0.008)
    ax.set_ylim(y_lo, y_hi)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))

    x_vals = [display_x(pt) for pt in measured]
    x_max = max(pt["size_kb"] for pt in measured)
    ax.set_xlim(-1.0, x_max * 1.08)
    ax.set_xlabel("Estimated model size (KB)")
    ax.set_ylabel("Macro F1-score")

    ax.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=6.5,
        handletextpad=0.35,
        borderpad=0.35,
        labelspacing=0.35,
        frameon=True,
        facecolor="white",
        edgecolor="#d1d5db",
        framealpha=0.95,
    )

    save_figure(fig, out_pdf, out_png)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=COMPRESSION_RESULTS / "manifest.json")
    p.add_argument("--out-pdf", type=Path, default=PAPER_FIGURES / "pareto_front.pdf")
    p.add_argument("--out-png", type=Path, default=PAPER_FIGURES / "pareto_front.png")
    args = p.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"Run: python analysis/build_kfold_illustration_manifest.py first ({args.manifest})")

    plot_pareto(args.manifest, out_pdf=args.out_pdf, out_png=args.out_png)
    print(f"Saved {args.out_pdf}")


if __name__ == "__main__":
    main()
