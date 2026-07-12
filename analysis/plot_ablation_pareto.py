#!/usr/bin/env python3
"""Plot multi-point Pareto fronts from TCN-family ablation summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH_ROOT))

from analysis.plot_style import apply_paper_style, save_figure  # noqa: E402
from analysis.aggregate_ablation import pareto_frontier  # noqa: E402
from shared.paths import PAPER_FIGURES  # noqa: E402

MODEL_COLORS = {"tcn": "#2563eb"}


def plot_model(ax, points: list[dict], *, model: str, show_labels: bool) -> None:
    frontier, dominated = pareto_frontier(points)
    color = MODEL_COLORS.get(model, "#374151")
    if frontier:
        ax.plot(
            [p["size_kb"] for p in frontier],
            [p["macro_f1"] for p in frontier],
            "-",
            color=color,
            linewidth=1.2,
            alpha=0.85,
            label=f"{model.upper()} frontier",
        )
    for pt in frontier:
        ax.scatter(pt["size_kb"], pt["macro_f1"], s=42, color=color, edgecolor="white", linewidth=0.6, zorder=3)
        if show_labels:
            ax.annotate(pt["name"], (pt["size_kb"], pt["macro_f1"]), fontsize=6, xytext=(4, 4), textcoords="offset points")
    for pt in dominated:
        ax.scatter(
            pt["size_kb"],
            pt["macro_f1"],
            s=34,
            facecolor="white",
            edgecolor=color,
            linewidth=0.9,
            alpha=0.7,
            zorder=2,
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, default=RESEARCH_ROOT / "analysis" / "ablation_summary.json")
    p.add_argument("--out-pdf", type=Path, default=PAPER_FIGURES / "pareto_ablation.pdf")
    p.add_argument("--out-png", type=Path, default=PAPER_FIGURES / "pareto_ablation.png")
    p.add_argument("--combined", action="store_true", help="Overlay TinyTCN and TCN on one axis")
    args = p.parse_args()

    data = json.loads(args.summary.read_text())
    apply_paper_style()

    if args.combined:
        fig, ax = plt.subplots(figsize=(3.5, 2.7), layout="constrained")
        for model, payload in data["models"].items():
            plot_model(ax, payload["pareto_grid"]["points"], model=model, show_labels=False)
        ax.legend(fontsize=7, loc="lower left")
        ax.set_xlabel("Model size (KB)")
        ax.set_ylabel("Macro F1-score")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        save_figure(fig, args.out_pdf, args.out_png)
        print(f"Saved {args.out_pdf}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), layout="constrained")
    for ax, (model, payload) in zip(axes, data["models"].items()):
        plot_model(ax, payload["pareto_grid"]["points"], model=model, show_labels=True)
        ax.set_title(model.upper())
        ax.set_xlabel("Model size (KB)")
        ax.set_ylabel("Macro F1-score")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    save_figure(fig, args.out_pdf, args.out_png)
    print(f"Saved {args.out_pdf}")


if __name__ == "__main__":
    main()
