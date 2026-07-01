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

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH_ROOT))

from analysis.plot_style import CONFIG_COLORS, apply_paper_style, save_figure  # noqa: E402
from shared.paths import COMPRESSION_RESULTS, PAPER_FIGURES  # noqa: E402

# Manual label offsets (points) tuned so annotations do not overlap given the
# known clustering of INT8 (12.7 KB) and Prune50 (13.4 KB).
LABEL_OFFSETS = {
    "FP32": (8, -3),
    "INT8": (-6, 9),
    "Prune50": (10, 6),
    "INT8+Prune50": (-3, -18),
}
LABEL_HA = {
    "FP32": "left",
    "INT8": "right",
    "Prune50": "left",
    "INT8+Prune50": "center",
}


def pareto_frontier(points: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split points into the Pareto-optimal frontier and dominated points.

    A point is on the frontier if no smaller-or-equal-size point achieves an
    equal-or-higher macro F1 (i.e. it is not dominated).
    """
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=COMPRESSION_RESULTS / "manifest.json")
    p.add_argument("--out-pdf", type=Path, default=PAPER_FIGURES / "pareto_front.pdf")
    p.add_argument("--out-png", type=Path, default=PAPER_FIGURES / "pareto_front.png")
    args = p.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"Run: python analysis/collect_pareto.py first ({args.manifest})")

    data = json.loads(args.manifest.read_text())
    measured = [pt for pt in data["points"] if pt.get("macro_f1") is not None]
    pending = sorted(
        [pt for pt in data["points"] if pt.get("macro_f1") is None],
        key=lambda pt: pt["size_kb"],
    )
    if not measured:
        raise SystemExit("No measured F1 points in manifest.")

    frontier, dominated = pareto_frontier(measured)

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(3.35, 2.55), layout="constrained")

    fx = [pt["size_kb"] for pt in frontier]
    fy = [pt["macro_f1"] for pt in frontier]
    ax.plot(fx, fy, "-", color="#4b5563", linewidth=1.1, zorder=1)
    for pt in frontier:
        ax.scatter(
            pt["size_kb"],
            pt["macro_f1"],
            s=46,
            color=CONFIG_COLORS.get(pt["name"], "#374151"),
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
            label="_nolegend_",
        )
        dx, dy = LABEL_OFFSETS.get(pt["name"], (6, 6))
        ax.annotate(
            pt["name"],
            (pt["size_kb"], pt["macro_f1"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=7.5,
            ha=LABEL_HA.get(pt["name"], "left"),
        )

    for pt in dominated:
        ax.scatter(
            pt["size_kb"],
            pt["macro_f1"],
            s=42,
            facecolor="white",
            edgecolor="#9ca3af",
            linewidth=1.1,
            marker="o",
            zorder=2,
        )
        dx, dy = LABEL_OFFSETS.get(pt["name"], (6, 6))
        ax.annotate(
            f"{pt['name']}\n(dominated)",
            (pt["size_kb"], pt["macro_f1"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=7,
            color="#6b7280",
            style="italic",
            ha=LABEL_HA.get(pt["name"], "left"),
        )

    if pending:
        ax.scatter(
            [pt["size_kb"] for pt in pending],
            [0.02] * len(pending),
            marker="x",
            color="#9ca3af",
        )
        for pt in pending:
            ax.annotate(
                pt["name"],
                (pt["size_kb"], 0.02),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=7,
            )

    all_f1 = fy + [pt["macro_f1"] for pt in dominated]
    y_lo = max(0.0, min(all_f1) - 0.03)
    y_hi = min(1.0, max(all_f1) + 0.03)
    ax.set_ylim(y_lo, y_hi)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))

    x_max = max(pt["size_kb"] for pt in measured)
    ax.set_xlim(0, x_max * 1.15)

    ax.set_xlabel("Model size (KB)")
    ax.set_ylabel("Macro F1-score")
    ax.text(
        0.02,
        0.03,
        "\u2191 smaller & higher is better",
        transform=ax.transAxes,
        fontsize=6.5,
        style="italic",
        color="#6b7280",
        ha="left",
        va="bottom",
    )

    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, args.out_pdf, args.out_png)
    print(f"Saved {args.out_pdf}")


if __name__ == "__main__":
    main()
