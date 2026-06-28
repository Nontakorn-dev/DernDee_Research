#!/usr/bin/env python3
"""Plot Pareto front for the paper."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH_ROOT))

from shared.paths import COMPRESSION_RESULTS, PAPER_FIGURES  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=COMPRESSION_RESULTS / "manifest.json")
    p.add_argument("--out-pdf", type=Path, default=PAPER_FIGURES / "pareto_front.pdf")
    p.add_argument("--out-png", type=Path, default=PAPER_FIGURES / "pareto_front.png")
    args = p.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"Run: python analysis/collect_pareto.py first ({args.manifest})")

    data = json.loads(args.manifest.read_text())
    measured = sorted(
        [p for p in data["points"] if p.get("macro_f1") is not None],
        key=lambda p: p["size_kb"],
    )
    pending = sorted(
        [p for p in data["points"] if p.get("macro_f1") is None],
        key=lambda p: p["size_kb"],
    )
    if not measured:
        raise SystemExit("No measured F1 points in manifest.")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    xs = [p["size_kb"] for p in measured]
    ys = [p["macro_f1"] for p in measured]
    ax.plot(xs, ys, "o-", color="#2563eb", linewidth=1.5, markersize=8)
    for x, y, pt in zip(xs, ys, measured):
        ax.annotate(pt["name"], (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
    if pending:
        ax.scatter([p["size_kb"] for p in pending], [0.02] * len(pending), marker="x", color="#9ca3af")
        for pt in pending:
            ax.annotate(pt["name"], (pt["size_kb"], 0.02), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Model size (KB)")
    ax.set_ylabel("Macro F1-score")
    ax.set_title("Pareto: macro F1 vs. model size")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out_pdf)
    fig.savefig(args.out_png, dpi=150)
    plt.close(fig)
    print(f"Saved {args.out_pdf}")


if __name__ == "__main__":
    main()
