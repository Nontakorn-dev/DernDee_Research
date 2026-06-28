#!/usr/bin/env python3
"""Plot TinyTCN training convergence curves."""

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

from analysis.plot_style import apply_paper_style, save_figure, split_caption  # noqa: E402
from shared.paths import PAPER_FIGURES, SHARED_SPLITS, TINYTCN_RUNS  # noqa: E402


def load_split_counts() -> dict[str, int]:
    split_file = SHARED_SPLITS / "subject_split.json"
    if not split_file.exists():
        return {}
    data = json.loads(split_file.read_text())
    return {split: len(subjects) for split, subjects in data.items()}


def load_history(history_path: Path) -> list[dict]:
    if not history_path.exists():
        raise FileNotFoundError(f"Training history not found: {history_path}")
    return json.loads(history_path.read_text())


def best_epoch(history: list[dict], metric_key: str = "val_macro_f1") -> int | None:
    scored = [(row["epoch"], row.get(metric_key)) for row in history if row.get(metric_key) is not None]
    if not scored:
        return None
    return max(scored, key=lambda item: item[1])[0]


def plot_training(history_path: Path, *, out_pdf: Path, out_png: Path | None = None) -> None:
    history = load_history(history_path)
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_macro_f1 = [row.get("val_macro_f1") for row in history]
    val_loss = [row.get("val_loss") for row in history]

    split_counts = load_split_counts()
    val_n = split_counts.get("val")
    train_n = split_counts.get("train")
    best = best_epoch(history)

    apply_paper_style()
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.0), sharex=True)

    ax_loss = axes[0]
    ax_loss.plot(epochs, train_loss, color="#2563eb", linewidth=2.0, label="Train loss")
    if any(value is not None for value in val_loss):
        ax_loss.plot(
            epochs,
            [value if value is not None else float("nan") for value in val_loss],
            color="#059669",
            linewidth=2.0,
            linestyle="--",
            label="Validation loss",
        )
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.set_title(f"(a) Loss curves ({split_caption('train', train_n)} / {split_caption('val', val_n)})")
    ax_loss.legend(loc="upper right", frameon=False)

    ax_f1 = axes[1]
    ax_f1.plot(
        epochs,
        [value if value is not None else float("nan") for value in val_macro_f1],
        color="#7c3aed",
        linewidth=2.0,
        marker="o",
        markersize=3.5,
        label="Validation macro F1",
    )
    ax_f1.set_xlabel("Epoch")
    ax_f1.set_ylabel("Macro F1")
    ax_f1.set_ylim(0, 1.02)
    ax_f1.set_title(f"(b) Validation macro F1 ({split_caption('val', val_n)})")
    ax_f1.legend(loc="lower right", frameon=False)

    if best is not None:
        best_f1 = next(row["val_macro_f1"] for row in history if row["epoch"] == best)
        ax_f1.axvline(best, color="#94a3b8", linestyle=":", linewidth=1.2)
        ax_f1.scatter([best], [best_f1], color="#dc2626", s=36, zorder=5)
        ax_f1.annotate(
            f"Best epoch {best}\nmacro F1={best_f1:.3f}",
            xy=(best, best_f1),
            xytext=(best + max(1, len(epochs) * 0.04), best_f1 - 0.08),
            arrowprops={"arrowstyle": "->", "color": "#64748b"},
            fontsize=8,
        )

    fig.suptitle("TinyTCN FP32 training convergence", y=1.02, fontsize=12)
    save_figure(fig, out_pdf, out_png)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--history", type=Path, default=TINYTCN_RUNS / "fp32_100hz" / "history.json")
    p.add_argument("--out-pdf", type=Path, default=PAPER_FIGURES / "training_convergence.pdf")
    p.add_argument("--out-png", type=Path, default=PAPER_FIGURES / "training_convergence.png")
    args = p.parse_args()

    plot_training(args.history, out_pdf=args.out_pdf, out_png=args.out_png)
    print(f"Saved {args.out_pdf}")


if __name__ == "__main__":
    main()
