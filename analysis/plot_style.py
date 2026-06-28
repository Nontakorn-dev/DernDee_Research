"""Shared matplotlib styling for paper figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

PHASES = ("LR", "LS", "PSw", "Sw")
CONFIG_ORDER = ("FP32", "INT8", "INT4", "Prune50", "INT8+Prune50")

CONFIG_COLORS = {
    "FP32": "#2563eb",
    "INT8": "#059669",
    "INT4": "#d97706",
    "Prune50": "#7c3aed",
    "INT8+Prune50": "#dc2626",
}

SPLIT_LABELS = {
    "train": "Train",
    "val": "Validation",
    "test": "Test",
}


def apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
        }
    )


def save_figure(fig: plt.Figure, pdf_path: Path, png_path: Path | None = None) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    if png_path is not None:
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def split_caption(split: str, n_subjects: int | None = None) -> str:
    label = SPLIT_LABELS.get(split, split.title())
    if n_subjects is None:
        return label
    return f"{label} ({n_subjects} subjects)"


def ordered_configs(names: list[str]) -> list[str]:
    order = {name: idx for idx, name in enumerate(CONFIG_ORDER)}
    return sorted(names, key=lambda name: order.get(name, len(CONFIG_ORDER)))
