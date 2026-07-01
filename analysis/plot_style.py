"""Shared matplotlib styling for paper figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

PHASES = ("LR", "LS", "PSw", "Sw")
CONFIG_ORDER = ("FP32", "INT8", "Prune50", "INT8+Prune50")

# Okabe-Ito colorblind-safe qualitative palette; also chosen to remain
# distinguishable when the paper is printed in grayscale.
CONFIG_COLORS = {
    "FP32": "#0072B2",
    "INT8": "#009E73",
    "Prune50": "#CC79A7",
    "INT8+Prune50": "#D55E00",
}

# Hatch patterns give a second (non-color) encoding for grayscale printing.
CONFIG_HATCHES = {
    "FP32": "",
    "INT8": "//",
    "Prune50": "\\\\",
    "INT8+Prune50": "xx",
}

SPLIT_LABELS = {
    "train": "Train",
    "val": "Validation",
    "test": "Test",
}


def apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#c7ccd1",
            "grid.alpha": 0.6,
            "grid.linestyle": ":",
            "grid.linewidth": 0.6,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
        }
    )


def save_figure(fig: plt.Figure, pdf_path: Path, png_path: Path | None = None) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    if png_path is not None:
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def split_caption(split: str, n_subjects: int | None = None) -> str:
    label = SPLIT_LABELS.get(split, split.title())
    if n_subjects is None:
        return label
    return f"{label} ({n_subjects} subjects)"


def ordered_configs(names: list[str]) -> list[str]:
    order = {name: idx for idx, name in enumerate(CONFIG_ORDER)}
    return sorted(names, key=lambda name: order.get(name, len(CONFIG_ORDER)))
