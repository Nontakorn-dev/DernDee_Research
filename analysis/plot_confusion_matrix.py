#!/usr/bin/env python3
"""Plot confusion-matrix heatmaps with clear train/val/test fold separation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH_ROOT))

from analysis.plot_style import (  # noqa: E402
    PHASES,
    apply_paper_style,
    ordered_configs,
    save_figure,
    split_caption,
)
from shared.paths import COMPRESSION_RESULTS, PAPER_FIGURES, SHARED_SPLITS, TINYTCN_RUNS  # noqa: E402


@dataclass(frozen=True)
class ConfusionSource:
    config: str
    split: str
    matrix: np.ndarray
    accuracy: float | None = None


def load_split_counts() -> dict[str, int]:
    split_file = SHARED_SPLITS / "subject_split.json"
    if not split_file.exists():
        return {}
    data = json.loads(split_file.read_text())
    return {split: len(subjects) for split, subjects in data.items()}


def matrix_accuracy(cm: np.ndarray) -> float:
    total = cm.sum()
    if total == 0:
        return 0.0
    return float(np.trace(cm) / total)


def load_fp32_sources(run_dir: Path, splits: tuple[str, ...]) -> list[ConfusionSource]:
    sources: list[ConfusionSource] = []
    for split in splits:
        path = run_dir / f"{split}_confusion_matrix.json"
        if not path.exists():
            continue
        cm = np.array(json.loads(path.read_text()), dtype=np.int64)
        sources.append(ConfusionSource(config="FP32", split=split, matrix=cm, accuracy=matrix_accuracy(cm)))
    return sources


def load_compression_sources(runs_dir: Path, configs: list[str] | None = None) -> list[ConfusionSource]:
    if not runs_dir.exists():
        return []

    names = configs or [p.name for p in runs_dir.iterdir() if p.is_dir()]
    sources: list[ConfusionSource] = []
    for name in ordered_configs(names):
        metrics_path = runs_dir / name / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text())
        cm = metrics.get("confusion_matrix")
        if cm is None:
            continue
        matrix = np.array(cm, dtype=np.int64)
        accuracy = metrics.get("accuracy")
        sources.append(
            ConfusionSource(
                config=name,
                split="test",
                matrix=matrix,
                accuracy=float(accuracy) if accuracy is not None else matrix_accuracy(matrix),
            )
        )
    return sources


def discover_sources(
    fp32_run_dir: Path,
    compression_runs_dir: Path,
    *,
    include_val: bool = True,
    include_test: bool = True,
    configs: list[str] | None = None,
) -> list[ConfusionSource]:
    keyed: dict[tuple[str, str], ConfusionSource] = {}

    splits = tuple(
        split
        for split, enabled in (("val", include_val), ("test", include_test))
        if enabled
    )
    for source in load_fp32_sources(fp32_run_dir, splits):
        keyed[(source.config, source.split)] = source

    if include_test:
        for source in load_compression_sources(compression_runs_dir, configs):
            keyed[(source.config, source.split)] = source

    return [keyed[key] for key in sorted(keyed)]


def plot_confusion_matrix(
    sources: list[ConfusionSource],
    *,
    out_pdf: Path,
    out_png: Path | None = None,
) -> None:
    if not sources:
        raise ValueError("No confusion-matrix sources found.")

    split_counts = load_split_counts()
    configs = ordered_configs(sorted({s.config for s in sources}))
    splits = [split for split in ("val", "test") if any(s.split == split for s in sources)]

    apply_paper_style()
    n_rows = len(configs)
    n_cols = len(splits)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 3.8 * n_rows),
        squeeze=False,
        layout="constrained",
    )

    vmax = max(source.matrix.max() for source in sources)
    im = None
    for row, config in enumerate(configs):
        for col, split in enumerate(splits):
            ax = axes[row, col]
            match = next(
                (source for source in sources if source.config == config and source.split == split),
                None,
            )
            if match is None:
                ax.axis("off")
                ax.text(
                    0.5,
                    0.5,
                    "Not evaluated",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                    color="#64748b",
                )
                ax.set_title(f"{config} | {split_caption(split, split_counts.get(split))}")
                continue

            cm = match.matrix
            im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)
            ax.set_xticks(range(len(PHASES)))
            ax.set_yticks(range(len(PHASES)))
            ax.set_xticklabels(PHASES)
            ax.set_yticklabels(PHASES)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            acc_pct = (match.accuracy or matrix_accuracy(cm)) * 100
            ax.set_title(
                f"{config} | {split_caption(split, split_counts.get(split))}\n"
                f"Accuracy {acc_pct:.2f}%"
            )

            threshold = vmax * 0.55
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    value = int(cm[i, j])
                    color = "white" if value > threshold else "#0f172a"
                    ax.text(j, i, f"{value:,}", ha="center", va="center", fontsize=7, color=color)

    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label="Window count")
    fig.suptitle("Confusion matrices by compression config and evaluation fold", y=1.02, fontsize=12)
    save_figure(fig, out_pdf, out_png)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fp32-run-dir", type=Path, default=TINYTCN_RUNS / "fp32_100hz")
    p.add_argument("--compression-runs-dir", type=Path, default=COMPRESSION_RESULTS / "runs")
    p.add_argument("--configs", nargs="*", default=None)
    p.add_argument("--skip-val", action="store_true")
    p.add_argument("--skip-test", action="store_true")
    p.add_argument("--out-pdf", type=Path, default=PAPER_FIGURES / "confusion_matrix.pdf")
    p.add_argument("--out-png", type=Path, default=PAPER_FIGURES / "confusion_matrix.png")
    args = p.parse_args()

    sources = discover_sources(
        args.fp32_run_dir,
        args.compression_runs_dir,
        include_val=not args.skip_val,
        include_test=not args.skip_test,
        configs=args.configs,
    )
    if not sources:
        raise SystemExit(
            "No confusion matrices found. Train TinyTCN first or run compression evaluation."
        )
    plot_confusion_matrix(sources, out_pdf=args.out_pdf, out_png=args.out_png)
    print(f"Saved {args.out_pdf}")


if __name__ == "__main__":
    main()
