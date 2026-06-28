#!/usr/bin/env python3
"""Evaluate a TinyTCN checkpoint with structured, paper-ready metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
TINYTCN_DIR = RESEARCH_ROOT / "experiments" / "tinytcn"
sys.path.insert(0, str(RESEARCH_ROOT / "shared"))
sys.path.insert(0, str(TINYTCN_DIR))

from data.dataset import NormStats, probe_label_column  # noqa: E402
from data.splits import files_for_split, load_split  # noqa: E402
from evaluate import evaluate_single  # noqa: E402
from gait_labels import IMU_CHANNEL_SETS  # noqa: E402
from model import build_model  # noqa: E402
from paths import DATA_XY, SHARED_SPLITS, TINYTCN_RUNS  # noqa: E402
from training import make_loader, pick_device  # noqa: E402


def load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Run bash experiments/tinytcn/scripts/train.sh first or pass --checkpoint."
        )
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def model_from_checkpoint(ckpt: dict[str, Any]) -> nn.Module:
    cfg = ckpt["config"]
    model = build_model(
        n_channels=int(cfg.get("n_channels", len(cfg["feature_columns"]))),
        n_classes=int(cfg.get("n_classes", 4)),
        hidden=int(cfg.get("hidden", 32)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def norm_from_checkpoint(ckpt: dict[str, Any], checkpoint_path: Path | None = None) -> NormStats:
    if "norm_stats" in ckpt:
        return NormStats(**ckpt["norm_stats"])
    if checkpoint_path is not None:
        norm_path = checkpoint_path.with_name("norm_stats.json")
        if norm_path.exists():
            return NormStats.load(norm_path)
    raise KeyError("Checkpoint does not contain norm_stats and no sibling norm_stats.json was found.")


def make_eval_loader_from_config(
    cfg: dict[str, Any],
    norm: NormStats,
    *,
    data_root: Path = DATA_XY,
    split_file: Path = SHARED_SPLITS / "subject_split.csv",
    split: str = "test",
    batch_size: int = 512,
    stride: int = 1,
    eager: bool = True,
    cache_trials: int = 3,
):
    split_map = load_split(split_file)
    files = files_for_split(data_root, split_map, split)
    if not files:
        raise ValueError(f"No trial files found for split {split!r} under {data_root}")

    channels = cfg.get("channels", "bilateral")
    feature_columns = cfg.get("feature_columns") or IMU_CHANNEL_SETS[channels]
    label_column = cfg.get("label_column") or probe_label_column(files, channels)

    loader, labels = make_loader(
        files,
        feature_columns=feature_columns,
        label_column=label_column,
        window_size=int(cfg["window"]),
        stride=stride,
        norm=norm,
        decimate=int(cfg.get("decimate", 1)),
        batch_size=batch_size,
        shuffle=False,
        eager=eager,
        cache_trials=cache_trials,
    )
    return loader, labels


def evaluate_model_from_checkpoint_config(
    model: nn.Module,
    ckpt: dict[str, Any],
    *,
    checkpoint_path: Path | None = None,
    data_root: Path = DATA_XY,
    split_file: Path = SHARED_SPLITS / "subject_split.csv",
    split: str = "test",
    batch_size: int = 512,
    stride: int = 1,
    device_name: str = "auto",
    eager: bool = True,
    cache_trials: int = 3,
    desc: str | None = None,
) -> dict[str, Any]:
    device = pick_device(device_name)
    norm = norm_from_checkpoint(ckpt, checkpoint_path)
    loader, labels = make_eval_loader_from_config(
        ckpt["config"],
        norm,
        data_root=data_root,
        split_file=split_file,
        split=split,
        batch_size=batch_size,
        stride=stride,
        eager=eager,
        cache_trials=cache_trials,
    )
    model = model.to(device)
    metrics = evaluate_single(model, loader, device, desc=desc or split)
    metrics["split"] = split
    metrics["windows"] = int(len(labels))
    metrics["checkpoint"] = str(checkpoint_path) if checkpoint_path else None
    return metrics


def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    data_root: Path = DATA_XY,
    split_file: Path = SHARED_SPLITS / "subject_split.csv",
    split: str = "test",
    batch_size: int = 512,
    stride: int = 1,
    device_name: str = "auto",
    eager: bool = True,
    cache_trials: int = 3,
) -> dict[str, Any]:
    ckpt = load_checkpoint(checkpoint_path)
    model = model_from_checkpoint(ckpt)
    return evaluate_model_from_checkpoint_config(
        model,
        ckpt,
        checkpoint_path=checkpoint_path,
        data_root=data_root,
        split_file=split_file,
        split=split,
        batch_size=batch_size,
        stride=stride,
        device_name=device_name,
        eager=eager,
        cache_trials=cache_trials,
    )


def write_metrics(metrics: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=TINYTCN_RUNS / "fp32_100hz" / "best_model.pt")
    p.add_argument("--data-root", type=Path, default=DATA_XY)
    p.add_argument("--split-file", type=Path, default=SHARED_SPLITS / "subject_split.csv")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--lazy", action="store_true")
    p.add_argument("--cache-trials", type=int, default=3)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    metrics = evaluate_checkpoint(
        args.checkpoint,
        data_root=args.data_root,
        split_file=args.split_file,
        split=args.split,
        batch_size=args.batch_size,
        stride=args.stride,
        device_name=args.device,
        eager=not args.lazy,
        cache_trials=args.cache_trials,
    )
    print(json.dumps({k: v for k, v in metrics.items() if k != "report"}, indent=2))
    if args.out:
        write_metrics(metrics, args.out)


if __name__ == "__main__":
    main()
