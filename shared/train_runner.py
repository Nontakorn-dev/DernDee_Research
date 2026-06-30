"""Shared fair-comparison training runner for all gait phase models."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from tqdm.auto import tqdm

from config_loader import DEFAULT_CONFIG, load_train_config, model_kwargs_for, resolve_split_file
from data.dataset import fit_norm_stats, probe_label_column
from data.splits import files_for_split, load_split
from evaluate import evaluate_single
from gait_labels import IMU_CHANNEL_SETS
from model_registry import build_model_from_config
from paths import DATA_XY, RESEARCH_ROOT, experiment_runs
from training import make_loader, pick_device


STANDARD_ARTIFACTS = (
    "best_model.pt",
    "norm_stats.json",
    "history.json",
    "val_report.txt",
    "test_report.txt",
    "val_confusion_matrix.json",
    "test_confusion_matrix.json",
    "val_metrics.json",
    "test_metrics.json",
    "config.json",
)


@dataclass
class TrainRunConfig:
    model_name: str
    data_root: Path
    output_dir: Path
    config: dict[str, Any]
    device: str
    lazy: bool
    cache_trials: int


def parse_train_args(model_name: str, description: str | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description or f"Train {model_name} with shared protocol.")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--data-root", type=Path, default=DATA_XY)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=experiment_runs(model_name) / "fp32_100hz",
    )
    p.add_argument("--device", type=str, default=None, help="auto, cuda, cpu, or mps")
    p.add_argument("--lazy", action="store_true", help="Lazy trial loading for low RAM (Colab).")
    p.add_argument("--cache-trials", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None, help="Override config epochs.")
    p.add_argument("--batch-size", type=int, default=None, help="Override config batch size.")
    return p.parse_args()


def build_run_config(model_name: str, args: argparse.Namespace) -> TrainRunConfig:
    cfg = load_train_config(args.config)
    runtime = cfg.get("runtime", {})
    train_cfg = cfg["train"]

    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size

    device = args.device or runtime.get("device", "auto")
    lazy = args.lazy or bool(runtime.get("lazy", False))
    cache_trials = args.cache_trials if args.cache_trials is not None else int(runtime.get("cache_trials", 3))

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = RESEARCH_ROOT / output_dir

    data_root = args.data_root
    if not data_root.is_absolute():
        data_root = RESEARCH_ROOT / data_root

    return TrainRunConfig(
        model_name=model_name,
        data_root=data_root,
        output_dir=output_dir,
        config=cfg,
        device=device,
        lazy=lazy,
        cache_trials=cache_trials,
    )


def _write_metrics_bundle(output_dir: Path, split: str, metrics: dict[str, Any]) -> None:
    (output_dir / f"{split}_report.txt").write_text(metrics["report"])
    (output_dir / f"{split}_confusion_matrix.json").write_text(
        json.dumps(metrics["confusion_matrix"], indent=2)
    )
    payload = {
        "split": split,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "loss": metrics.get("loss"),
        "phase_f1": metrics["phase_f1"],
        "phase_accuracy": metrics["phase_accuracy"],
        "phase_support": metrics["phase_support"],
        "windows": metrics.get("windows"),
    }
    (output_dir / f"{split}_metrics.json").write_text(json.dumps(payload, indent=2))


def _checkpoint_config(
    run: TrainRunConfig,
    *,
    feature_columns: list[str],
    label_column: str,
    n_channels: int,
    decimate: int,
    window: int,
    model_kwargs: dict[str, Any],
) -> dict[str, Any]:
    data_cfg = run.config["data"]
    return {
        "model_name": run.model_name,
        "model_kwargs": model_kwargs,
        "window": window,
        "channels": data_cfg["channels"],
        "feature_columns": feature_columns,
        "label_column": label_column,
        "n_channels": n_channels,
        "n_classes": int(model_kwargs.get("n_classes", 4)),
        "hidden": int(model_kwargs.get("hidden", 32)),
        "source_hz": int(data_cfg["source_hz"]),
        "target_hz": int(data_cfg["target_hz"]),
        "decimate": decimate,
        "train_stride": int(data_cfg["train_stride"]),
        "val_stride": int(data_cfg["val_stride"]),
        "test_stride": int(data_cfg["test_stride"]),
    }


def run_training(run: TrainRunConfig) -> dict[str, Any]:
    cfg = run.config
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    torch.manual_seed(int(train_cfg["seed"]))
    np.random.seed(int(train_cfg["seed"]))

    channels = data_cfg["channels"]
    feature_columns = IMU_CHANNEL_SETS[channels]
    n_channels = len(feature_columns)

    source_hz = int(data_cfg["source_hz"])
    target_hz = int(data_cfg["target_hz"])
    if source_hz % target_hz != 0:
        raise ValueError(f"source_hz {source_hz} must divide target_hz {target_hz}")
    decimate = source_hz // target_hz

    window = int(data_cfg["window"])
    train_stride = int(data_cfg["train_stride"])
    val_stride = int(data_cfg["val_stride"])
    test_stride = int(data_cfg["test_stride"])

    run.output_dir.mkdir(parents=True, exist_ok=True)
    (run.output_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    split_file = resolve_split_file(cfg)
    split_map = load_split(split_file)
    train_files = files_for_split(run.data_root, split_map, "train")
    val_files = files_for_split(run.data_root, split_map, "val")
    test_files = files_for_split(run.data_root, split_map, "test")
    label_column = probe_label_column(train_files, channels)

    device = pick_device(run.device)
    eager = not run.lazy
    model_kwargs = model_kwargs_for(cfg, run.model_name)

    print("=" * 60)
    print(f"{run.model_name} training (shared fair-comparison protocol)")
    print("=" * 60)
    print(
        f"Train: {len(split_map['train'])} subj | "
        f"Val: {len(split_map['val'])} | Test: {len(split_map['test'])}"
    )
    print(f"Device: {device} | Lazy: {run.lazy} | Out: {run.output_dir}\n")

    norm = fit_norm_stats(train_files, feature_columns=feature_columns, decimate=decimate)
    norm.save(run.output_dir / "norm_stats.json")

    batch_size = int(train_cfg["batch_size"])
    train_loader, train_labels = make_loader(
        train_files,
        feature_columns=feature_columns,
        label_column=label_column,
        window_size=window,
        stride=train_stride,
        norm=norm,
        decimate=decimate,
        batch_size=batch_size,
        shuffle=True,
        eager=eager,
        cache_trials=run.cache_trials,
    )
    val_loader, val_labels = make_loader(
        val_files,
        feature_columns=feature_columns,
        label_column=label_column,
        window_size=window,
        stride=val_stride,
        norm=norm,
        decimate=decimate,
        batch_size=batch_size,
        shuffle=False,
        eager=eager,
        cache_trials=run.cache_trials,
    )

    if train_cfg["class_weights"] != "balanced":
        raise ValueError(f"Unsupported class_weights: {train_cfg['class_weights']!r}")
    weights = compute_class_weight("balanced", classes=np.arange(4), y=train_labels)

    model = build_model_from_config(
        run.model_name,
        n_channels=n_channels,
        model_kwargs=model_kwargs,
    ).to(device)

    if train_cfg["optimizer"] != "adam":
        raise ValueError(f"Unsupported optimizer: {train_cfg['optimizer']!r}")
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg["lr"]))
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )

    ckpt_config = _checkpoint_config(
        run,
        feature_columns=feature_columns,
        label_column=label_column,
        n_channels=n_channels,
        decimate=decimate,
        window=window,
        model_kwargs=model_kwargs,
    )

    best_f1 = -1.0
    best_epoch = 0
    history: list[dict[str, Any]] = []
    epochs = int(train_cfg["epochs"])
    grad_clip = float(train_cfg["grad_clip"])

    for epoch in tqdm(range(1, epochs + 1), desc="epochs"):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        n_samples = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            running_loss += loss.item() * len(yb)
            n_samples += len(yb)

        train_loss = running_loss / n_samples
        val_metrics = evaluate_single(model, val_loader, device, desc="val")
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_macro_f1": val_metrics["macro_f1"],
                "val_accuracy": val_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
            }
        )
        tqdm.write(
            f"Epoch {epoch:02d}  val_f1={val_metrics['macro_f1']:.4f}  "
            f"({time.time() - t0:.0f}s)"
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": ckpt_config,
                    "norm_stats": {"mean": norm.mean, "std": norm.std},
                    "split": split_map,
                    "epoch": epoch,
                    "val_macro_f1": best_f1,
                },
                run.output_dir / "best_model.pt",
            )

    (run.output_dir / "history.json").write_text(json.dumps(history, indent=2))

    best_ckpt = torch.load(run.output_dir / "best_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    val_metrics = evaluate_single(model, val_loader, device, desc="val-final")
    val_metrics["windows"] = int(len(val_labels))
    _write_metrics_bundle(run.output_dir, "val", val_metrics)
    print(f"Best epoch: {best_epoch}  val macro-F1: {best_f1:.4f}")

    test_summary: dict[str, Any] | None = None
    if test_files:
        test_loader, test_labels = make_loader(
            test_files,
            feature_columns=feature_columns,
            label_column=label_column,
            window_size=window,
            stride=test_stride,
            norm=norm,
            decimate=decimate,
            batch_size=batch_size,
            shuffle=False,
            eager=eager,
            cache_trials=run.cache_trials,
        )
        test_metrics = evaluate_single(model, test_loader, device, desc="test")
        test_metrics["windows"] = int(len(test_labels))
        _write_metrics_bundle(run.output_dir, "test", test_metrics)
        print(
            f"Test macro-F1: {test_metrics['macro_f1']:.4f}  "
            f"({len(test_labels):,} windows)"
        )
        test_summary = {
            "macro_f1": test_metrics["macro_f1"],
            "accuracy": test_metrics["accuracy"],
            "windows": test_metrics["windows"],
        }

    return {
        "model_name": run.model_name,
        "output_dir": str(run.output_dir),
        "best_epoch": best_epoch,
        "val_macro_f1": best_f1,
        "test": test_summary,
    }


def main_for_model(model_name: str, description: str | None = None) -> None:
    args = parse_train_args(model_name, description=description)
    run = build_run_config(model_name, args)
    run_training(run)


if __name__ == "__main__":
    raise SystemExit("Use experiments/<model>/train.py or import run_training() from train_runner.")
