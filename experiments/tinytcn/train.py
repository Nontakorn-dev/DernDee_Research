#!/usr/bin/env python3
"""Train TinyTCN (12ch bilateral, 100 Hz) → runs/fp32_100hz/."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from tqdm.auto import tqdm

EXP_DIR = Path(__file__).resolve().parent
RESEARCH_ROOT = EXP_DIR.parents[1]
SHARED = RESEARCH_ROOT / "shared"
sys.path.insert(0, str(SHARED))
sys.path.insert(0, str(EXP_DIR))

from gait_labels import IMU_CHANNEL_SETS  # noqa: E402
from data.dataset import SOURCE_HZ, fit_norm_stats, probe_label_column  # noqa: E402
from data.splits import files_for_split, load_split  # noqa: E402
from evaluate import evaluate_single  # noqa: E402
from model import build_model  # noqa: E402
from paths import DATA_XY, SHARED_SPLITS, TINYTCN_RUNS  # noqa: E402
from training import make_loader, pick_device, resolve_hz_params  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DATA_XY)
    p.add_argument("--split-file", type=Path, default=SHARED_SPLITS / "subject_split.csv")
    p.add_argument("--channels", type=str, default="bilateral", choices=sorted(IMU_CHANNEL_SETS))
    p.add_argument("--output-dir", type=Path, default=TINYTCN_RUNS / "fp32_100hz")
    p.add_argument("--target-hz", type=int, default=100)
    p.add_argument("--source-hz", type=int, default=SOURCE_HZ)
    p.add_argument("--window", type=int, default=None)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--val-stride", type=int, default=None)
    p.add_argument("--lazy", action="store_true")
    p.add_argument("--cache-trials", type=int, default=3)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    feature_columns = IMU_CHANNEL_SETS[args.channels]
    n_channels = len(feature_columns)
    decimate, window, stride = resolve_hz_params(
        source_hz=args.source_hz,
        target_hz=args.target_hz,
        window=args.window,
        stride=args.stride,
    )
    val_stride = 1 if args.val_stride is None else args.val_stride
    eager = not args.lazy
    device = pick_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_map = load_split(args.split_file)
    train_files = files_for_split(args.data_root, split_map, "train")
    val_files = files_for_split(args.data_root, split_map, "val")
    test_files = files_for_split(args.data_root, split_map, "test")
    label_column = probe_label_column(train_files, args.channels)

    print("=" * 60)
    print("TinyTCN training")
    print("=" * 60)
    print(f"Train: {len(split_map['train'])} subj | Val: {len(split_map['val'])} | Test: {len(split_map['test'])}")
    print(f"Out:   {args.output_dir}\n")

    norm = fit_norm_stats(train_files, feature_columns=feature_columns, decimate=decimate)
    norm.save(args.output_dir / "norm_stats.json")

    train_loader, train_labels = make_loader(
        train_files,
        feature_columns=feature_columns,
        label_column=label_column,
        window_size=window,
        stride=stride,
        norm=norm,
        decimate=decimate,
        batch_size=args.batch_size,
        shuffle=True,
        eager=eager,
        cache_trials=args.cache_trials,
    )
    val_loader, _ = make_loader(
        val_files,
        feature_columns=feature_columns,
        label_column=label_column,
        window_size=window,
        stride=val_stride,
        norm=norm,
        decimate=decimate,
        batch_size=args.batch_size,
        shuffle=False,
        eager=eager,
        cache_trials=args.cache_trials,
    )

    weights = compute_class_weight("balanced", classes=np.arange(4), y=train_labels)
    model = build_model(n_channels=n_channels, hidden=args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )

    best_f1 = -1.0
    history: list[dict] = []
    for epoch in tqdm(range(1, args.epochs + 1), desc="epochs"):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        n_samples = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * len(yb)
            n_samples += len(yb)

        train_loss = running_loss / n_samples
        metrics = evaluate_single(model, val_loader, device)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_macro_f1": metrics["macro_f1"],
        })
        tqdm.write(f"Epoch {epoch:02d}  f1={metrics['macro_f1']:.4f}  ({time.time()-t0:.0f}s)")

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "window": window,
                        "channels": args.channels,
                        "feature_columns": feature_columns,
                        "label_column": label_column,
                        "n_channels": n_channels,
                        "n_classes": 4,
                        "hidden": args.hidden,
                        "source_hz": args.source_hz,
                        "target_hz": args.target_hz,
                        "decimate": decimate,
                    },
                    "norm_stats": {"mean": norm.mean, "std": norm.std},
                    "split": split_map,
                    "epoch": epoch,
                    "val_macro_f1": best_f1,
                },
                args.output_dir / "best_model.pt",
            )

    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
    model.load_state_dict(torch.load(args.output_dir / "best_model.pt", weights_only=False)["model_state_dict"])

    if test_files:
        test_loader, test_labels = make_loader(
            test_files,
            feature_columns=feature_columns,
            label_column=label_column,
            window_size=window,
            stride=1,
            norm=norm,
            decimate=decimate,
            batch_size=args.batch_size,
            shuffle=False,
            eager=eager,
            cache_trials=args.cache_trials,
        )
        test_m = evaluate_single(model, test_loader, device, desc="test")
        (args.output_dir / "test_report.txt").write_text(test_m["report"])
        print(f"Test macro-F1: {test_m['macro_f1']:.4f}  ({len(test_labels):,} windows)")


if __name__ == "__main__":
    main()
