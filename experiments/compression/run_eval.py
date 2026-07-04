#!/usr/bin/env python3
"""Evaluate TinyTCN compression configs with PyTorch-first accuracy proxies.

This runner measures the research question first: how much each gait phase
degrades after compression. ESP32/TFLite artifacts are a later measurement
phase and should not be inferred from these PyTorch metrics.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from tqdm.auto import tqdm

RESEARCH_ROOT = Path(__file__).resolve().parents[2]
SHARED = RESEARCH_ROOT / "shared"
TINYTCN_DIR = RESEARCH_ROOT / "experiments" / "tinytcn"
sys.path.insert(0, str(SHARED))
sys.path.insert(0, str(TINYTCN_DIR))
sys.path.insert(0, str(RESEARCH_ROOT))

from data.dataset import probe_label_column  # noqa: E402
from data.splits import SplitPolicy, assert_files_match_split, files_for_split, load_split  # noqa: E402
from eval_checkpoint import (  # noqa: E402
    evaluate_model_from_checkpoint_config,
    load_checkpoint,
    model_from_checkpoint,
    norm_from_checkpoint,
)
from gait_labels import IMU_CHANNEL_SETS  # noqa: E402
from model import TinyTCN  # noqa: E402
from paths import COMPRESSION_RESULTS, DATA_XY, SHARED_SPLITS, TINYTCN_RUNS  # noqa: E402
from training import make_loader, pick_device  # noqa: E402


CONFIGS = ("FP32", "INT8", "Prune50", "INT8+Prune50")


def prune_config_name(keep_ratio: float) -> str:
    return f"Prune{int(round(keep_ratio * 100))}"


def combined_config_name(keep_ratio: float) -> str:
    return f"INT8+{prune_config_name(keep_ratio)}"


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def compressed_size_kb(params: int, *, bits: int, overhead_kb: float = 2.0) -> float:
    overhead = 0.0 if bits == 32 else overhead_kb
    return round(params * bits / 8 / 1024 + overhead, 2)


def quantize_dequantize_tensor(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    if bits >= 32:
        return tensor.detach().clone()
    qmax = (2 ** (bits - 1)) - 1
    max_abs = tensor.detach().abs().max()
    if float(max_abs) == 0.0:
        return tensor.detach().clone()
    scale = max_abs / qmax
    quantized = torch.clamp(torch.round(tensor / scale), -qmax, qmax)
    return (quantized * scale).to(dtype=tensor.dtype)


def quantize_dequantize_weights(model: nn.Module, *, bits: int) -> nn.Module:
    q_model = copy.deepcopy(model).cpu()
    with torch.no_grad():
        for module in q_model.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                module.weight.copy_(quantize_dequantize_tensor(module.weight, bits))
    return q_model


def select_hidden_channels(model: TinyTCN, keep_count: int) -> torch.Tensor:
    with torch.no_grad():
        importance = torch.zeros(model.head[2].in_features)
        for block in model.stem:
            conv = block.conv
            importance += conv.weight.detach().abs().sum(dim=(1, 2)).cpu()
        keep = torch.topk(importance, k=keep_count, largest=True).indices
        return torch.sort(keep).values


def _copy_bn(dst: dict[str, torch.Tensor], src: dict[str, torch.Tensor], prefix: str, keep: torch.Tensor) -> None:
    for name in ("weight", "bias", "running_mean", "running_var"):
        dst[f"{prefix}.{name}"] = src[f"{prefix}.{name}"][keep].clone()
    dst[f"{prefix}.num_batches_tracked"] = src[f"{prefix}.num_batches_tracked"].clone()


def prune_tinytcn_hidden(model: TinyTCN, *, keep_ratio: float = 0.5) -> TinyTCN:
    old_hidden = model.head[2].in_features
    new_hidden = max(1, int(round(old_hidden * keep_ratio)))
    keep = select_hidden_channels(model, new_hidden)

    src = model.state_dict()
    pruned = TinyTCN(n_channels=model.stem[0].conv.in_channels, n_classes=model.head[2].out_features, hidden=new_hidden)
    dst = pruned.state_dict()

    dst["stem.0.conv.weight"] = src["stem.0.conv.weight"][keep].clone()
    dst["stem.0.conv.bias"] = src["stem.0.conv.bias"][keep].clone()
    _copy_bn(dst, src, "stem.0.bn", keep)
    dst["stem.0.residual.weight"] = src["stem.0.residual.weight"][keep].clone()
    dst["stem.0.residual.bias"] = src["stem.0.residual.bias"][keep].clone()

    for idx in (1, 2):
        dst[f"stem.{idx}.conv.weight"] = src[f"stem.{idx}.conv.weight"][keep][:, keep].clone()
        dst[f"stem.{idx}.conv.bias"] = src[f"stem.{idx}.conv.bias"][keep].clone()
        _copy_bn(dst, src, f"stem.{idx}.bn", keep)

    dst["head.2.weight"] = src["head.2.weight"][:, keep].clone()
    dst["head.2.bias"] = src["head.2.bias"].clone()
    pruned.load_state_dict(dst)
    return pruned


def finetune_model(
    model: nn.Module,
    ckpt: dict[str, Any],
    *,
    data_root: Path,
    split_file: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    device_name: str,
    eager: bool,
    cache_trials: int,
) -> None:
    if epochs <= 0:
        return

    cfg = ckpt["config"]
    split_map = load_split(split_file)
    train_files = files_for_split(data_root, split_map, "train")
    if not train_files:
        raise ValueError(f"No train files found under {data_root}")

    assert_files_match_split(
        train_files, split_map, policy=SplitPolicy.TRAIN_ONLY, context="prune fine-tune"
    )

    channels = cfg.get("channels", "bilateral")
    feature_columns = cfg.get("feature_columns") or IMU_CHANNEL_SETS[channels]
    label_column = cfg.get("label_column") or probe_label_column(train_files, channels)
    norm = norm_from_checkpoint(ckpt)
    train_loader, train_labels = make_loader(
        train_files,
        feature_columns=feature_columns,
        label_column=label_column,
        window_size=int(cfg["window"]),
        stride=int(cfg.get("stride", max(1, int(0.05 * cfg.get("target_hz", 100))))),
        norm=norm,
        decimate=int(cfg.get("decimate", 1)),
        batch_size=batch_size,
        shuffle=True,
        eager=eager,
        cache_trials=cache_trials,
    )

    device = pick_device(device_name)
    model.to(device)
    model.train()
    weights = compute_class_weight("balanced", classes=np.arange(4), y=train_labels)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(1, epochs + 1), desc="prune fine-tune", unit="epoch"):
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
        tqdm.write(f"fine-tune epoch {epoch:02d}: loss={running_loss / n_samples:.4f}")


def save_artifact(
    model: nn.Module,
    ckpt: dict[str, Any],
    path: Path,
    *,
    config_name: str,
    method_note: str,
    hidden: int | None = None,
) -> float:
    cfg = dict(ckpt["config"])
    if hidden is not None:
        cfg["hidden"] = hidden
    payload = {
        "model_state_dict": model.cpu().state_dict(),
        "config": cfg,
        "norm_stats": ckpt["norm_stats"],
        "split": ckpt.get("split"),
        "compression": {"config": config_name, "method_note": method_note},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return round(path.stat().st_size / 1024, 2)


def build_config_model(
    name: str,
    fp32_model: TinyTCN,
    ckpt: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[nn.Module, dict[str, Any], int, str]:
    keep_ratio = float(getattr(args, "keep_ratio", 0.5))
    if name == "FP32":
        return copy.deepcopy(fp32_model), copy.deepcopy(ckpt), 32, "FP32 checkpoint evaluation"
    if name == "INT8":
        return (
            quantize_dequantize_weights(fp32_model, bits=8),
            copy.deepcopy(ckpt),
            8,
            "PyTorch weight quantize-dequantize INT8 accuracy proxy",
        )
    if name.startswith("Prune") and not name.startswith("INT8+"):
        model = prune_tinytcn_hidden(fp32_model, keep_ratio=keep_ratio)
        pruned_ckpt = copy.deepcopy(ckpt)
        pruned_ckpt["config"] = dict(pruned_ckpt["config"], hidden=model.head[2].in_features)
        finetune_model(
            model,
            pruned_ckpt,
            data_root=args.data_root,
            split_file=args.split_file,
            epochs=args.prune_finetune_epochs,
            batch_size=args.batch_size,
            lr=args.finetune_lr,
            device_name=args.device,
            eager=not args.lazy,
            cache_trials=args.cache_trials,
        )
        pct = int(round(keep_ratio * 100))
        return model, pruned_ckpt, 32, f"Structured {pct}% hidden-channel prune with fine-tuning"
    if name.startswith("INT8+Prune"):
        model = prune_tinytcn_hidden(fp32_model, keep_ratio=keep_ratio)
        pruned_ckpt = copy.deepcopy(ckpt)
        pruned_ckpt["config"] = dict(pruned_ckpt["config"], hidden=model.head[2].in_features)
        finetune_model(
            model,
            pruned_ckpt,
            data_root=args.data_root,
            split_file=args.split_file,
            epochs=args.prune_finetune_epochs,
            batch_size=args.batch_size,
            lr=args.finetune_lr,
            device_name=args.device,
            eager=not args.lazy,
            cache_trials=args.cache_trials,
        )
        pct = int(round(keep_ratio * 100))
        return (
            quantize_dequantize_weights(model, bits=8),
            pruned_ckpt,
            8,
            f"Structured {pct}% prune with fine-tuning, then INT8 weight quantize-dequantize",
        )
    raise ValueError(f"Unknown config {name!r}. Expected one of: {', '.join(CONFIGS)}")


def compact_metrics(
    *,
    name: str,
    metrics: dict[str, Any],
    params: int,
    size_kb: float,
    file_size_kb: float,
    artifact: Path,
    method_note: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "phase_f1": metrics["phase_f1"],
        "phase_accuracy": metrics["phase_accuracy"],
        "phase_support": metrics["phase_support"],
        "confusion_matrix": metrics["confusion_matrix"],
        "size_kb": size_kb,
        "file_size_kb": file_size_kb,
        "params": params,
        "artifact": str(artifact),
        "method_note": method_note,
        "report": metrics["report"],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=TINYTCN_RUNS / "fp32_100hz" / "best_model.pt")
    p.add_argument("--data-root", type=Path, default=DATA_XY)
    p.add_argument("--split-file", type=Path, default=SHARED_SPLITS / "subject_split.csv")
    p.add_argument("--out-dir", type=Path, default=COMPRESSION_RESULTS / "runs")
    p.add_argument("--results", type=Path, default=COMPRESSION_RESULTS / "results" / "metrics.json")
    p.add_argument("--overrides", type=Path, default=COMPRESSION_RESULTS / "overrides.json")
    p.add_argument("--configs", nargs="+", default=list(CONFIGS))
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--lazy", action="store_true")
    p.add_argument("--cache-trials", type=int, default=3)
    p.add_argument("--prune-finetune-epochs", type=int, default=5)
    p.add_argument("--finetune-lr", type=float, default=1e-4)
    p.add_argument("--keep-ratio", type=float, default=0.5, help="Structured prune keep ratio (0.25/0.5/0.75)")
    args = p.parse_args()

    ckpt = load_checkpoint(args.checkpoint)
    fp32_model = model_from_checkpoint(ckpt, args.checkpoint)
    all_results: dict[str, Any] = {}
    t0 = time.time()

    for name in args.configs:
        print(f"\n=== {name} ===")
        model, eval_ckpt, bits, method_note = build_config_model(name, fp32_model, ckpt, args)
        artifact = args.out_dir / name / "model.pt"
        hidden = int(eval_ckpt["config"].get("hidden", 32))
        file_size_kb = save_artifact(
            model,
            eval_ckpt,
            artifact,
            config_name=name,
            method_note=method_note,
            hidden=hidden,
        )
        params = count_parameters(model)
        size_kb = compressed_size_kb(params, bits=bits)
        metrics = evaluate_model_from_checkpoint_config(
            model,
            eval_ckpt,
            checkpoint_path=args.checkpoint,
            data_root=args.data_root,
            split_file=args.split_file,
            split="test",
            batch_size=args.batch_size,
            stride=1,
            device_name=args.device,
            eager=not args.lazy,
            cache_trials=args.cache_trials,
            desc=f"test {name}",
        )
        result = compact_metrics(
            name=name,
            metrics=metrics,
            params=params,
            size_kb=size_kb,
            file_size_kb=file_size_kb,
            artifact=artifact,
            method_note=method_note,
        )
        all_results[name] = result
        run_dir = args.out_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics.json").write_text(json.dumps(result, indent=2))
        (run_dir / "test_report.txt").write_text(result["report"])
        print(f"{name}: macro_f1={result['macro_f1']:.4f}, size={size_kb:.2f} KB")

    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps({"generated_at_unix": time.time(), "results": all_results}, indent=2))
    args.overrides.write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote {args.results}")
    print(f"Wrote {args.overrides}")
    print(f"Done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
