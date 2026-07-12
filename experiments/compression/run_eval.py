#!/usr/bin/env python3
"""Evaluate TCN-family compression configs with PyTorch-first accuracy proxies.

Supports Standard TCN baseline, INT8 quantization proxy,
structured channel pruning at configurable keep ratios, and optional prune fine-tuning.
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
COMPRESSION_DIR = RESEARCH_ROOT / "experiments" / "compression"
sys.path.insert(0, str(SHARED))
sys.path.insert(0, str(COMPRESSION_DIR))
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
from paths import COMPRESSION_RESULTS, DATA_XY, EXPERIMENTS, SHARED_SPLITS  # noqa: E402
from prune_utils import prune_hidden_channels  # noqa: E402
from training import make_loader, pick_device  # noqa: E402

DEFAULT_CONFIGS = ("FP32", "INT8", "Prune50", "INT8+Prune50")
SUPPORTED_MODELS = ("tcn",)


def default_checkpoint(model_name: str) -> Path:
    return EXPERIMENTS / model_name / "runs" / "fp32_100hz" / "best_model.pt"

def prune_config_name(keep_ratio: float) -> str:
    return f"Prune{int(round(keep_ratio * 100))}"


def quant_config_name(bits: int) -> str:
    return f"INT{bits}"


def combined_config_name(bits: int, keep_ratio: float) -> str:
    return f"INT{bits}+{prune_config_name(keep_ratio)}"


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
    model_name: str,
    hidden: int | None = None,
) -> float:
    cfg = dict(ckpt["config"])
    cfg["model_name"] = model_name
    if hidden is not None:
        cfg["hidden"] = hidden
        model_kwargs = dict(cfg.get("model_kwargs", {}))
        model_kwargs["hidden"] = hidden
        cfg["model_kwargs"] = model_kwargs
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


def parse_config_name(name: str, *, default_keep_ratio: float) -> tuple[str, int | None, float | None]:
    """Return (kind, quant_bits, keep_ratio) for a config label."""
    from kfold_configs import normalize_config_label

    base_name, _ = normalize_config_label(name)
    if base_name == "FP32":
        return "fp32", None, None
    if base_name.startswith("INT") and "+" not in base_name:
        bits = int(base_name.replace("INT", ""))
        return "quant", bits, None
    if base_name.startswith("INT") and "+" in base_name:
        quant_part, prune_part = base_name.split("+", 1)
        bits = int(quant_part.replace("INT", ""))
        keep_ratio = int(prune_part.replace("Prune", "")) / 100.0
        return "quant_prune", bits, keep_ratio
    if base_name.startswith("Prune"):
        keep_ratio = int(base_name.replace("Prune", "")) / 100.0
        return "prune", None, keep_ratio
    raise ValueError(f"Unknown config {name!r}")


def build_config_model(
    name: str,
    fp32_model: nn.Module,
    ckpt: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[nn.Module, dict[str, Any], int, str]:
    from kfold_configs import normalize_config_label

    kind, quant_bits, parsed_keep = parse_config_name(name, default_keep_ratio=args.keep_ratio)
    keep_ratio = args.keep_ratio if parsed_keep is None else parsed_keep
    _, ft_override = normalize_config_label(name)
    finetune_epochs = (
        ft_override if ft_override is not None else args.prune_finetune_epochs
    )

    if kind == "fp32":
        return copy.deepcopy(fp32_model), copy.deepcopy(ckpt), 32, "FP32 checkpoint evaluation"
    if kind == "quant":
        assert quant_bits is not None
        return (
            quantize_dequantize_weights(fp32_model, bits=quant_bits),
            copy.deepcopy(ckpt),
            quant_bits,
            f"PyTorch weight quantize-dequantize INT{quant_bits} accuracy proxy",
        )
    if kind == "prune":
        model = prune_hidden_channels(fp32_model, keep_ratio=keep_ratio)
        pruned_ckpt = copy.deepcopy(ckpt)
        pruned_ckpt["config"] = dict(pruned_ckpt["config"], hidden=model.head[-1].in_features)
        finetune_model(
            model,
            pruned_ckpt,
            data_root=args.data_root,
            split_file=args.split_file,
            epochs=finetune_epochs,
            batch_size=args.batch_size,
            lr=args.finetune_lr,
            device_name=args.device,
            eager=not args.lazy,
            cache_trials=args.cache_trials,
        )
        pct = int(round(keep_ratio * 100))
        return model, pruned_ckpt, 32, f"Structured {pct}% hidden-channel prune + {finetune_epochs}-epoch fine-tune"
    if kind == "quant_prune":
        assert quant_bits is not None
        model = prune_hidden_channels(fp32_model, keep_ratio=keep_ratio)
        pruned_ckpt = copy.deepcopy(ckpt)
        pruned_ckpt["config"] = dict(pruned_ckpt["config"], hidden=model.head[-1].in_features)
        finetune_model(
            model,
            pruned_ckpt,
            data_root=args.data_root,
            split_file=args.split_file,
            epochs=finetune_epochs,
            batch_size=args.batch_size,
            lr=args.finetune_lr,
            device_name=args.device,
            eager=not args.lazy,
            cache_trials=args.cache_trials,
        )
        pct = int(round(keep_ratio * 100))
        return (
            quantize_dequantize_weights(model, bits=quant_bits),
            pruned_ckpt,
            quant_bits,
            f"Structured {pct}% prune + {finetune_epochs}-epoch fine-tune, then INT{quant_bits} QDQ",
        )
    raise ValueError(f"Unhandled config kind {kind!r}")


def compact_metrics(
    *,
    name: str,
    metrics: dict[str, Any],
    params: int,
    size_kb: float,
    file_size_kb: float,
    artifact: Path,
    method_note: str,
    keep_ratio: float | None = None,
    finetune_epochs: int | None = None,
    quant_bits: int | None = None,
) -> dict[str, Any]:
    out = {
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
    if keep_ratio is not None:
        out["keep_ratio"] = keep_ratio
    if finetune_epochs is not None:
        out["finetune_epochs"] = finetune_epochs
    if quant_bits is not None:
        out["quant_bits"] = quant_bits
    return out


def load_model_checkpoint(checkpoint: Path, model_name: str) -> tuple[dict[str, Any], nn.Module]:
    ckpt = load_checkpoint(checkpoint)
    cfg = dict(ckpt["config"])
    cfg["model_name"] = model_name
    ckpt = {**ckpt, "config": cfg}
    return ckpt, model_from_checkpoint(ckpt, checkpoint)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=SUPPORTED_MODELS, default="tcn")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--data-root", type=Path, default=DATA_XY)
    p.add_argument("--split-file", type=Path, default=SHARED_SPLITS / "subject_split.csv")
    p.add_argument("--out-dir", type=Path, default=COMPRESSION_RESULTS / "runs")
    p.add_argument("--results", type=Path, default=None)
    p.add_argument("--overrides", type=Path, default=None)
    p.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS))
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--lazy", action="store_true")
    p.add_argument("--cache-trials", type=int, default=3)
    p.add_argument("--prune-finetune-epochs", type=int, default=5)
    p.add_argument("--finetune-lr", type=float, default=1e-4)
    p.add_argument("--keep-ratio", type=float, default=0.5, help="Default keep ratio when config name omits it")
    args = p.parse_args()

    checkpoint = args.checkpoint or default_checkpoint(args.model)
    if args.results is None:
        args.results = COMPRESSION_RESULTS / "results" / args.model / "metrics.json"
    if args.overrides is None:
        args.overrides = COMPRESSION_RESULTS / "results" / args.model / "overrides.json"

    ckpt, fp32_model = load_model_checkpoint(checkpoint, args.model)
    all_results: dict[str, Any] = {}
    t0 = time.time()

    for name in args.configs:
        print(f"\n=== {args.model} / {name} ===")
        model, eval_ckpt, bits, method_note = build_config_model(name, fp32_model, ckpt, args)
        artifact = args.out_dir / name / "model.pt"
        hidden = int(eval_ckpt["config"].get("hidden", 32))
        file_size_kb = save_artifact(
            model,
            eval_ckpt,
            artifact,
            config_name=name,
            method_note=method_note,
            model_name=args.model,
            hidden=hidden,
        )
        params = count_parameters(model)
        size_kb = compressed_size_kb(params, bits=bits)
        metrics = evaluate_model_from_checkpoint_config(
            model,
            eval_ckpt,
            checkpoint_path=checkpoint,
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
        from kfold_configs import finetune_epochs_for_config, normalize_config_label

        _, _, parsed_keep = parse_config_name(name, default_keep_ratio=args.keep_ratio)
        keep_ratio = parsed_keep
        finetune_epochs = finetune_epochs_for_config(name)
        base_name, _ = normalize_config_label(name)
        if base_name == "FP32":
            quant_bits = 32
        elif base_name.startswith("INT") or "+" in base_name:
            quant_bits = bits
        else:
            quant_bits = None
        result = compact_metrics(
            name=name,
            metrics=metrics,
            params=params,
            size_kb=size_kb,
            file_size_kb=file_size_kb,
            artifact=artifact,
            method_note=method_note,
            keep_ratio=keep_ratio,
            finetune_epochs=finetune_epochs,
            quant_bits=quant_bits,
        )
        all_results[name] = result
        run_dir = args.out_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics.json").write_text(json.dumps(result, indent=2))
        (run_dir / "test_report.txt").write_text(result["report"])
        print(f"{name}: macro_f1={result['macro_f1']:.4f}, size={size_kb:.2f} KB")

    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps({"generated_at_unix": time.time(), "model": args.model, "results": all_results}, indent=2))
    args.overrides.write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote {args.results}")
    print(f"Wrote {args.overrides}")
    print(f"Done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
