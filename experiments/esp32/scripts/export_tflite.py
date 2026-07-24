#!/usr/bin/env python3
"""Export TCN compression configs to TFLite for ESP32-C3 deployment.

FP32 / Prune50: float32 TFLite from compression checkpoints.
INT8 / INT8+Prune50: full-integer PTQ from FP32 / Prune50 float weights
(not the PyTorch quantize-dequantize proxy checkpoints).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

RESEARCH_ROOT = Path(__file__).resolve().parents[3]
SHARED = RESEARCH_ROOT / "shared"
ESP32_DIR = RESEARCH_ROOT / "experiments" / "esp32"
COMPRESSION_RUNS = RESEARCH_ROOT / "experiments" / "compression" / "runs"
DEFAULT_MODEL = "tcn"

sys.path.insert(0, str(SHARED))
sys.path.insert(0, str(RESEARCH_ROOT))
sys.path.insert(0, str(RESEARCH_ROOT / "experiments" / "esp32"))

from data.dataset import NormStats, load_trial  # noqa: E402
from data.splits import files_for_split, load_split  # noqa: E402
from eval_checkpoint import load_checkpoint, model_from_checkpoint, norm_from_checkpoint  # noqa: E402
from gait_labels import IMU_INPUT_COLUMNS  # noqa: E402
from paths import DATA_XY, SHARED_SPLITS  # noqa: E402

EXPORT_CONFIGS = ("FP32", "INT8", "Prune25", "Prune50", "Prune75", "INT8+Prune25", "INT8+Prune50", "INT8+Prune75")

# Which checkpoint supplies float weights, and whether to apply TFLite INT8 PTQ.
EXPORT_PLAN: dict[str, tuple[str, str]] = {
    "FP32": ("FP32", "float"),
    "INT8": ("FP32", "int8"),
    "Prune25": ("Prune25", "float"),
    "Prune50": ("Prune50", "float"),
    "Prune75": ("Prune75", "float"),
    "INT8+Prune25": ("Prune25", "int8"),
    "INT8+Prune50": ("Prune50", "int8"),
    "INT8+Prune75": ("Prune75", "int8"),
}


def load_export_model(ckpt: dict[str, Any], checkpoint_path: Path) -> torch.nn.Module:
    """Load deployment model respecting pruned `hidden` saved in compression checkpoints."""
    return model_from_checkpoint(ckpt, checkpoint_path)


def checkpoint_for_config(
    config: str,
    *,
    model: str,
    fold: int,
    seed: int,
) -> Path:
    from kfold_export_paths import compression_checkpoint

    source, _ = EXPORT_PLAN[config]
    return compression_checkpoint(source, model=model, fold=fold, seed=seed)


def sample_rep_windows(
    ckpt: dict[str, Any],
    *,
    data_root: Path,
    split_file: Path,
    n_windows: int,
    seed: int,
) -> np.ndarray:
    """Return normalized windows (N, 50, 12) for INT8 calibration."""
    cfg = ckpt["config"]
    norm = norm_from_checkpoint(ckpt)
    split_map = load_split(split_file)
    files = files_for_split(data_root, split_map, "train")
    if not files:
        raise ValueError(f"No train files under {data_root}")

    window = int(cfg.get("window", 50))
    decimate = int(cfg.get("decimate", 1))
    feature_columns = cfg.get("feature_columns") or IMU_INPUT_COLUMNS

    rng = np.random.default_rng(seed)
    file_order = rng.permutation(len(files))
    windows: list[np.ndarray] = []

    for file_idx in file_order:
        path = files[int(file_idx)]
        x, _ = load_trial(
            path,
            feature_columns=feature_columns,
            label_column=cfg.get("label_column", "phase"),
            decimate=decimate,
        )
        x = norm.normalize(x)
        if len(x) < window:
            continue
        stride = max(1, (len(x) - window) // max(1, n_windows // len(files)))
        for end in range(window - 1, len(x), stride):
            start = end - window + 1
            windows.append(x[start : end + 1].astype(np.float32))
            if len(windows) >= n_windows:
                return np.stack(windows[:n_windows], axis=0)

    if not windows:
        raise RuntimeError("Could not sample representative windows from train split.")
    return np.stack(windows[:n_windows], axis=0)


def window_ntc_to_nchw(window: np.ndarray) -> np.ndarray:
    """Map (50, 12) or (N, 50, 12) windows to TFLite NCHW layout (1, 12, 50)."""
    arr = np.asarray(window, dtype=np.float32)
    if arr.ndim == 2:
        return arr.T.reshape(1, arr.shape[1], arr.shape[0])
    if arr.ndim == 3:
        return np.transpose(arr, (0, 2, 1))
    raise ValueError(f"Expected window rank 2 or 3, got shape {arr.shape}")


def export_onnx(model: torch.nn.Module, onnx_path: Path) -> None:
    model.eval().cpu()
    dummy = torch.randn(1, 50, 12, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=18,
        do_constant_folding=True,
        dynamic_axes=None,
    )


def onnx_to_saved_model(onnx_path: Path, saved_model_dir: Path) -> None:
    from onnx2tf import convert

    convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(saved_model_dir),
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True,
    )


def saved_model_to_float_tflite(saved_model_dir: Path, tflite_path: Path) -> None:
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    tflite_model = converter.convert()
    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    tflite_path.write_bytes(tflite_model)


def quantize_float_tflite(
    float_tflite_path: Path,
    int8_tflite_path: Path,
    rep_windows: np.ndarray,
) -> None:
    from ai_edge_litert.interpreter import Interpreter
    from ai_edge_quantizer import algorithm_manager, qtyping, quantizer
    from ai_edge_quantizer.utils import tfl_interpreter_utils

    interpreter = Interpreter(model_path=str(float_tflite_path))
    interpreter.allocate_tensors()
    signature_runner = interpreter.get_signature_runner(
        tfl_interpreter_utils.DEFAULT_SIGNATURE_KEY
    )
    signature_input_name = next(iter(signature_runner.get_input_details().keys()))

    qt = quantizer.Quantizer(str(float_tflite_path))
    qt.update_quantization_recipe(
        regex=".*",
        operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        op_config=qtyping.OpQuantizationConfig(
            activation_tensor_config=qtyping.TensorQuantizationConfig(
                num_bits=8,
                symmetric=False,
            ),
            weight_tensor_config=qtyping.TensorQuantizationConfig(
                num_bits=8,
                symmetric=True,
            ),
            compute_precision=qtyping.ComputePrecision.INTEGER,
        ),
        algorithm_key=algorithm_manager.AlgorithmName.MIN_MAX_UNIFORM_QUANT,
    )

    samples = []
    for window in rep_windows:
        tensor = window_ntc_to_nchw(window)
        samples.append({signature_input_name: tensor})

    calibration_data = {
        tfl_interpreter_utils.DEFAULT_SIGNATURE_KEY: samples,
    }
    calibration_result = qt.calibrate(calibration_data)
    qt.quantize(
        calibration_result=calibration_result,
        serialize_to_path=str(int8_tflite_path),
        enable_progress_report=False,
    )


def export_config(
    config: str,
    *,
    model: str,
    fold: int,
    seed: int,
    out_dir: Path,
    data_root: Path,
    split_file: Path,
    rep_windows: int,
    seed_calib: int,
    checkpoint_override: Path | None = None,
) -> dict[str, Any]:
    if checkpoint_override is not None:
        if config not in ("FP32", "INT8"):
            raise ValueError(
                f"--checkpoint override only supports FP32/INT8 (no pruned source "
                f"checkpoint exists for a baseline-only model); got config={config!r}."
            )
        source_ckpt_path = checkpoint_override
        quantize = "int8" if config == "INT8" else "float"
    else:
        source_ckpt_path = checkpoint_for_config(config, model=model, fold=fold, seed=seed)
        _, quantize = EXPORT_PLAN[config]

    ckpt = load_checkpoint(source_ckpt_path)
    model = load_export_model(ckpt, source_ckpt_path)

    config_dir = out_dir / config
    config_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = config_dir / "model.onnx"
    saved_model_dir = config_dir / "saved_model"
    float_tflite_path = config_dir / "model.float.tflite"
    tflite_path = config_dir / "model.tflite"

    export_onnx(model, onnx_path)

    if saved_model_dir.exists():
        shutil.rmtree(saved_model_dir)
    onnx_to_saved_model(onnx_path, saved_model_dir)
    saved_model_to_float_tflite(saved_model_dir, float_tflite_path)

    rep = None
    if quantize == "int8":
        rep = sample_rep_windows(
            ckpt,
            data_root=data_root,
            split_file=split_file,
            n_windows=rep_windows,
            seed=seed_calib,
        )
        rep_path = config_dir / "rep_windows.npy"
        np.save(rep_path, rep)
        quantize_float_tflite(float_tflite_path, tflite_path, rep)
    else:
        shutil.copyfile(float_tflite_path, tflite_path)

    hidden = int(ckpt["config"].get("hidden", 32))
    meta = {
        "config": config,
        "source_checkpoint": str(source_ckpt_path),
        "quantize": quantize,
        "hidden": hidden,
        "tflite_bytes": tflite_path.stat().st_size,
        "tflite_kb": round(tflite_path.stat().st_size / 1024, 2),
        "onnx_path": str(onnx_path),
        "tflite_path": str(tflite_path),
        "input_shape": [1, 12, 50],
        "input_layout": "NCHW (channels-first conv1d)",
        "pytorch_input_shape": [1, 50, 12],
        "output_shape": [1, 4],
        "input_dtype": "float32",
        "note": (
            "INT8 uses TFLite full-integer PTQ from float weights; "
            "PyTorch INT8 QDQ proxy is not exported."
            if quantize == "int8"
            else "Float32 TFLite export."
        ),
    }
    (config_dir / "export_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{config}: wrote {tflite_path} ({meta['tflite_kb']} KB, hidden={hidden})")
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--configs",
        nargs="+",
        default=list(EXPORT_CONFIGS),
        choices=EXPORT_CONFIGS,
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=ESP32_DIR / "exports")
    p.add_argument("--data-root", type=Path, default=DATA_XY)
    p.add_argument("--split-file", type=Path, default=None, help="Default: shared/splits/folds/fold{N}.csv")
    p.add_argument("--rep-windows", type=int, default=100, help="Calibration windows for INT8 PTQ.")
    p.add_argument("--calib-seed", type=int, default=42, help="RNG seed for INT8 calibration windows.")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Override source checkpoint (e.g., a baseline-only model with no compression "
            "grid). Only FP32/INT8 configs are supported in this mode."
        ),
    )
    args = p.parse_args()

    split_file = args.split_file
    if split_file is None:
        from kfold_export_paths import fold_split_file

        split_file = fold_split_file(args.fold)

    summary: dict[str, Any] = {
        "model": args.model,
        "fold": args.fold,
        "seed": args.seed,
        "split_file": str(split_file),
        "configs": {},
    }
    for config in args.configs:
        summary["configs"][config] = export_config(
            config,
            model=args.model,
            fold=args.fold,
            seed=args.seed,
            out_dir=args.out_dir,
            data_root=args.data_root,
            split_file=split_file,
            rep_windows=args.rep_windows,
            seed_calib=args.calib_seed,
            checkpoint_override=args.checkpoint,
        )

    summary_path = args.out_dir / "export_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
