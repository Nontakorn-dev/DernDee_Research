#!/usr/bin/env python3
"""Smoke-validate TFLite exports against PyTorch on one fixed replay window."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

RESEARCH_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = Path(__file__).resolve().parent
SHARED = RESEARCH_ROOT / "shared"
TINYTCN_DIR = RESEARCH_ROOT / "experiments" / "tinytcn"
ESP32_DIR = RESEARCH_ROOT / "experiments" / "esp32"
COMPRESSION_RUNS = RESEARCH_ROOT / "experiments" / "compression" / "runs"

sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SHARED))
sys.path.insert(0, str(TINYTCN_DIR))

from eval_checkpoint import load_checkpoint  # noqa: E402
from export_headers import deterministic_replay_window  # noqa: E402
from export_tflite import EXPORT_PLAN, EXPORT_CONFIGS, load_export_model, window_ntc_to_nchw  # noqa: E402
from gait_labels import PHASE_NAMES  # noqa: E402
from paths import DATA_XY, SHARED_SPLITS, TINYTCN_RUNS  # noqa: E402


def run_tflite(tflite_path: Path, normalized_window: np.ndarray) -> np.ndarray:
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    x = window_ntc_to_nchw(normalized_window)
    if input_details["dtype"] == np.int8:
        scale, zero_point = input_details["quantization"]
        x = np.round(x / scale + zero_point).astype(np.int8)
    else:
        x = x.astype(np.float32)

    interpreter.set_tensor(input_details["index"], x)
    interpreter.invoke()
    logits = interpreter.get_tensor(output_details["index"]).reshape(-1)
    if output_details["dtype"] == np.int8:
        scale, zero_point = output_details["quantization"]
        logits = (logits.astype(np.float32) - zero_point) * scale
    return logits.astype(np.float32)


def run_pytorch(checkpoint_path: Path, normalized_window: np.ndarray) -> np.ndarray:
    ckpt = load_checkpoint(checkpoint_path)
    model = load_export_model(ckpt, checkpoint_path)
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(normalized_window.reshape(1, 50, 12).astype(np.float32))
        logits = model(x).cpu().numpy().reshape(-1)
    return logits.astype(np.float32)


def default_atol(config: str) -> float:
    if config in ("INT8", "INT8+Prune50"):
        return 1.0
    return 0.05


def validate_config(
    config: str,
    *,
    exports_dir: Path,
    normalized_window: np.ndarray,
    label: int,
    atol: float | None,
) -> dict:
    source, _ = EXPORT_PLAN[config]
    ckpt_path = COMPRESSION_RUNS / source / "model.pt"
    tflite_path = exports_dir / config / "model.tflite"

    tol = default_atol(config) if atol is None else atol
    pt_logits = run_pytorch(ckpt_path, normalized_window)
    tflite_logits = run_tflite(tflite_path, normalized_window)

    pt_pred = int(np.argmax(pt_logits))
    tflite_pred = int(np.argmax(tflite_logits))
    max_abs_diff = float(np.max(np.abs(pt_logits - tflite_logits)))

    result = {
        "config": config,
        "source_checkpoint": str(ckpt_path),
        "tflite_path": str(tflite_path),
        "label": label,
        "label_name": PHASE_NAMES.get(label, "UNK"),
        "pytorch_logits": pt_logits.tolist(),
        "tflite_logits": tflite_logits.tolist(),
        "pytorch_pred": pt_pred,
        "tflite_pred": tflite_pred,
        "pred_match": pt_pred == tflite_pred,
        "max_abs_logit_diff": max_abs_diff,
        "atol": tol,
        "within_tol": max_abs_diff <= tol,
        "pass": (pt_pred == tflite_pred) and (max_abs_diff <= tol),
    }
    status = "PASS" if result["pass"] else "FAIL"
    print(
        f"{config}: {status} pred={tflite_pred} ({PHASE_NAMES.get(tflite_pred, '?')}) "
        f"max_abs_diff={max_abs_diff:.6f}"
    )
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--configs", nargs="+", default=list(EXPORT_CONFIGS), choices=EXPORT_CONFIGS)
    p.add_argument("--exports-dir", type=Path, default=ESP32_DIR / "exports")
    p.add_argument("--checkpoint", type=Path, default=TINYTCN_RUNS / "fp32_100hz" / "best_model.pt")
    p.add_argument("--data-root", type=Path, default=DATA_XY)
    p.add_argument("--split-file", type=Path, default=SHARED_SPLITS / "subject_split.csv")
    p.add_argument("--window-index", type=int, default=0)
    p.add_argument("--atol", type=float, default=None, help="Max abs logit diff tolerance (auto per config if unset).")
    p.add_argument("--out", type=Path, default=ESP32_DIR / "benchmarks" / "host_validation.json")
    args = p.parse_args()

    _, normalized, label, source_trial = deterministic_replay_window(
        data_root=args.data_root,
        split_file=args.split_file,
        checkpoint_path=args.checkpoint,
        window_index=args.window_index,
    )

    results = []
    for config in args.configs:
        results.append(
            validate_config(
                config,
                exports_dir=args.exports_dir,
                normalized_window=normalized,
                label=label,
                atol=args.atol,
            )
        )

    payload = {
        "window_index": args.window_index,
        "source_trial": source_trial,
        "label": label,
        "label_name": PHASE_NAMES.get(label, "UNK"),
        "results": results,
        "all_pass": all(r["pass"] for r in results),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.out}")
    if not payload["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
