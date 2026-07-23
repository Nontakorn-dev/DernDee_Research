#!/usr/bin/env python3
"""Export + evaluate real deployed TFLite accuracy for a batch of (config, fold, seed) jobs.

Writes one JSON result file per job under --out-dir; skips jobs whose result file
already exists (safe to re-run/resume across multiple invocations).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

RESEARCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RESEARCH_ROOT / "shared"))
sys.path.insert(0, str(RESEARCH_ROOT))
sys.path.insert(0, str(RESEARCH_ROOT / "experiments" / "esp32"))
sys.path.insert(0, str(RESEARCH_ROOT / "experiments" / "esp32" / "scripts"))

import numpy as np  # noqa: E402
from eval_checkpoint import load_checkpoint, norm_from_checkpoint, make_eval_loader_from_config  # noqa: E402
from evaluate import classification_metrics  # noqa: E402
from export_tflite import export_config, window_ntc_to_nchw  # noqa: E402
from paths import DATA_XY  # noqa: E402


def eval_tflite(tflite_path: Path, ckpt_path: Path, split_file: Path) -> dict:
    from ai_edge_litert.interpreter import Interpreter

    ckpt = load_checkpoint(ckpt_path)
    cfg = ckpt["config"]
    norm = norm_from_checkpoint(ckpt, ckpt_path)
    loader, _ = make_eval_loader_from_config(
        cfg, norm, data_root=DATA_XY, split_file=split_file,
        split="test", batch_size=512, stride=1, eager=True, cache_trials=3,
    )
    interp = Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    is_int8 = inp["dtype"] == np.int8
    if is_int8:
        in_scale, in_zp = inp["quantization"]
        out_scale, out_zp = out["quantization"]
    y_true, y_pred = [], []
    for xb, yb in loader:
        xb_np = xb.numpy()
        yb_np = yb.numpy()
        for i in range(xb_np.shape[0]):
            x = window_ntc_to_nchw(xb_np[i])
            if is_int8:
                xq = np.round(x / in_scale + in_zp).astype(np.int8)
                interp.set_tensor(inp["index"], xq)
            else:
                interp.set_tensor(inp["index"], x.astype(np.float32))
            interp.invoke()
            logits = interp.get_tensor(out["index"]).reshape(-1).astype(np.float32)
            if is_int8:
                logits = (logits - out_zp) * out_scale
            y_pred.append(int(np.argmax(logits)))
        y_true.extend(yb_np.tolist())
    return classification_metrics(np.array(y_true), np.array(y_pred))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, choices=["INT8", "INT8+Prune50"])
    p.add_argument("--folds", nargs="+", type=int, required=True)
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--out-dir", type=Path, default=RESEARCH_ROOT / "analysis" / "tflite_real_accuracy_results")
    p.add_argument("--scratch-dir", type=Path, default=Path("/tmp/tflite_real_acc_export"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        for fold in args.folds:
            out_path = args.out_dir / f"{args.config.replace('+', '_plus_')}_fold{fold}_seed{seed}.json"
            if out_path.exists():
                print(f"skip (cached): {out_path.name}")
                continue

            t0 = time.time()
            split_file = RESEARCH_ROOT / "shared" / "splits" / "folds" / f"fold{fold}.csv"
            ckpt_path = (
                RESEARCH_ROOT / "experiments" / "compression" / "runs" / "tcn"
                / f"fold{fold}_seed{seed}" / args.config / "model.pt"
            )

            export_dir = args.scratch_dir / f"{args.config.replace('+', '_plus_')}_{fold}_{seed}"
            if export_dir.exists():
                shutil.rmtree(export_dir)

            meta = export_config(
                args.config,
                model="tcn",
                fold=fold,
                seed=seed,
                out_dir=export_dir,
                data_root=DATA_XY,
                split_file=split_file,
                rep_windows=100,
                seed_calib=42,
            )
            tflite_path = Path(meta["tflite_path"])

            metrics = eval_tflite(tflite_path, ckpt_path, split_file)
            elapsed = time.time() - t0

            result = {
                "config": args.config,
                "fold": fold,
                "seed": seed,
                "tflite_kb": meta["tflite_kb"],
                "elapsed_s": round(elapsed, 1),
                **{k: v for k, v in metrics.items() if k != "report"},
            }
            out_path.write_text(json.dumps(result, indent=2))
            print(
                f"done fold{fold}_seed{seed} {args.config}: macro_f1={metrics['macro_f1']*100:.2f} "
                f"({elapsed:.0f}s) -> {out_path.name}"
            )

            shutil.rmtree(export_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
