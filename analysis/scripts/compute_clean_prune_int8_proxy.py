#!/usr/bin/env python3
"""PyTorch QDQ proxy for INT8+Prune25 / INT8+Prune75, computed cleanly.

Applies weight-only quantize-dequantize directly to the EXISTING Prune25/
Prune75 fine-tuned checkpoint (no independent re-finetune), so the proxy and
the real deployed TFLite accuracy (tflite_real_accuracy_batch.py, which also
sources from the Prune25/Prune75 checkpoint) differ only in quantization
method -- unlike the original INT8+Prune50 point, which re-derived pruning
and fine-tuning independently inside run_eval.py's CLI path (see Limitations).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RESEARCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RESEARCH_ROOT / "shared"))
sys.path.insert(0, str(RESEARCH_ROOT / "experiments" / "compression"))
sys.path.insert(0, str(RESEARCH_ROOT))

from eval_checkpoint import (  # noqa: E402
    evaluate_model_from_checkpoint_config,
    load_checkpoint,
    model_from_checkpoint,
)
from paths import DATA_XY  # noqa: E402
from run_eval import quantize_dequantize_weights  # noqa: E402

FOLDS = (0, 1, 2, 3, 4)
SEEDS = (42, 123, 456)
SOURCE_CONFIGS = {"INT8+Prune25": "Prune25", "INT8+Prune75": "Prune75"}


def main() -> None:
    out_dir = RESEARCH_ROOT / "analysis" / "clean_prune_int8_proxy_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    for combo_name, source_config in SOURCE_CONFIGS.items():
        for fold in FOLDS:
            for seed in SEEDS:
                out_path = out_dir / f"{combo_name.replace('+', '_plus_')}_fold{fold}_seed{seed}.json"
                if out_path.exists():
                    print(f"skip (cached): {out_path.name}")
                    continue

                ckpt_path = (
                    RESEARCH_ROOT / "experiments" / "compression" / "runs" / "tcn"
                    / f"fold{fold}_seed{seed}" / source_config / "model.pt"
                )
                split_file = RESEARCH_ROOT / "shared" / "splits" / "folds" / f"fold{fold}.csv"

                ckpt = load_checkpoint(ckpt_path)
                model = model_from_checkpoint(ckpt, ckpt_path)
                q_model = quantize_dequantize_weights(model, bits=8)

                metrics = evaluate_model_from_checkpoint_config(
                    q_model,
                    ckpt,
                    checkpoint_path=ckpt_path,
                    data_root=DATA_XY,
                    split_file=split_file,
                    split="test",
                    batch_size=512,
                    stride=1,
                    device_name="auto",
                    eager=True,
                    cache_trials=3,
                    desc=f"{combo_name} fold{fold}_seed{seed}",
                )
                result = {
                    "config": combo_name,
                    "source_config": source_config,
                    "fold": fold,
                    "seed": seed,
                    **{k: v for k, v in metrics.items() if k != "report"},
                }
                out_path.write_text(json.dumps(result, indent=2))
                print(
                    f"done {combo_name} fold{fold}_seed{seed}: "
                    f"macro_f1={metrics['macro_f1'] * 100:.2f} -> {out_path.name}"
                )


if __name__ == "__main__":
    main()
