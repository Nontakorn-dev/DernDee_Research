#!/usr/bin/env python3
"""Sample N test windows and emit an Arduino header + host-side companion files.

Used for an on-device TFLite Micro vs. desktop ai_edge_litert interpreter
prediction cross-check (argmax agreement across many windows, not just the
single deterministic replay window used by the latency benchmark sketch).

Usage:
  python3 experiments/esp32/scripts/export_multiwindow_header.py \
      --active-config INT8 --num-windows 500

Then build/flash experiments/esp32/arduino/tcn_multiwindow, capture the Serial
CSV output, and run:
  python3 experiments/esp32/scripts/compare_ondevice_predictions.py \
      --serial-log <captured.csv> --active-config INT8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RESEARCH_ROOT = Path(__file__).resolve().parents[3]
SHARED = RESEARCH_ROOT / "shared"
ESP32_DIR = RESEARCH_ROOT / "experiments" / "esp32"

sys.path.insert(0, str(SHARED))
sys.path.insert(0, str(RESEARCH_ROOT))
sys.path.insert(0, str(RESEARCH_ROOT / "experiments" / "esp32"))

from data.dataset import load_trial  # noqa: E402
from data.splits import files_for_split, load_split  # noqa: E402
from eval_checkpoint import load_checkpoint, norm_from_checkpoint  # noqa: E402
from gait_labels import IMU_INPUT_COLUMNS, PHASE_NAMES  # noqa: E402
from paths import DATA_XY  # noqa: E402

EXPORT_CONFIGS = ("FP32", "INT8", "Prune25", "Prune50", "Prune75", "INT8+Prune25", "INT8+Prune50", "INT8+Prune75")


def sample_windows_across_subjects(
    *,
    data_root: Path,
    split_file: Path,
    checkpoint_path: Path,
    num_windows: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, list[int], list[str]]:
    """Spread ~num_windows evenly across every test-split trial file."""
    ckpt = load_checkpoint(checkpoint_path)
    cfg = ckpt["config"]
    norm = norm_from_checkpoint(ckpt, checkpoint_path)

    split_map = load_split(split_file)
    files = sorted(files_for_split(data_root, split_map, "test"))
    if not files:
        raise ValueError("No test files found.")

    window = int(cfg.get("window", 50))
    decimate = int(cfg.get("decimate", 1))
    feature_columns = cfg.get("feature_columns") or IMU_INPUT_COLUMNS
    label_column = cfg.get("label_column", "phase")

    per_file_budget = max(1, num_windows // len(files))

    normalized_windows: list[np.ndarray] = []
    labels: list[int] = []
    trial_ids: list[int] = []
    trial_paths: list[str] = []

    rng = np.random.default_rng(42)
    for trial_id, path in enumerate(files):
        x_raw, y = load_trial(
            path, feature_columns=feature_columns, label_column=label_column, decimate=decimate
        )
        if len(x_raw) < window:
            continue
        x_norm_full = norm.normalize(x_raw.astype(np.float32).copy())
        # Evenly spaced offsets across the whole trial (not just its start) so
        # sampled windows cover the full gait-phase cycle, not just one phase.
        candidate_ends = np.arange(window - 1, len(x_raw))
        n_take = min(per_file_budget, len(candidate_ends))
        if n_take <= 0:
            continue
        chosen = np.linspace(0, len(candidate_ends) - 1, num=n_take, dtype=int)
        chosen = sorted(set(candidate_ends[chosen].tolist()))
        for end in chosen:
            start = end - window + 1
            normalized_windows.append(x_norm_full[start : end + 1])
            labels.append(int(y[end]))
            trial_ids.append(trial_id)
        trial_paths.append(str(path))
        if len(normalized_windows) >= num_windows:
            break

    return (
        np.stack(normalized_windows[:num_windows], axis=0),
        np.array(labels[:num_windows], dtype=np.int64),
        trial_ids[:num_windows],
        trial_paths,
    )


def _fmt_float_array(values, *, per_line: int = 12) -> str:
    vals = [float(v) for v in values]
    lines = []
    for i in range(0, len(vals), per_line):
        chunk = ", ".join(f"{v:.8f}f" for v in vals[i : i + per_line])
        lines.append(f"  {chunk},")
    return "\n".join(lines)


def write_multiwindow_header(
    windows: np.ndarray, labels: np.ndarray, out_path: Path
) -> None:
    n, t, c = windows.shape
    flat = windows.reshape(n, t * c)
    rows = []
    for i in range(n):
        rows.append(f"// window {i}\n{{\n{_fmt_float_array(flat[i])}\n}},")
    content = f"""#pragma once
// Auto-generated: {n} normalized test windows for on-device vs. desktop
// TFLite-interpreter prediction cross-check. Flattened (window, time*channel).

static const int kNumMultiWindows = {n};
static const int kMultiWindowFlatSize = {t * c};

static const float kMultiWindowNormalized[kNumMultiWindows][kMultiWindowFlatSize] = {{
{chr(10).join(rows)}
}};

static const int kMultiWindowLabels[kNumMultiWindows] = {{
{", ".join(str(int(v)) for v in labels)}
}};
"""
    out_path.write_text(content)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="tcn")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--active-config", default="INT8", choices=EXPORT_CONFIGS)
    p.add_argument("--num-windows", type=int, default=500)
    p.add_argument("--stride", type=int, default=15, help="Stride between sampled windows within a trial.")
    p.add_argument("--data-root", type=Path, default=DATA_XY)
    p.add_argument(
        "--arduino-dir", type=Path, default=ESP32_DIR / "arduino" / "tcn_multiwindow"
    )
    args = p.parse_args()

    from export_tflite import EXPORT_PLAN
    from kfold_export_paths import compression_checkpoint, fold_split_file

    source_config, _ = EXPORT_PLAN[args.active_config]
    checkpoint_path = compression_checkpoint(
        source_config, model=args.model, fold=args.fold, seed=args.seed
    )
    split_file = fold_split_file(args.fold)

    windows, labels, trial_ids, trial_paths = sample_windows_across_subjects(
        data_root=args.data_root,
        split_file=split_file,
        checkpoint_path=checkpoint_path,
        num_windows=args.num_windows,
        stride=args.stride,
    )

    args.arduino_dir.mkdir(parents=True, exist_ok=True)
    write_multiwindow_header(windows, labels, args.arduino_dir / "multiwindow_data.h")

    # Host-side companion: exact same windows + labels, for the desktop-interpreter
    # cross-check script to replay without re-deriving the sampling.
    np.savez(
        args.arduino_dir / "multiwindow_reference.npz",
        windows=windows,
        labels=labels,
        trial_ids=np.array(trial_ids),
    )
    meta = {
        "active_config": args.active_config,
        "source_config": source_config,
        "fold": args.fold,
        "seed": args.seed,
        "num_windows": int(len(labels)),
        "stride": args.stride,
        "trial_files": trial_paths,
        "phase_names": PHASE_NAMES,
    }
    (args.arduino_dir / "multiwindow_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {len(labels)} windows -> {args.arduino_dir / 'multiwindow_data.h'}")
    print(f"Wrote reference arrays -> {args.arduino_dir / 'multiwindow_reference.npz'}")


if __name__ == "__main__":
    main()
