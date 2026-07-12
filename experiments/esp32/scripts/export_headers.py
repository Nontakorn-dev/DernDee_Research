#!/usr/bin/env python3
"""Generate Arduino headers: norm stats, replay input window, and TFLite C arrays."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RESEARCH_ROOT = Path(__file__).resolve().parents[3]
SHARED = RESEARCH_ROOT / "shared"
TCN_DIR = RESEARCH_ROOT / "experiments" / "tcn"
ESP32_DIR = RESEARCH_ROOT / "experiments" / "esp32"

sys.path.insert(0, str(SHARED))
sys.path.insert(0, str(RESEARCH_ROOT))
sys.path.insert(0, str(RESEARCH_ROOT / "experiments" / "esp32"))

from data.dataset import NormStats, load_trial  # noqa: E402
from data.splits import files_for_split, load_split  # noqa: E402
from eval_checkpoint import load_checkpoint, norm_from_checkpoint  # noqa: E402
from gait_labels import IMU_INPUT_COLUMNS, PHASE_NAMES  # noqa: E402
from paths import DATA_XY, SHARED_SPLITS, TCN_RUNS  # noqa: E402

EXPORT_CONFIGS = ("FP32", "INT8", "Prune50", "INT8+Prune50")


def _fmt_float_array(values: list[float] | np.ndarray, *, per_line: int = 4) -> str:
    vals = [float(v) for v in values]
    lines: list[str] = []
    for i in range(0, len(vals), per_line):
        chunk = ", ".join(f"{v:.8f}f" for v in vals[i : i + per_line])
        lines.append(f"  {chunk},")
    return "\n".join(lines)


def _fmt_uint8_array(data: bytes, *, per_line: int = 12) -> str:
    vals = list(data)
    lines: list[str] = []
    for i in range(0, len(vals), per_line):
        chunk = ", ".join(f"0x{b:02x}" for b in vals[i : i + per_line])
        lines.append(f"  {chunk},")
    return "\n".join(lines)


def write_norm_stats_header(norm: NormStats, out_path: Path) -> None:
    content = f"""#pragma once
// Auto-generated from training norm_stats.json. Channel order matches IMU_INPUT_COLUMNS.

static const int kNumChannels = 12;
static const int kWindowSize = 50;

static const float kNormMean[kNumChannels] = {{
{_fmt_float_array(norm.mean)}
}};

static const float kNormStd[kNumChannels] = {{
{_fmt_float_array(norm.std)}
}};
"""
    out_path.write_text(content)


def deterministic_replay_window(
    *,
    data_root: Path,
    split_file: Path,
    checkpoint_path: Path,
    window_index: int,
) -> tuple[np.ndarray, np.ndarray, int, str]:
    """Return raw (50,12), normalized (50,12), label, and source trial path."""
    ckpt = load_checkpoint(checkpoint_path)
    cfg = ckpt["config"]
    norm = norm_from_checkpoint(ckpt, checkpoint_path)

    split_map = load_split(split_file)
    files = sorted(files_for_split(data_root, split_map, "test"))
    if not files:
        raise ValueError("No test files found for replay window.")

    window = int(cfg.get("window", 50))
    decimate = int(cfg.get("decimate", 1))
    feature_columns = cfg.get("feature_columns") or IMU_INPUT_COLUMNS
    label_column = cfg.get("label_column", "phase")

    collected = 0
    for path in files:
        x_raw, y = load_trial(
            path,
            feature_columns=feature_columns,
            label_column=label_column,
            decimate=decimate,
        )
        if len(x_raw) < window:
            continue
        for end in range(window - 1, len(x_raw)):
            if collected == window_index:
                start = end - window + 1
                raw = x_raw[start : end + 1].astype(np.float32)
                normalized = norm.normalize(raw.copy())
                label = int(y[end])
                return raw, normalized, label, str(path)
            collected += 1

    raise IndexError(f"window_index={window_index} exceeds available test windows ({collected}).")


def write_test_window_header(
    raw: np.ndarray,
    normalized: np.ndarray,
    label: int,
    source_trial: str,
    out_path: Path,
) -> None:
    raw_rows = []
    norm_rows = []
    for t in range(raw.shape[0]):
        raw_rows.append("  {" + ", ".join(f"{v:.8f}f" for v in raw[t]) + "},")
        norm_rows.append("  {" + ", ".join(f"{v:.8f}f" for v in normalized[t]) + "},")

    phase_name = PHASE_NAMES.get(label, "UNK")
    content = f"""#pragma once
// Deterministic replay window for host/device smoke tests.
// Source trial: {source_trial}
// Label at window end: {label} ({phase_name})

static const float kReplayWindowRaw[kWindowSize][kNumChannels] = {{
{chr(10).join(raw_rows)}
}};

static const float kReplayWindowNormalized[kWindowSize][kNumChannels] = {{
{chr(10).join(norm_rows)}
}};

static const int kReplayLabel = {label};
"""
    out_path.write_text(content)


def write_model_header(config: str, tflite_path: Path, out_path: Path) -> None:
    data = tflite_path.read_bytes()
    guard = config.upper().replace("+", "_PLUS_").replace("-", "_")
    content = f"""#pragma once
// TFLite model bytes for config {config}
// Source: {tflite_path}
// Size: {len(data)} bytes

alignas(8) const unsigned char g_model_data[] = {{
{_fmt_uint8_array(data)}
}};

const unsigned int g_model_data_len = {len(data)};
"""
    out_path.write_text(content)


def write_benchmark_config(config: str, out_path: Path, *, tensor_arena_kb: int) -> None:
    guard = config.upper().replace("+", "_PLUS_").replace("-", "_")
    content = f"""#pragma once
// Active benchmark build configuration.

#define MODEL_CONFIG "{config}"
#define TENSOR_ARENA_SIZE ({tensor_arena_kb} * 1024)
#define WARMUP_INVOCATIONS 20
#define BENCHMARK_INVOCATIONS 1000
"""
    out_path.write_text(content)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--configs", nargs="+", default=list(EXPORT_CONFIGS), choices=EXPORT_CONFIGS)
    p.add_argument("--exports-dir", type=Path, default=ESP32_DIR / "exports")
    p.add_argument(
        "--arduino-dir",
        type=Path,
        default=ESP32_DIR / "arduino" / "tcn_benchmark",
    )
    p.add_argument("--model", default="tcn")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--norm-stats", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--data-root", type=Path, default=DATA_XY)
    p.add_argument("--split-file", type=Path, default=None)
    p.add_argument("--window-index", type=int, default=0, help="Deterministic test-window index.")
    p.add_argument("--tensor-arena-kb", type=int, default=120)
    p.add_argument("--active-config", type=str, default="INT8", choices=EXPORT_CONFIGS)
    args = p.parse_args()

    from kfold_export_paths import (
        baseline_checkpoint,
        baseline_norm_stats,
        compression_checkpoint,
        fold_split_file,
    )

    norm_stats_path = args.norm_stats or baseline_norm_stats(
        model=args.model, fold=args.fold, seed=args.seed
    )
    checkpoint_path = args.checkpoint or compression_checkpoint(
        "FP32", model=args.model, fold=args.fold, seed=args.seed
    )
    split_file = args.split_file or fold_split_file(args.fold)

    headers_dir = args.arduino_dir
    headers_dir.mkdir(parents=True, exist_ok=True)

    norm = NormStats.load(norm_stats_path)
    write_norm_stats_header(norm, headers_dir / "norm_stats.h")

    raw, normalized, label, source_trial = deterministic_replay_window(
        data_root=args.data_root,
        split_file=split_file,
        checkpoint_path=checkpoint_path,
        window_index=args.window_index,
    )
    write_test_window_header(raw, normalized, label, source_trial, headers_dir / "test_window.h")

    replay_meta = {
        "window_index": args.window_index,
        "source_trial": source_trial,
        "label": label,
        "phase": PHASE_NAMES.get(label, "UNK"),
    }
    (headers_dir / "test_window_meta.json").write_text(json.dumps(replay_meta, indent=2))

    generated: dict[str, str] = {}
    for config in args.configs:
        tflite_path = args.exports_dir / config / "model.tflite"
        if not tflite_path.exists():
            raise FileNotFoundError(
                f"Missing {tflite_path}. Run experiments/esp32/scripts/export_tflite.py first."
            )
        header_name = f"model_data_{config.lower().replace('+', '_plus')}.h"
        out_path = headers_dir / header_name
        write_model_header(config, tflite_path, out_path)
        generated[config] = str(out_path)

    write_benchmark_config(args.active_config, headers_dir / "benchmark_config.h", tensor_arena_kb=args.tensor_arena_kb)

    active_header = headers_dir / f"model_data_{args.active_config.lower().replace('+', '_plus')}.h"
    (headers_dir / "model_data.h").write_text(active_header.read_text())

    manifest = {
        "active_config": args.active_config,
        "tensor_arena_kb": args.tensor_arena_kb,
        "headers_dir": str(headers_dir),
        "model_headers": generated,
        "replay_window": replay_meta,
    }
    (headers_dir / "headers_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
