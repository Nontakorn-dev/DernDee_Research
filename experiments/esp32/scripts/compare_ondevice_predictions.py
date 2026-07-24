#!/usr/bin/env python3
"""Compare on-device (ESP32-C3 TFLite Micro) predictions against the desktop
ai_edge_litert interpreter on the identical set of windows.

Usage:
  1. Flash experiments/esp32/arduino/tcn_multiwindow (see its header comment).
  2. Capture Serial Monitor output to a text file (from the "idx,label,pred,..."
     header line through "# done").
  3. Run:
       python3 experiments/esp32/scripts/compare_ondevice_predictions.py \
           --serial-log captured_output.txt --active-config INT8
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

RESEARCH_ROOT = Path(__file__).resolve().parents[3]
SHARED = RESEARCH_ROOT / "shared"
ESP32_DIR = RESEARCH_ROOT / "experiments" / "esp32"

sys.path.insert(0, str(SHARED))
sys.path.insert(0, str(RESEARCH_ROOT))
sys.path.insert(0, str(RESEARCH_ROOT / "experiments" / "esp32"))
sys.path.insert(0, str(RESEARCH_ROOT / "experiments" / "esp32" / "scripts"))

from export_tflite import window_ntc_to_nchw  # noqa: E402
from gait_labels import PHASE_NAMES  # noqa: E402

ROW_RE = re.compile(r"^(\d+),(-?\d+),(-?\d+),([01]),(\d+)\s*$")


def parse_serial_log(path: Path) -> list[tuple[int, int, int, int, int]]:
    rows = []
    for line in path.read_text().splitlines():
        m = ROW_RE.match(line.strip())
        if m:
            rows.append(tuple(int(g) for g in m.groups()))
    if not rows:
        raise ValueError(
            f"No CSV rows matched in {path}. Expected lines like '0,1,1,1,142'."
        )
    return rows


def run_desktop_interpreter(tflite_path: Path, windows: np.ndarray) -> list[int]:
    from ai_edge_litert.interpreter import Interpreter

    interp = Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    is_int8 = inp["dtype"] == np.int8
    if is_int8:
        in_scale, in_zp = inp["quantization"]
        out_scale, out_zp = out["quantization"]

    preds = []
    for w in windows:
        x = window_ntc_to_nchw(w)
        if is_int8:
            xq = np.clip(np.round(x / in_scale + in_zp), -128, 127).astype(np.int8)
            interp.set_tensor(inp["index"], xq)
        else:
            interp.set_tensor(inp["index"], x.astype(np.float32))
        interp.invoke()
        logits = interp.get_tensor(out["index"]).reshape(-1).astype(np.float32)
        if is_int8:
            logits = (logits - out_zp) * out_scale
        preds.append(int(np.argmax(logits)))
    return preds


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--serial-log", type=Path, required=True)
    p.add_argument("--active-config", default="INT8")
    p.add_argument(
        "--reference",
        type=Path,
        default=ESP32_DIR / "arduino" / "tcn_multiwindow" / "multiwindow_reference.npz",
    )
    p.add_argument(
        "--tflite-path",
        type=Path,
        default=None,
        help="Default: experiments/esp32/exports/<active-config>/model.tflite",
    )
    args = p.parse_args()

    tflite_path = args.tflite_path or (ESP32_DIR / "exports" / args.active_config / "model.tflite")
    if not tflite_path.exists():
        raise FileNotFoundError(
            f"Missing {tflite_path}. Run export_tflite.py for config={args.active_config} first."
        )

    ref = np.load(args.reference)
    windows, ref_labels = ref["windows"], ref["labels"]

    device_rows = parse_serial_log(args.serial_log)
    n = min(len(device_rows), len(windows))
    if len(device_rows) != len(windows):
        print(
            f"WARNING: serial log has {len(device_rows)} rows but reference has "
            f"{len(windows)} windows; comparing the first {n}."
        )

    device_rows = device_rows[:n]
    windows = windows[:n]
    ref_labels = ref_labels[:n]

    for idx, label, _pred, _match, _lat in device_rows:
        if int(ref_labels[idx]) != label:
            raise ValueError(
                f"Label mismatch at window {idx}: serial log says {label}, "
                f"reference says {ref_labels[idx]}. Serial log does not match "
                f"this reference set -- regenerate multiwindow_data.h and reflash."
            )

    device_preds = [row[2] for row in device_rows]
    desktop_preds = run_desktop_interpreter(tflite_path, windows)

    device_preds_arr = np.array(device_preds)
    desktop_preds_arr = np.array(desktop_preds)
    labels_arr = np.array(ref_labels)

    agree = device_preds_arr == desktop_preds_arr
    device_acc = (device_preds_arr == labels_arr).mean()
    desktop_acc = (desktop_preds_arr == labels_arr).mean()

    print(f"Config: {args.active_config}")
    print(f"Windows compared: {n}")
    print(f"On-device (TFLM) vs desktop (ai_edge_litert) argmax agreement: "
          f"{agree.mean() * 100:.2f}% ({agree.sum()}/{n})")
    print(f"On-device accuracy vs ground truth:  {device_acc * 100:.2f}%")
    print(f"Desktop accuracy vs ground truth:    {desktop_acc * 100:.2f}%")

    if not agree.all():
        print("\nDisagreements (window idx, label, device_pred, desktop_pred):")
        for i in np.where(~agree)[0]:
            phase = PHASE_NAMES.get(int(labels_arr[i]), "?")
            print(f"  {i}: label={labels_arr[i]}({phase}) device={device_preds_arr[i]} desktop={desktop_preds_arr[i]}")


if __name__ == "__main__":
    main()
