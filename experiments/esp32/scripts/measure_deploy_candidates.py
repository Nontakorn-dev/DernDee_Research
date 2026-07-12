#!/usr/bin/env python3
"""Flash and measure ESP32 deploy candidates (manual hardware step).

This script documents the measurement workflow and validates esp32_c3_metrics.json
structure. Actual flashing requires Arduino IDE + connected ESP32-C3 board.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESEARCH_ROOT = Path(__file__).resolve().parents[3]
METRICS_PATH = RESEARCH_ROOT / "experiments" / "esp32" / "benchmarks" / "esp32_c3_metrics.json"

DEPLOY_CANDIDATES = (
    "INT8",
    "INT8+Prune50",
    "INT8+Prune25",
    "INT8+Prune75",
)


def ensure_candidate_entries(metrics: dict) -> dict:
    configs = metrics.setdefault("configs", {})
    for name in DEPLOY_CANDIDATES:
        if name not in configs:
            configs[name] = {
                "status": "pending_measurement",
                "artifact": f"experiments/esp32/exports/{name}/model.tflite",
                "trials": 1000,
                "warmup": 20,
                "tensor_arena_kb": 120,
            }
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", type=Path, default=METRICS_PATH)
    p.add_argument("--validate-only", action="store_true")
    args = p.parse_args()

    if args.metrics.exists():
        metrics = json.loads(args.metrics.read_text())
    else:
        metrics = {"board": "ESP32-C3 Dev Module", "configs": {}}

    metrics = ensure_candidate_entries(metrics)
    for name in ("INT8+Prune25", "INT8+Prune75"):
        entry = metrics["configs"].setdefault(name, {})
        if "latency_mean_ms" not in entry:
            entry.setdefault("status", "pending_measurement")
            entry.setdefault(
                "notes",
                "Requires export + ESP32 flash; see experiments/esp32/scripts/export_tflite.py",
            )
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2))

    pending = [k for k, v in metrics["configs"].items() if v.get("status") == "pending_measurement"]
    print(f"Updated {args.metrics}")
    if pending:
        print(f"Pending hardware measurement: {', '.join(pending)}")
        print("Flash via Arduino IDE: experiments/esp32/arduino/tcn_benchmark/")
    else:
        print("All deploy candidates have latency measurements.")


if __name__ == "__main__":
    main()
