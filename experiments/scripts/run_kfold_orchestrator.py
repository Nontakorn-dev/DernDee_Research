#!/usr/bin/env python3
"""Orchestrate k-fold overnight jobs with manifest skip/resume."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

RESEARCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RESEARCH_ROOT))
sys.path.insert(0, str(RESEARCH_ROOT / "shared"))

from experiments.compression.kfold_configs import compression_eval_cli_args
from experiments.kfold_manifest import (  # noqa: E402
    MANIFEST_PATH,
    baseline_run_dir,
    compression_run_dir,
    mark_job,
    next_pending_job,
    split_file_for_fold,
    sync_manifest,
)

TRAIN_SCRIPT = {
    "tcn": RESEARCH_ROOT / "experiments" / "tcn" / "train.py",
    "cnn1d": RESEARCH_ROOT / "experiments" / "cnn1d" / "train.py",
    "lstm_gru": RESEARCH_ROOT / "experiments" / "lstm_gru" / "train.py",
    "transformer": RESEARCH_ROOT / "experiments" / "transformer" / "train.py",
    "cnn_lstm": RESEARCH_ROOT / "experiments" / "cnn_lstm" / "train.py",
}


def load_config_template() -> dict:
    path = RESEARCH_ROOT / "shared" / "configs" / "train_fair_comparison.json"
    return json.loads(path.read_text())


def config_for_job(job: dict) -> Path:
    cfg = load_config_template()
    fold = job["fold"]
    seed = job["seed"]
    cfg["train"]["seed"] = seed
    cfg["data"]["split_file"] = f"shared/splits/folds/fold{fold}.csv"

    out_dir = RESEARCH_ROOT / "experiments" / job["model"] / "configs" / f"fold{fold}_seed{seed}.json"
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    out_dir.write_text(json.dumps(cfg, indent=2))
    return out_dir


def run_baseline(
    job: dict,
    *,
    lazy: bool,
    data_root: Path | None,
    device: str | None,
) -> int:
    model = job["model"]
    fold = job["fold"]
    seed = job["seed"]
    config_path = config_for_job(job)
    run_dir = baseline_run_dir(model, fold, seed)

    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT[model]),
        "--config",
        str(config_path),
        "--output-dir",
        str(run_dir),
    ]
    if data_root is not None:
        cmd.extend(["--data-root", str(data_root)])
    if device is not None:
        cmd.extend(["--device", device])
    if lazy:
        cmd.append("--lazy")

    print(f"Running baseline: {model} fold={fold} seed={seed}")
    mark_job(MANIFEST_PATH, job, status="running")
    result = subprocess.run(cmd, cwd=RESEARCH_ROOT)
    if result.returncode != 0:
        mark_job(MANIFEST_PATH, job, status="failed", error=f"exit {result.returncode}")
        return 1

    mark_job(MANIFEST_PATH, job, status="done")
    return 0


def run_compression(
    job: dict,
    *,
    lazy: bool,
    data_root: Path | None,
    device: str | None,
) -> int:
    model = job["model"]
    fold = job["fold"]
    seed = job["seed"]
    config_name = job["config"]
    checkpoint = baseline_run_dir(model, fold, seed) / "best_model.pt"
    out_dir = compression_run_dir(model, fold, seed, config_name)
    split_file = split_file_for_fold(fold)

    cmd = [
        sys.executable,
        str(RESEARCH_ROOT / "experiments" / "compression" / "run_eval.py"),
        "--model",
        model,
        "--checkpoint",
        str(checkpoint),
        "--split-file",
        str(split_file),
        "--out-dir",
        str(out_dir.parent),
        "--configs",
        config_name,
    ]
    if data_root is not None:
        cmd.extend(["--data-root", str(data_root)])
    if device is not None:
        cmd.extend(["--device", device])
    if lazy:
        cmd.append("--lazy")
    cmd.extend(compression_eval_cli_args(config_name))

    print(f"Running compression: {model} / {config_name} fold={fold} seed={seed}")
    mark_job(MANIFEST_PATH, job, status="running")
    result = subprocess.run(cmd, cwd=RESEARCH_ROOT)
    if result.returncode != 0:
        mark_job(MANIFEST_PATH, job, status="failed", error=f"exit {result.returncode}")
        return 1

    mark_job(MANIFEST_PATH, job, status="done")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sync-only", action="store_true")
    p.add_argument("--next", action="store_true", help="Run next pending job from manifest.")
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--phase", choices=["baseline", "compression"], default=None)
    p.add_argument("--all-folds", action="store_true")
    p.add_argument("--all-seeds", action="store_true")
    p.add_argument("--max-jobs", type=int, default=1)
    p.add_argument("--lazy", action="store_true")
    p.add_argument("--data-root", type=Path, default=None, help="Path to dataset/Xy (Colab Drive).")
    p.add_argument("--device", type=str, default=None, help="cuda, cpu, mps, or auto.")
    p.add_argument(
        "--reset-running",
        action="store_true",
        help="Reset non-completed running jobs to pending before selecting work.",
    )
    args = p.parse_args()

    payload = sync_manifest(reset_running=args.reset_running)

    if args.sync_only:
        print(f"Synced {len(payload['jobs'])} jobs → {MANIFEST_PATH}")
        return

    ran = 0
    while ran < args.max_jobs:
        job = next_pending_job(payload, model=args.model, phase=args.phase)
        if job is None:
            if ran == 0:
                print("No pending jobs.")
            break

        if job["phase"] == "baseline":
            rc = run_baseline(
                job,
                lazy=args.lazy,
                data_root=args.data_root,
                device=args.device,
            )
        else:
            rc = run_compression(
                job,
                lazy=args.lazy,
                data_root=args.data_root,
                device=args.device,
            )

        if rc != 0:
            sys.exit(rc)

        ran += 1
        payload = sync_manifest()


if __name__ == "__main__":
    main()
