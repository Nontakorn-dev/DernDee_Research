#!/usr/bin/env python3
"""Per-subject blocked macro F1 for the FP32 TCN baseline (K=5 folds x 3 seeds).

Window-level macro F1 (Table I) pools all ~98%-overlapping stride-1 test windows
together, which lets subjects with more/longer trials dominate the metric. This
script instead evaluates each test subject in isolation, computes that subject's
own macro F1, then averages across subjects with equal weight per fold x seed --
the "per-subject blocked" aggregation promised in Section III-A.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RESEARCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RESEARCH_ROOT / "shared"))
sys.path.insert(0, str(RESEARCH_ROOT))

import torch  # noqa: E402
from data.dataset import probe_label_column  # noqa: E402
from data.splits import discover_trial_files, load_split  # noqa: E402
from eval_checkpoint import load_checkpoint, model_from_checkpoint, norm_from_checkpoint  # noqa: E402
from evaluate import classification_metrics  # noqa: E402
from gait_labels import IMU_CHANNEL_SETS  # noqa: E402
from paths import DATA_XY  # noqa: E402
from training import make_loader, pick_device  # noqa: E402

FOLDS = (0, 1, 2, 3, 4)
SEEDS = (42, 123, 456)


def subject_files_for_fold(split_file: Path, data_root: Path) -> dict[str, list[Path]]:
    split_map = load_split(split_file)
    grouped: dict[str, list[Path]] = {}
    for subject in split_map["test"]:
        files = discover_trial_files(data_root, [subject])
        if files:
            grouped[subject] = files
    return grouped


def evaluate_subject(
    model: torch.nn.Module,
    files: list[Path],
    *,
    cfg: dict,
    norm,
    device: torch.device,
    batch_size: int = 512,
) -> dict:
    channels = cfg.get("channels", "bilateral")
    feature_columns = cfg.get("feature_columns") or IMU_CHANNEL_SETS[channels]
    label_column = cfg.get("label_column") or probe_label_column(files, channels)
    loader, labels = make_loader(
        files,
        feature_columns=feature_columns,
        label_column=label_column,
        window_size=int(cfg["window"]),
        stride=1,
        norm=norm,
        decimate=int(cfg.get("decimate", 1)),
        batch_size=batch_size,
        shuffle=False,
        eager=True,
        cache_trials=3,
    )
    model.eval()
    preds: list[int] = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds.extend(logits.argmax(dim=1).cpu().tolist())
    return classification_metrics(np.asarray(labels), np.asarray(preds))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="tcn")
    p.add_argument("--config", default="FP32")
    p.add_argument("--folds", nargs="+", type=int, default=list(FOLDS))
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--data-root", type=Path, default=DATA_XY)
    p.add_argument(
        "--out",
        type=Path,
        default=RESEARCH_ROOT / "analysis" / "blocked_macro_f1_results.json",
    )
    args = p.parse_args()

    device = pick_device("auto")
    per_run = []

    for seed in args.seeds:
        for fold in args.folds:
            ckpt_path = (
                RESEARCH_ROOT / "experiments" / "compression" / "runs" / args.model
                / f"fold{fold}_seed{seed}" / args.config / "model.pt"
            )
            split_file = RESEARCH_ROOT / "shared" / "splits" / "folds" / f"fold{fold}.csv"

            ckpt = load_checkpoint(ckpt_path)
            model = model_from_checkpoint(ckpt, ckpt_path).to(device)
            norm = norm_from_checkpoint(ckpt, ckpt_path)
            cfg = ckpt["config"]

            grouped = subject_files_for_fold(split_file, args.data_root)
            subject_f1s = []
            subject_details = {}
            for subject, files in grouped.items():
                metrics = evaluate_subject(model, files, cfg=cfg, norm=norm, device=device)
                subject_f1s.append(metrics["macro_f1"])
                subject_details[subject] = metrics["macro_f1"]

            blocked_macro_f1 = float(np.mean(subject_f1s))
            per_run.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "n_subjects": len(subject_f1s),
                    "blocked_macro_f1": blocked_macro_f1,
                    "subject_macro_f1": subject_details,
                }
            )
            print(
                f"fold{fold}_seed{seed}: blocked_macro_f1={blocked_macro_f1 * 100:.2f} "
                f"over {len(subject_f1s)} subjects"
            )

    scores = np.array([r["blocked_macro_f1"] for r in per_run])
    summary = {
        "config": args.config,
        "n_runs": len(per_run),
        "blocked_macro_f1_mean": float(scores.mean()),
        "blocked_macro_f1_std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
        "runs": per_run,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nBlocked macro F1: {summary['blocked_macro_f1_mean'] * 100:.2f} +/- {summary['blocked_macro_f1_std'] * 100:.2f}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
