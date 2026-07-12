"""Checkpoint loading respects pruned top-level hidden over stale model_kwargs."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH_ROOT / "shared"))

from eval_checkpoint import model_from_checkpoint  # noqa: E402
from model_registry import build_model_from_config  # noqa: E402


def test_model_from_checkpoint_uses_cfg_hidden_after_prune():
    model = build_model_from_config(
        "tcn",
        n_channels=12,
        model_kwargs={"hidden": 32, "n_classes": 4},
    )
    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": {
            "model_name": "tcn",
            "feature_columns": [f"c{i}" for i in range(12)],
            "hidden": 32,
            "model_kwargs": {"hidden": 32, "n_classes": 4},
        },
    }

    pruned = build_model_from_config(
        "tcn",
        n_channels=12,
        model_kwargs={"hidden": 16, "n_classes": 4},
    )
    ckpt["model_state_dict"] = pruned.state_dict()
    ckpt["config"]["hidden"] = 16
    # Stale model_kwargs still says 32, as saved by older compression runs.

    restored = model_from_checkpoint(ckpt, Path("pruned.pt"))
    assert restored.head[-1].in_features == 16
    assert restored.stem[0].conv.out_channels == 16
