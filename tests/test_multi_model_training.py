from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "shared"))

from config_loader import load_train_config, model_kwargs_for
from eval_checkpoint import model_from_checkpoint
from model_registry import SUPPORTED_MODELS, build_model_from_config, load_build_model
from train_runner import STANDARD_ARTIFACTS


@pytest.fixture
def shared_config():
    return load_train_config()


def test_shared_config_contains_fairness_keys(shared_config):
    data = shared_config["data"]
    train = shared_config["train"]

    assert data["channels"] == "bilateral"
    assert data["source_hz"] == 200
    assert data["target_hz"] == 100
    assert data["window"] == 50
    assert data["train_stride"] == 5
    assert data["val_stride"] == 1
    assert data["test_stride"] == 1
    assert train["epochs"] == 50
    assert train["batch_size"] == 512
    assert train["lr"] == 0.001
    assert train["optimizer"] == "adam"
    assert train["class_weights"] == "balanced"
    assert shared_config["selection"]["metric"] == "val_macro_f1"


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_model_forward_contract(model_name):
    cfg = load_train_config()
    kwargs = model_kwargs_for(cfg, model_name)
    model = build_model_from_config(model_name, n_channels=12, model_kwargs=kwargs)
    xb = torch.zeros(2, 50, 12)
    logits = model(xb)
    assert logits.shape == (2, 4)


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_build_model_loader(model_name):
    build_model = load_build_model(model_name)
    model = build_model(n_channels=12, n_classes=4, hidden=32)
    assert model(torch.zeros(1, 50, 12)).shape == (1, 4)


def test_standard_artifact_names():
    expected = {
        "best_model.pt",
        "norm_stats.json",
        "history.json",
        "val_report.txt",
        "test_report.txt",
        "val_confusion_matrix.json",
        "test_confusion_matrix.json",
        "val_metrics.json",
        "test_metrics.json",
        "config.json",
    }
    assert set(STANDARD_ARTIFACTS) == expected


def test_checkpoint_reload_with_model_name():
    cfg = load_train_config()
    kwargs = model_kwargs_for(cfg, "cnn1d")
    model = build_model_from_config("cnn1d", n_channels=12, model_kwargs=kwargs)

    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": {
            "model_name": "cnn1d",
            "model_kwargs": kwargs,
            "window": 50,
            "channels": "bilateral",
            "feature_columns": [f"ch{i}" for i in range(12)],
            "label_column": "phase_lt",
            "n_channels": 12,
            "n_classes": 4,
            "hidden": 32,
            "source_hz": 200,
            "target_hz": 100,
            "decimate": 2,
        },
        "norm_stats": {"mean": [0.0] * 12, "std": [1.0] * 12},
    }

    restored = model_from_checkpoint(ckpt, Path("experiments/cnn1d/runs/fp32_100hz/best_model.pt"))
    out = restored(torch.zeros(1, 50, 12))
    assert out.shape == (1, 4)


def test_legacy_tinytcn_checkpoint_fallback():
    sys.path.insert(0, str(ROOT / "experiments" / "tinytcn"))
    from model import build_model

    model = build_model(n_channels=12, n_classes=4, hidden=32)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": {
            "window": 50,
            "channels": "bilateral",
            "feature_columns": ["lt_acc_x"] * 12,
            "label_column": "phase_lt",
            "n_channels": 12,
            "n_classes": 4,
            "hidden": 32,
            "source_hz": 200,
            "target_hz": 100,
            "decimate": 2,
        },
        "norm_stats": {"mean": [0.0] * 12, "std": [1.0] * 12},
    }
    restored = model_from_checkpoint(
        ckpt,
        Path("experiments/tinytcn/runs/fp32_100hz/best_model.pt"),
    )
    assert restored(torch.zeros(1, 50, 12)).shape == (1, 4)


def test_config_json_is_valid():
    config_path = ROOT / "shared" / "configs" / "train_fair_comparison.json"
    payload = json.loads(config_path.read_text())
    assert payload["name"] == "train_fair_comparison"
    assert isinstance(payload["model_overrides"], dict)
