"""Tests for structured channel pruning utilities."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
COMPRESSION_DIR = RESEARCH_ROOT / "experiments" / "compression"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_model_builder(model_name: str):
    path = RESEARCH_ROOT / "experiments" / model_name / "model.py"
    spec = importlib.util.spec_from_file_location(f"{model_name}.model", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.build_model


def test_prune_tcn_halves_hidden_channels():
    prune_utils = _load_module("prune_utils", COMPRESSION_DIR / "prune_utils.py")
    build_model = _load_model_builder("tcn")
    model = build_model(n_channels=12, hidden=32)
    pruned = prune_utils.prune_hidden_channels(model, keep_ratio=0.5)
    assert pruned.head[-1].in_features == 16
    assert len(pruned.stem) == 4
    assert pruned.stem[-1].conv.out_channels == 16


def test_prune_preserves_forward_shape():
    prune_utils = _load_module("prune_utils", COMPRESSION_DIR / "prune_utils.py")
    build_model = _load_model_builder("tcn")
    model = build_model(n_channels=12, hidden=32)
    pruned = prune_utils.prune_hidden_channels(model, keep_ratio=0.75)
    x = torch.randn(2, 50, 12)
    out = pruned(x)
    assert out.shape == (2, 4)
