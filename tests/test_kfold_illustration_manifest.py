"""Manifest builder for single-fold paper figures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH_ROOT))

from analysis.build_kfold_illustration_manifest import build_manifest  # noqa: E402


def test_build_manifest_fold0_seed42():
    manifest = build_manifest(model="tcn", fold=0, seed=42, configs=["FP32", "INT8", "Prune50", "INT8+Prune50"])
    assert manifest["source"] == "tcn/fold0_seed42"
    names = [p["name"] for p in manifest["points"]]
    assert names == ["FP32", "INT8", "Prune50", "INT8+Prune50"]

    fp32 = manifest["points"][0]
    prune = next(p for p in manifest["points"] if p["name"] == "INT8+Prune50")
    lr_f1_delta = (prune["phase_f1"]["LR"] - fp32["phase_f1"]["LR"]) * 100
    lr_acc_delta = (prune["phase_accuracy"]["LR"] - fp32["phase_accuracy"]["LR"]) * 100
    assert lr_f1_delta < -0.5
    assert abs(lr_acc_delta) < 0.1
