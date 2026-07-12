"""Tests for k-fold compression config registry and name parsing."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH_ROOT))
sys.path.insert(0, str(RESEARCH_ROOT / "experiments" / "compression"))

from experiments.compression.kfold_configs import (  # noqa: E402
    COMPRESSION_CONFIGS,
    FINETUNE_ABLATION_CONFIGS,
    PRIMARY_COMPRESSION_CONFIGS,
    compression_eval_cli_args,
    finetune_epochs_for_config,
    normalize_config_label,
)
from run_eval import parse_config_name  # noqa: E402


def test_compression_config_counts():
    assert len(PRIMARY_COMPRESSION_CONFIGS) == 6
    assert len(FINETUNE_ABLATION_CONFIGS) == 2
    assert len(COMPRESSION_CONFIGS) == 8


def test_normalize_config_label():
    assert normalize_config_label("Prune50") == ("Prune50", None)
    assert normalize_config_label("Prune50_ft15") == ("Prune50", 15)
    assert normalize_config_label("Prune50_ft50") == ("Prune50", 50)


def test_compression_eval_cli_args():
    assert compression_eval_cli_args("FP32") == []
    assert compression_eval_cli_args("Prune50_ft15") == ["--prune-finetune-epochs", "15"]


@pytest.mark.parametrize(
    ("name", "kind", "bits", "keep"),
    [
        ("FP32", "fp32", None, None),
        ("INT8", "quant", 8, None),
        ("INT4", "quant", 4, None),
        ("Prune25", "prune", None, 0.25),
        ("Prune50", "prune", None, 0.5),
        ("Prune75", "prune", None, 0.75),
        ("INT8+Prune50", "quant_prune", 8, 0.5),
        ("INT4+Prune50", "quant_prune", 4, 0.5),
        ("Prune50_ft15", "prune", None, 0.5),
        ("Prune50_ft50", "prune", None, 0.5),
    ],
)
def test_parse_config_name(name: str, kind: str, bits: int | None, keep: float | None):
    parsed_kind, parsed_bits, parsed_keep = parse_config_name(name, default_keep_ratio=0.5)
    assert parsed_kind == kind
    assert parsed_bits == bits
    assert parsed_keep == keep


def test_finetune_epochs_for_config():
    assert finetune_epochs_for_config("FP32") is None
    assert finetune_epochs_for_config("INT8") is None
    assert finetune_epochs_for_config("Prune50") == 5
    assert finetune_epochs_for_config("Prune50_ft15") == 15
    assert finetune_epochs_for_config("INT8+Prune50") == 5
