from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.plot_confusion_matrix import discover_sources, matrix_accuracy
from analysis.plot_phase_degradation import measured_points, phase_deltas, phase_values
from analysis.plot_training import best_epoch
from analysis.plot_style import ordered_configs


def test_ordered_configs_follows_manifest_order() -> None:
    names = ["INT4", "FP32", "Prune50", "INT8"]
    assert ordered_configs(names) == ["FP32", "INT8", "INT4", "Prune50"]


def test_phase_values_scale_to_percent() -> None:
    point = {"phase_accuracy": {"LR": 0.9732, "LS": 0.9656, "PSw": 0.9230, "Sw": 0.9582}}
    values = phase_values(point, "phase_accuracy")
    assert values[0] == pytest.approx(97.32)
    assert values[-1] == pytest.approx(95.82)


def test_phase_deltas_relative_to_fp32() -> None:
    baseline = {"phase_accuracy": {"LR": 0.97, "LS": 0.96, "PSw": 0.92, "Sw": 0.95}}
    compressed = {"phase_accuracy": {"LR": 0.94, "LS": 0.96, "PSw": 0.88, "Sw": 0.95}}
    deltas = phase_deltas(compressed, baseline, "phase_accuracy")
    assert deltas == pytest.approx([-3.0, 0.0, -4.0, 0.0])


def test_measured_points_requires_macro_f1() -> None:
    manifest = {
        "points": [
            {"name": "FP32", "macro_f1": 0.91, "phase_accuracy": {"LR": 0.97}},
            {"name": "INT8", "macro_f1": None},
        ]
    }
    points = measured_points(manifest)
    assert [p["name"] for p in points] == ["FP32"]


def test_best_epoch_from_history() -> None:
    history = [
        {"epoch": 1, "val_macro_f1": 0.88},
        {"epoch": 2, "val_macro_f1": 0.91},
        {"epoch": 3, "val_macro_f1": 0.90},
    ]
    assert best_epoch(history) == 2


def test_discover_confusion_sources_from_fp32_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "fp32_100hz"
    run_dir.mkdir()
    (run_dir / "val_confusion_matrix.json").write_text(json.dumps([[2, 0, 0, 0], [0, 3, 0, 0], [0, 0, 4, 0], [0, 0, 0, 5]]))
    (run_dir / "test_confusion_matrix.json").write_text(json.dumps([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]]))

    sources = discover_sources(run_dir, tmp_path / "runs", include_val=True, include_test=True)

    assert len(sources) == 2
    assert {(s.config, s.split) for s in sources} == {("FP32", "val"), ("FP32", "test")}
    assert matrix_accuracy(sources[0].matrix) > 0.99


def test_discover_confusion_sources_prefers_compression_test(tmp_path: Path) -> None:
    run_dir = tmp_path / "fp32_100hz"
    run_dir.mkdir()
    (run_dir / "test_confusion_matrix.json").write_text(json.dumps([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

    runs = tmp_path / "runs" / "INT8"
    runs.mkdir(parents=True)
    (runs / "metrics.json").write_text(
        json.dumps({"accuracy": 0.5, "confusion_matrix": [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]})
    )

    sources = discover_sources(run_dir, tmp_path / "runs", include_val=False, include_test=True)
    by_key = {(s.config, s.split): s for s in sources}

    assert ("FP32", "test") in by_key
    assert ("INT8", "test") in by_key
    assert by_key[("INT8", "test")].accuracy == 0.5
