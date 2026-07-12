from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "tcn"))

from model import build_model

from analysis.collect_pareto import parse_report_metrics
from data.dataset import make_windows_from_trial
from evaluate import classification_metrics
from gait_labels import resolve_label_column


def test_label_resolution_prefers_dual_phase_columns() -> None:
    columns = ["lt_acc_x", "phase", "phase_lt", "phase_rt"]

    assert resolve_label_column(columns, "bilateral") == "phase_lt"
    assert resolve_label_column(columns, "left") == "phase_lt"
    assert resolve_label_column(columns, "right") == "phase_rt"


def test_windowing_uses_causal_window_end_label_and_skips_unknown() -> None:
    x = np.arange(12, dtype=np.float32).reshape(6, 2)
    y = np.array([0, -1, 1, 2, -1, 3], dtype=np.int64)

    batch = make_windows_from_trial(x, y, window_size=3, stride=1)

    assert batch is not None
    assert batch.x.shape == (3, 3, 2)
    assert batch.y.tolist() == [1, 2, 3]
    np.testing.assert_array_equal(batch.x[0], x[0:3])
    np.testing.assert_array_equal(batch.x[-1], x[3:6])


def test_tcn_forward_contract() -> None:
    model = build_model(n_channels=12, n_classes=4, hidden=32)
    xb = torch.zeros(2, 50, 12)

    logits = model(xb)

    assert logits.shape == (2, 4)


def test_classification_metrics_include_phase_breakdowns() -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    y_pred = np.array([0, 1, 1, 1, 2, 3, 3, 3])

    metrics = classification_metrics(y_true, y_pred)

    assert "phase_f1" in metrics
    assert "phase_accuracy" in metrics
    assert set(metrics["phase_f1"]) == {"LR", "LS", "PSw", "Sw"}
    assert metrics["phase_accuracy"]["LR"] == 0.5
    assert metrics["phase_accuracy"]["LS"] == 1.0


def test_parse_report_metrics_prefers_confusion_matrix_accuracy(tmp_path: Path) -> None:
    report = tmp_path / "test_report.txt"
    confusion = tmp_path / "test_confusion_matrix.json"
    report.write_text(
        """
              precision    recall  f1-score   support

          LR       0.50      0.50      0.50         2
          LS       1.00      1.00      1.00         2
         PSw       0.50      0.50      0.50         2
          Sw       1.00      1.00      1.00         2

    accuracy                           0.75         8
   macro avg       0.75      0.75      0.75         8
weighted avg       0.75      0.75      0.75         8
"""
    )
    confusion.write_text(json.dumps([[1, 1, 0, 0], [0, 2, 0, 0], [0, 0, 1, 1], [0, 0, 0, 2]]))

    metrics = parse_report_metrics(report, confusion)

    assert metrics["macro_f1"] == 0.75
    assert metrics["accuracy"] == 0.75
    assert metrics["phase_accuracy"]["LR"] == 0.5
    assert metrics["phase_accuracy"]["Sw"] == 1.0
