from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "shared"))

from analysis import aggregate_runs
from analysis.subject_blocked_metrics import labels_to_indices


def test_labels_to_indices_accepts_numeric_and_string_labels() -> None:
    numeric = pd.Series([0, 1, 2, 3, -1])
    strings = pd.Series(["LR", "LS", "PSw", "Sw", "UNK"])

    np.testing.assert_array_equal(labels_to_indices(numeric), np.array([0, 1, 2, 3, -1]))
    np.testing.assert_array_equal(labels_to_indices(strings), np.array([0, 1, 2, 3, -1]))


def test_collect_scores_reads_phase_f1(tmp_path: Path) -> None:
    run_dir = tmp_path / "model" / "fold0_seed42"
    run_dir.mkdir(parents=True)
    (run_dir / "test_metrics.json").write_text(
        json.dumps(
            {
                "macro_f1": 0.91,
                "phase_f1": {"LR": 0.8, "LS": 0.97, "PSw": 0.75, "Sw": 0.96},
            }
        )
    )

    scores, missing = aggregate_runs.collect_scores(
        lambda label, fold, seed: tmp_path / label / f"fold{fold}_seed{seed}",
        labels=("model",),
        n_folds=1,
        seeds=(42,),
        metric="phase_f1",
        phase="PSw",
    )

    assert missing == []
    assert scores.shape == (1, 1, 1)
    assert scores[0, 0, 0] == 0.75
