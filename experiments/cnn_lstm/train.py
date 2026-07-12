#!/usr/bin/env python3
"""Train CNN-LSTM hybrid baseline with the shared fair-comparison protocol."""

from __future__ import annotations

import sys
from pathlib import Path

SHARED = Path(__file__).resolve().parents[2] / "shared"
sys.path.insert(0, str(SHARED))

from train_runner import main_for_model  # noqa: E402

if __name__ == "__main__":
    main_for_model("cnn_lstm", description=__doc__)
