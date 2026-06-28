"""Canonical paths for the Research workspace."""

from pathlib import Path

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
DATA_XY = RESEARCH_ROOT / "dataset" / "Xy"
SHARED_SPLITS = RESEARCH_ROOT / "shared" / "splits"
PAPER_DIR = RESEARCH_ROOT / "paper"
PAPER_FIGURES = PAPER_DIR / "figures"
EXPERIMENTS = RESEARCH_ROOT / "experiments"
TINYTCN_RUNS = EXPERIMENTS / "tinytcn" / "runs"
COMPRESSION_RESULTS = EXPERIMENTS / "compression"


def experiment_runs(name: str) -> Path:
    return EXPERIMENTS / name / "runs"
