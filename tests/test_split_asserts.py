"""Tests for split integrity assertions."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))

from data.splits import SplitPolicy, assert_files_match_split, subjects_from_files


def _files(subjects: list[str]) -> list[Path]:
    return [Path(f"dataset/Xy/{s}/S001.csv") for s in subjects]


def test_subjects_from_files() -> None:
    files = _files(["S001", "S002"])
    assert subjects_from_files(files) == {"S001", "S002"}


def test_train_only_passes() -> None:
    split_map = {
        "train": ["S001", "S002"],
        "val": ["S003"],
        "test": ["S004"],
    }
    assert_files_match_split(
        _files(["S001", "S002"]), split_map, policy=SplitPolicy.TRAIN_ONLY, context="test"
    )


def test_train_only_rejects_test_subject() -> None:
    split_map = {
        "train": ["S001", "S002"],
        "val": ["S003"],
        "test": ["S004"],
    }
    with pytest.raises(AssertionError, match="extra subjects"):
        assert_files_match_split(
            _files(["S001", "S004"]), split_map, policy=SplitPolicy.TRAIN_ONLY, context="test"
        )


def test_train_only_rejects_val_subject() -> None:
    split_map = {
        "train": ["S001"],
        "val": ["S003"],
        "test": ["S004"],
    }
    with pytest.raises(AssertionError, match="extra subjects"):
        assert_files_match_split(
            _files(["S001", "S003"]), split_map, policy=SplitPolicy.TRAIN_ONLY, context="finetune"
        )


def test_test_only_rejects_train_subject() -> None:
    split_map = {
        "train": ["S001"],
        "val": ["S003"],
        "test": ["S004"],
    }
    with pytest.raises(AssertionError, match="extra subjects"):
        assert_files_match_split(
            _files(["S004", "S001"]), split_map, policy=SplitPolicy.TEST_ONLY, context="eval"
        )
