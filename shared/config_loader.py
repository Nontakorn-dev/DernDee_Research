"""Load and validate shared training configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from paths import RESEARCH_ROOT

DEFAULT_CONFIG = RESEARCH_ROOT / "shared" / "configs" / "train_fair_comparison.json"

REQUIRED_TOP_KEYS = ("data", "train", "selection", "model_defaults")
REQUIRED_DATA_KEYS = (
    "channels",
    "source_hz",
    "target_hz",
    "window",
    "train_stride",
    "val_stride",
    "test_stride",
    "split_file",
)
REQUIRED_TRAIN_KEYS = ("epochs", "batch_size", "lr", "optimizer", "grad_clip", "seed", "class_weights")


def load_train_config(path: Path | str | None = None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG
    if not config_path.is_absolute():
        config_path = RESEARCH_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")

    cfg = json.loads(config_path.read_text())
    _validate_config(cfg)
    cfg["_config_path"] = str(config_path)
    return cfg


def resolve_split_file(cfg: dict[str, Any]) -> Path:
    split_file = Path(cfg["data"]["split_file"])
    if not split_file.is_absolute():
        split_file = RESEARCH_ROOT / split_file
    return split_file


def model_kwargs_for(cfg: dict[str, Any], model_name: str) -> dict[str, Any]:
    defaults = dict(cfg.get("model_defaults", {}))
    overrides = cfg.get("model_overrides", {}).get(model_name, {})
    merged = {**defaults, **overrides}
    return merged


def _validate_config(cfg: dict[str, Any]) -> None:
    for key in REQUIRED_TOP_KEYS:
        if key not in cfg:
            raise ValueError(f"Training config missing required key: {key!r}")

    for key in REQUIRED_DATA_KEYS:
        if key not in cfg["data"]:
            raise ValueError(f"Training config data section missing key: {key!r}")

    for key in REQUIRED_TRAIN_KEYS:
        if key not in cfg["train"]:
            raise ValueError(f"Training config train section missing key: {key!r}")

    if cfg["selection"].get("metric") != "val_macro_f1":
        raise ValueError("Only val_macro_f1 selection metric is supported.")
    if cfg["selection"].get("mode", "max") != "max":
        raise ValueError("Only max selection mode is supported.")
