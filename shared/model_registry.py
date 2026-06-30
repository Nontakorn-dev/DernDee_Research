"""Dynamic import of experiment model builders."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from paths import EXPERIMENTS

SUPPORTED_MODELS = ("tinytcn", "cnn1d", "lstm_gru", "tcn", "transformer")


def model_module_path(model_name: str) -> Path:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model {model_name!r}. Supported: {', '.join(SUPPORTED_MODELS)}"
        )
    return EXPERIMENTS / model_name / "model.py"


def load_build_model(model_name: str):
    module_path = model_module_path(model_name)
    if not module_path.exists():
        raise FileNotFoundError(f"Model module not found: {module_path}")

    module_name = f"experiments.{model_name}.model"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load model module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    build_model = getattr(module, "build_model", None)
    if build_model is None:
        raise AttributeError(f"{module_path} does not define build_model()")
    return build_model


def build_model_from_config(
    model_name: str,
    *,
    n_channels: int,
    model_kwargs: dict[str, Any],
) -> Any:
    build_model = load_build_model(model_name)
    n_classes = int(model_kwargs.get("n_classes", 4))
    hidden = int(model_kwargs.get("hidden", 32))
    extra = {k: v for k, v in model_kwargs.items() if k not in {"n_classes", "hidden"}}
    return build_model(n_channels=n_channels, n_classes=n_classes, hidden=hidden, **extra)
