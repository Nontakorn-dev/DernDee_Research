"""Estimate deployment size (KB) for TCN compression configs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import torch

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH_ROOT / "shared"))
sys.path.insert(0, str(RESEARCH_ROOT))

from model_registry import build_model_from_config  # noqa: E402


@dataclass
class SizeEstimate:
    name: str
    params: int
    size_kb: float
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_model_from_checkpoint(path: Path) -> torch.nn.Module:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model_name = str(cfg.get("model_name", "tcn"))
    model_kwargs = dict(cfg.get("model_kwargs", {}))
    model_kwargs.setdefault("n_classes", int(cfg.get("n_classes", 4)))
    model_kwargs.setdefault("hidden", int(cfg.get("hidden", 32)))
    model = build_model_from_config(
        model_name,
        n_channels=int(cfg.get("n_channels", 12)),
        model_kwargs=model_kwargs,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def estimate_sizes(params: int) -> list[SizeEstimate]:
    fp32 = params * 4 / 1024
    int8 = params * 1 / 1024 + 2.0
    return [
        SizeEstimate("FP32", params, round(fp32, 2), "full-precision weights"),
        SizeEstimate("INT8", params, round(int8, 2), "post-training INT8"),
        SizeEstimate("Prune50", params // 2, round(fp32 * 0.5, 2), "50% channel prune"),
        SizeEstimate("INT8+Prune50", params // 2, round(int8 * 0.5, 2), "INT8 after prune"),
    ]
