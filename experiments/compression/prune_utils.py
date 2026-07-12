"""Structured hidden-channel pruning for causal TCN-family models."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn


def _copy_bn(dst: dict[str, torch.Tensor], src: dict[str, torch.Tensor], prefix: str, keep: torch.Tensor) -> None:
    for name in ("weight", "bias", "running_mean", "running_var"):
        dst[f"{prefix}.{name}"] = src[f"{prefix}.{name}"][keep].clone()
    dst[f"{prefix}.num_batches_tracked"] = src[f"{prefix}.num_batches_tracked"].clone()


def select_hidden_channels(model: nn.Module, keep_count: int) -> torch.Tensor:
    """Rank hidden channels by summed |weight| across all stem conv layers."""
    with torch.no_grad():
        head = model.head
        if not isinstance(head, nn.Sequential):
            raise TypeError("Expected model.head to be nn.Sequential with Linear classifier.")
        linear = head[-1]
        if not isinstance(linear, nn.Linear):
            raise TypeError("Expected final head layer to be nn.Linear.")
        importance = torch.zeros(linear.in_features)
        for block in model.stem:
            conv = block.conv
            importance += conv.weight.detach().abs().sum(dim=(1, 2)).cpu()
        keep = torch.topk(importance, k=keep_count, largest=True).indices
        return torch.sort(keep).values


def _build_pruned_model(model: nn.Module, *, new_hidden: int) -> nn.Module:
    cfg = copy.deepcopy(model)
    model_cls = type(model)
    n_channels = model.stem[0].conv.in_channels
    n_classes = model.head[-1].out_features
    kwargs: dict[str, Any] = {"n_channels": n_channels, "n_classes": n_classes, "hidden": new_hidden}
    if hasattr(model, "stem") and len(model.stem) == 4:
        # StandardTCN stores dilations internally; rebuild with same depth.
        kwargs["dilations"] = [block.conv.dilation[0] for block in model.stem]
    return model_cls(**kwargs)


def prune_hidden_channels(model: nn.Module, *, keep_ratio: float = 0.5) -> nn.Module:
    """Return a channel-pruned copy of TinyTCN or StandardTCN."""
    old_hidden = model.head[-1].in_features
    new_hidden = max(1, int(round(old_hidden * keep_ratio)))
    if new_hidden >= old_hidden:
        return copy.deepcopy(model)

    keep = select_hidden_channels(model, new_hidden)
    src = model.state_dict()
    pruned = _build_pruned_model(model, new_hidden=new_hidden)
    dst = pruned.state_dict()

    block0 = pruned.stem[0]
    dst["stem.0.conv.weight"] = src["stem.0.conv.weight"][keep].clone()
    dst["stem.0.conv.bias"] = src["stem.0.conv.bias"][keep].clone()
    _copy_bn(dst, src, "stem.0.bn", keep)
    if "stem.0.residual.weight" in src:
        dst["stem.0.residual.weight"] = src["stem.0.residual.weight"][keep].clone()
        dst["stem.0.residual.bias"] = src["stem.0.residual.bias"][keep].clone()

    for idx in range(1, len(pruned.stem)):
        dst[f"stem.{idx}.conv.weight"] = src[f"stem.{idx}.conv.weight"][keep][:, keep].clone()
        dst[f"stem.{idx}.conv.bias"] = src[f"stem.{idx}.conv.bias"][keep].clone()
        _copy_bn(dst, src, f"stem.{idx}.bn", keep)

    dst["head.2.weight"] = src["head.2.weight"][:, keep].clone()
    dst["head.2.bias"] = src["head.2.bias"].clone()
    pruned.load_state_dict(dst)
    return pruned
