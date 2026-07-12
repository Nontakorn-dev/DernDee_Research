"""K-fold compression config registry (single source of truth)."""

from __future__ import annotations

DEFAULT_PRUNE_FINETUNE_EPOCHS = 5

# Primary Pareto / Wilcoxon grid (N=15 per config). INT4 omitted — not deployable on ESP32-C3.
PRIMARY_COMPRESSION_CONFIGS = (
    "FP32",
    "INT8",
    "Prune25",
    "Prune50",
    "Prune75",
    "INT8+Prune50",
)

# Fine-tune schedule ablation on Prune50 (Prune50 itself uses 5 epochs).
FINETUNE_ABLATION_CONFIGS = (
    "Prune50_ft15",
    "Prune50_ft50",
)

COMPRESSION_CONFIGS = PRIMARY_COMPRESSION_CONFIGS + FINETUNE_ABLATION_CONFIGS


def normalize_config_label(name: str) -> tuple[str, int | None]:
    """Return (base_config, finetune_epochs) where base strips optional ``_ft{N}`` suffix."""
    if "_ft" in name:
        base, suffix = name.rsplit("_ft", 1)
        return base, int(suffix)
    return name, None


def compression_eval_cli_args(config: str) -> list[str]:
    """Extra ``run_eval.py`` CLI args for a manifest config label."""
    _, finetune_epochs = normalize_config_label(config)
    if finetune_epochs is not None:
        return ["--prune-finetune-epochs", str(finetune_epochs)]
    return []


def finetune_epochs_for_config(config: str, *, default: int = DEFAULT_PRUNE_FINETUNE_EPOCHS) -> int | None:
    """Epoch budget recorded in metrics; ``None`` for non-pruned configs."""
    base, override = normalize_config_label(config)
    if base == "FP32" or (base.startswith("INT") and "+" not in base):
        return None
    if base.startswith("Prune") or "+" in base:
        return override if override is not None else default
    return None
