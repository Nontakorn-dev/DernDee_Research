# Shared code

Used by all experiments under `experiments/`.

| Path | Purpose |
|------|---------|
| `gait_labels.py` | 4-phase labels, channel sets |
| `training.py` | DataLoaders, device helpers |
| `train_runner.py` | Shared fair-comparison training loop |
| `config_loader.py` | Load/validate shared training config |
| `model_registry.py` | Dynamic `build_model()` import |
| `configs/train_fair_comparison.json` | Canonical fair-comparison protocol |
| `evaluate.py` | Classification metrics |
| `eval_checkpoint.py` | Model-agnostic checkpoint evaluation |
| `data/dataset.py`, `splits.py` | Windows + subject splits |
| `paths.py` | `DATA_XY`, `experiment_runs(name)` |
| `splits/subject_split.csv` | 24 / 5 / 6 split |

Each experiment implements `experiments/<name>/model.py` with `build_model()`.
