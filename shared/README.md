# Shared code

Used by all experiments under `experiments/`.

| Path | Purpose |
|------|---------|
| `gait_labels.py` | 4-phase labels, channel sets |
| `training.py` | DataLoaders, device helpers |
| `evaluate.py` | Classification metrics |
| `data/dataset.py`, `splits.py` | Windows + subject splits |
| `paths.py` | `DATA_XY`, `experiment_runs(name)` |
| `splits/subject_split.csv` | 24 / 5 / 6 split |

Each experiment implements `experiments/<name>/model.py` with `build_model()`.
