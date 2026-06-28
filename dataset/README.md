# Dataset symlink

`Xy` points to preprocessed trial CSVs:

```
Xy → ~/Desktop/Dataset/processed/Xy
```

Columns: `time` + 12 IMU channels + `phase_lt` + `phase_rt`.

Expected layout:

```
dataset/Xy/
├── S001/
│   └── S*.csv
├── S002/
│   └── S*.csv
└── ...
```

Run training and analysis scripts from the **Research repo root**. Code resolves this symlink through `shared/paths.py`.

Useful checks:

```bash
python shared/scripts/check_xy_labels.py --channels bilateral
python shared/scripts/check_xy_labels.py --channels right
```

If labels are missing, regenerate or upgrade them:

```bash
python shared/preprocess_gaitprint.py --input-dir NONAN_Dataset --output-dir dataset/Xy
python shared/scripts/upgrade_xy_labels.py --xy-root dataset/Xy
```
