# MacBook Overnight Training

Run k-fold Track B jobs locally on Apple Silicon (MPS) with job-level resume.

## Prerequisites

```bash
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
python shared/scripts/make_kfold_splits.py
python analysis/fold_qa_report.py
python experiments/scripts/run_kfold_orchestrator.py --sync-only
```

## Prevent sleep

```bash
caffeinate -dimsu bash experiments/scripts/run_overnight.sh --next --lazy
```

## Run order (model → fold → seed)

```bash
# One job at a time (default)
caffeinate -dimsu bash experiments/scripts/run_overnight.sh --next --lazy

# All CNN1D baseline jobs (15 runs)
caffeinate -dimsu bash experiments/scripts/run_overnight.sh \
  --model cnn1d --phase baseline --max-jobs 15 --lazy

# Sync manifest after manual runs
python experiments/scripts/run_kfold_orchestrator.py --sync-only
```

## Done criteria

- **Baseline:** `experiments/{model}/runs/fold{K}_seed{S}_fp32_100hz/test_metrics.json` + `best_model.pt`
- **Compression:** `experiments/compression/runs/fold{K}_seed{S}/{CONFIG}/metrics.json`

## Troubleshooting

| Issue | Fix |
|-------|-----|
| MPS OOM | Add `--lazy`; reduce batch size via config override |
| Sleep / lid close | Use `caffeinate -dimsu`; disable sleep in Energy settings |
| Job failed mid-run | Re-run `--next`; manifest skips completed jobs |
| Transformer >8 h | Run `--max-jobs 1` overnight; resume next night |

## Aggregate results

```bash
python analysis/aggregate_runs.py
```

Output: `analysis/aggregated_runs.json` (mean±std, σ_fold vs σ_seed, Wilcoxon p-values).

## Disk space

Each baseline run ~50–200 MB (history + checkpoints). Budget ~15 GB for full Track B.
