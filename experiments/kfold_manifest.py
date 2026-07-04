"""Job manifest for k-fold × seed training and compression runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = RESEARCH_ROOT / "experiments" / "kfold_run_manifest.json"

BASELINE_MODELS = ("cnn1d", "tcn", "tinytcn", "lstm_gru", "transformer")
COMPRESSION_CONFIGS = ("FP32", "INT8", "Prune50", "INT8+Prune50")
DEFAULT_SEEDS = (42, 123, 456)
DEFAULT_FOLDS = tuple(range(5))

JobPhase = Literal["baseline", "compression"]
JobStatus = Literal["pending", "running", "done", "failed"]


@dataclass
class KFoldJob:
    model: str
    fold: int
    seed: int
    phase: JobPhase
    config: str | None = None
    status: JobStatus = "pending"
    run_dir: str | None = None
    finished_at: str | None = None
    error: str | None = None


def baseline_run_dir(model: str, fold: int, seed: int) -> Path:
    return RESEARCH_ROOT / "experiments" / model / "runs" / f"fold{fold}_seed{seed}_fp32_100hz"


def compression_run_dir(fold: int, seed: int, config: str) -> Path:
    return RESEARCH_ROOT / "experiments" / "compression" / "runs" / f"fold{fold}_seed{seed}" / config


def split_file_for_fold(fold: int) -> Path:
    return RESEARCH_ROOT / "shared" / "splits" / "folds" / f"fold{fold}.csv"


def baseline_done(run_dir: Path) -> bool:
    return (run_dir / "test_metrics.json").exists() and (run_dir / "best_model.pt").exists()


def compression_done(run_dir: Path) -> bool:
    return (run_dir / "metrics.json").exists()


def infer_job_status(job: KFoldJob) -> JobStatus:
    if job.phase == "baseline":
        run_dir = baseline_run_dir(job.model, job.fold, job.seed)
    else:
        assert job.config is not None
        run_dir = compression_run_dir(job.fold, job.seed, job.config)

    job.run_dir = str(run_dir)
    if job.phase == "baseline" and baseline_done(run_dir):
        return "done"
    if job.phase == "compression" and compression_done(run_dir):
        return "done"
    return job.status if job.status in ("running", "failed") else "pending"


def generate_jobs(
    *,
    models: tuple[str, ...] = BASELINE_MODELS,
    folds: tuple[int, ...] = DEFAULT_FOLDS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    include_compression: bool = True,
) -> list[KFoldJob]:
    jobs: list[KFoldJob] = []
    for model in models:
        for fold in folds:
            for seed in seeds:
                jobs.append(KFoldJob(model=model, fold=fold, seed=seed, phase="baseline"))

    if include_compression:
        for fold in folds:
            for seed in seeds:
                for config in COMPRESSION_CONFIGS:
                    jobs.append(
                        KFoldJob(
                            model="tinytcn",
                            fold=fold,
                            seed=seed,
                            phase="compression",
                            config=config,
                        )
                    )
    return jobs


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"jobs": []}


def save_manifest(payload: dict, path: Path = MANIFEST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def sync_manifest(
    path: Path = MANIFEST_PATH,
    *,
    models: tuple[str, ...] = BASELINE_MODELS,
    folds: tuple[int, ...] = DEFAULT_FOLDS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    include_compression: bool = True,
    reset_running: bool = False,
) -> dict:
    existing = {job_key(j): j for j in load_manifest(path).get("jobs", [])}
    jobs = generate_jobs(models=models, folds=folds, seeds=seeds, include_compression=include_compression)

    synced: list[dict] = []
    for job in jobs:
        key = job_key(asdict(job))
        prior = existing.get(key, {})
        merged = KFoldJob(
            model=job.model,
            fold=job.fold,
            seed=job.seed,
            phase=job.phase,
            config=job.config,
            status=prior.get("status", "pending"),
            run_dir=prior.get("run_dir"),
            finished_at=prior.get("finished_at"),
            error=prior.get("error"),
        )
        if reset_running and merged.status == "running":
            merged.status = "pending"
            merged.error = "reset_running"
            merged.finished_at = None
        merged.status = infer_job_status(merged)
        if merged.status == "done" and not merged.finished_at:
            merged.finished_at = datetime.now(timezone.utc).isoformat()
        synced.append(asdict(merged))

    payload = {"updated_at": datetime.now(timezone.utc).isoformat(), "jobs": synced}
    save_manifest(payload, path)
    return payload


def job_key(job: dict) -> str:
    if job.get("phase") == "compression":
        return f"compression:fold{job['fold']}:seed{job['seed']}:{job['config']}"
    return f"baseline:{job['model']}:fold{job['fold']}:seed{job['seed']}"


def next_pending_job(
    payload: dict,
    *,
    model: str | None = None,
    phase: JobPhase | None = None,
) -> dict | None:
    """Model-first order: model → fold → seed; compression after baseline per fold×seed."""
    jobs = payload.get("jobs", [])
    for job in jobs:
        if job.get("status") != "pending":
            continue
        if model and job.get("model") != model and job.get("phase") == "baseline":
            continue
        if phase and job.get("phase") != phase:
            continue
        if job.get("phase") == "compression":
            baseline = baseline_run_dir("tinytcn", job["fold"], job["seed"])
            if not baseline_done(baseline):
                continue
        return job
    return None


def mark_job(path: Path, job: dict, *, status: JobStatus, error: str | None = None) -> None:
    payload = load_manifest(path)
    key = job_key(job)
    for entry in payload.get("jobs", []):
        if job_key(entry) == key:
            entry["status"] = status
            entry["error"] = error
            if status == "done":
                entry["finished_at"] = datetime.now(timezone.utc).isoformat()
            break
    save_manifest(payload, path)
