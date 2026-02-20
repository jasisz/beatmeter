"""Lightweight experiment tracking — zero external dependencies.

Two responsibilities:
1. Capture metadata (git commit, timestamp, args) at train/eval time
2. Append records to data/experiments.jsonl (one JSON per line)
"""

import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

EXPERIMENTS_LOG = Path("data/experiments.jsonl")


def get_git_info() -> dict:
    """Return {commit, branch, dirty}. Graceful on non-git dirs."""
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .strip()
            .decode()
        )
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .strip()
            .decode()
        )
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"commit": commit, "branch": branch, "dirty": dirty}
    except Exception:
        return {"commit": "unknown", "branch": "unknown", "dirty": False}


def checkpoint_sha256(path) -> str:
    """First 16 hex chars of SHA-256 of a checkpoint file."""
    p = Path(path)
    if not p.exists():
        return "missing"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def make_experiment_record(
    type: str,  # "train", "eval", "grid_search"
    model: str,  # "meter_net", "pipeline"
    params: dict = None,  # hyperparameters
    results: dict = None,
    checkpoint: str = None,
    extra: dict = None,
) -> dict:
    """Build a structured experiment record."""
    record = {
        "type": type,
        "model": model,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git": get_git_info(),
        "command": " ".join(sys.argv),
    }
    if params:
        record["params"] = params
    if results:
        record["results"] = results
    if checkpoint:
        record["checkpoint"] = checkpoint
        record["checkpoint_sha256"] = checkpoint_sha256(checkpoint)
    if extra:
        record.update(extra)
    return record


def log_experiment(record: dict) -> None:
    """Append one JSON record to data/experiments.jsonl."""
    EXPERIMENTS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENTS_LOG, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def enrich_checkpoint(
    ckpt: dict, args=None, train_size=0, val_size=0, test_size=0
) -> dict:
    """Add experiment metadata to a checkpoint dict before torch.save()."""
    ckpt["experiment"] = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git": get_git_info(),
        "command": " ".join(sys.argv),
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
    }
    if args:
        ckpt["experiment"]["args"] = vars(args)
    return ckpt
