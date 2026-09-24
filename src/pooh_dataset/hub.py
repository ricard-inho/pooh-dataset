"""Selective download of the dataset from the Hugging Face Hub into a local folder."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path

__all__ = ["DEFAULT_REPO_ID", "default_root", "download", "remote_manifest", "META_PATTERNS"]

DEFAULT_REPO_ID = os.environ.get("POOH_REPO_ID", "r3m3c3/off_test")

#: small files always fetched: manifest, dataset card, splits, calibration, object models
META_PATTERNS = ["README.md", "manifest.json", "splits.json", "calibration/*", "models/*"]


def default_root(repo_id: str = DEFAULT_REPO_ID) -> Path:
    """``$POOH_DATA_DIR/<repo>`` or ``~/.cache/pooh_dataset/<repo>``."""
    base = os.environ.get("POOH_DATA_DIR")
    base_path = Path(base) if base else Path.home() / ".cache" / "pooh_dataset"
    return base_path / repo_id.replace("/", "--")


def _as_list(x: str | Iterable[str] | None) -> list[str] | None:
    if x is None:
        return None
    return [x] if isinstance(x, str) else list(x)


def remote_manifest(repo_id: str = DEFAULT_REPO_ID, revision: str | None = None) -> dict:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id, "manifest.json", repo_type="dataset", revision=revision)
    return json.loads(Path(path).read_text())


def download(
    repo_id: str = DEFAULT_REPO_ID,
    root: str | Path | None = None,
    trajectories: str | Iterable[str] | None = None,
    modalities: str | Iterable[str] | None = None,
    revision: str | None = None,
    max_workers: int = 8,
) -> Path:
    """Download (or resume / update) part of the dataset and return the local root.

    ``trajectories`` / ``modalities`` = None means all of them. Already-downloaded files are
    skipped, so calling this again after new trajectories were added only fetches the new
    ones. Pin ``revision`` (a commit sha or tag) for reproducible experiments.
    """
    from huggingface_hub import snapshot_download

    root = Path(root) if root is not None else default_root(repo_id)
    trajs, mods = _as_list(trajectories), _as_list(modalities)
    if trajs is None and mods is None:
        data_patterns = ["*/*/*.parquet"]
    elif trajs is None:
        data_patterns = [f"{m}/*/*.parquet" for m in mods]
    elif mods is None:
        data_patterns = [f"*/{t}/*.parquet" for t in trajs]
    else:
        data_patterns = [f"{m}/{t}/*.parquet" for m in mods for t in trajs]

    snapshot_download(
        repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=root,
        allow_patterns=META_PATTERNS + data_patterns,
        max_workers=max_workers,
    )
    return root
