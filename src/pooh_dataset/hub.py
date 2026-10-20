"""Selective download of the dataset from the Hugging Face Hub into a local folder.

Hub layout (one repo for the real and the simulated recordings)::

    <modality>/<trajectory>/*.parquet               real: events, firefly, realsense_color, ...
    sim_<group>_<modality>/<trajectory>/*.parquet   simulated, group sim_3d or sim_zerog
    manifest.json, sim_manifest.json                trajectories of each domain
    calibration/<trajectory>.yaml, calibration/rig.yaml
    metadata/<trajectory>.yaml                      simulated only (scenario, rates, ...)
    models/                                         object models
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path

__all__ = [
    "DEFAULT_REPO_ID",
    "DOMAINS",
    "META_PATTERNS",
    "REAL_MODALITIES",
    "SIM_GROUPS",
    "SIM_MODALITIES",
    "default_root",
    "download",
    "remote_manifest",
    "remote_sim_manifest",
]

DEFAULT_REPO_ID = os.environ.get("POOH_REPO_ID", "r3m3c3/off_test")

REAL_MODALITIES = ("events", "firefly", "realsense_color", "realsense_depth", "lidar", "imu", "mocap",
                   "robot_messages")
#: simulated recordings: Hub folder ``<group>_<modality>``
SIM_GROUPS = ("sim_3d", "sim_zerog")
SIM_MODALITIES = ("rgb", "depth", "semantic", "events", "lidar", "mocap", "imu", "joint_states", "steps")
#: values of ``domain``: a domain, both, or one simulation group
DOMAINS = ("real", "synthetic", "all") + SIM_GROUPS

MANIFEST = "manifest.json"
SIM_MANIFEST = "sim_manifest.json"
#: small files always fetched: manifests, dataset card, splits, object models
META_PATTERNS = ["README.md", MANIFEST, SIM_MANIFEST, "splits.json", "models/*"]


def default_root(repo_id: str = DEFAULT_REPO_ID) -> Path:
    """``$POOH_DATA_DIR/<repo>`` or ``~/.cache/pooh_dataset/<repo>``."""
    base = os.environ.get("POOH_DATA_DIR")
    base_path = Path(base) if base else Path.home() / ".cache" / "pooh_dataset"
    return base_path / repo_id.replace("/", "--")


def _as_list(x: str | Iterable[str] | None) -> list[str] | None:
    if x is None:
        return None
    return [x] if isinstance(x, str) else list(x)


def domain_parts(domain: str) -> tuple[bool, list[str]]:
    """``domain`` -> (includes the real data, simulation groups it includes)."""
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r}; use one of {DOMAINS}")
    real = domain in ("real", "all")
    groups = list(SIM_GROUPS) if domain in ("synthetic", "all") else [g for g in SIM_GROUPS if g == domain]
    return real, groups


def remote_manifest(repo_id: str = DEFAULT_REPO_ID, revision: str | None = None) -> dict:
    """``manifest.json`` of the real recordings."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id, MANIFEST, repo_type="dataset", revision=revision)
    return json.loads(Path(path).read_text())


def remote_sim_manifest(repo_id: str = DEFAULT_REPO_ID, revision: str | None = None) -> dict:
    """``sim_manifest.json``: simulated trajectory -> group, scenario, counts, goal errors."""
    from huggingface_hub import hf_hub_download

    try:
        from huggingface_hub.errors import EntryNotFoundError
    except ImportError:  # huggingface_hub < 0.24
        from huggingface_hub.utils import EntryNotFoundError
    try:
        path = hf_hub_download(repo_id, SIM_MANIFEST, repo_type="dataset", revision=revision)
    except EntryNotFoundError:
        return {"trajectories": {}}
    return json.loads(Path(path).read_text())


def download(
    repo_id: str = DEFAULT_REPO_ID,
    root: str | Path | None = None,
    trajectories: str | Iterable[str] | None = None,
    modalities: str | Iterable[str] | None = None,
    revision: str | None = None,
    max_workers: int = 8,
    domain: str = "real",
    split: str | None = None,
    task: str | None = None,
) -> Path:
    """Download (or resume / update) part of the dataset and return the local root.

    ``trajectories`` / ``modalities`` = None means all of them. Without ``trajectories``,
    ``domain`` picks the recordings: ``"real"`` (default), ``"synthetic"`` (all simulated),
    ``"sim_3d"`` / ``"sim_zerog"`` (one simulation group) or ``"all"``. Explicit
    ``trajectories`` are fetched whatever their domain. Modality names are the folder names
    of the domain: ``realsense_color``, ... for real data, ``rgb``, ``depth``, ``semantic``, ...
    for simulated data (see :data:`REAL_MODALITIES`, :data:`SIM_MODALITIES`).

    ``split`` ("train", "val", "test", from ``splits.json``) and ``task`` ("approach",
    "orbit", "docking", "inspection") narrow the trajectories of ``domain`` further, e.g.
    ``download(domain="sim_zerog", task="docking", split="test")``.

    Already-downloaded files are skipped, so calling this again after new trajectories were
    added only fetches the new ones. Pin ``revision`` (a commit sha or tag) for reproducible
    experiments.
    """
    from huggingface_hub import snapshot_download

    root = Path(root) if root is not None else default_root(repo_id)
    trajs, mods = _as_list(trajectories), _as_list(modalities)
    real, groups = domain_parts(domain)
    if trajs is None and (split is not None or task is not None):
        # Pick the trajectories from the manifests and splits.json, then fetch them by name.
        from .dataset import PoohDataset

        snapshot_download(repo_id, repo_type="dataset", revision=revision, local_dir=root,
                          allow_patterns=META_PATTERNS, max_workers=max_workers)
        trajs = PoohDataset(root, repo_id).split(split or "all", domain, task=task)
        if not trajs:
            raise ValueError(f"no trajectory with domain={domain!r} split={split!r} task={task!r}")

    if trajs is not None:
        # Trajectory names are unique across domains: match both layouts.
        names = trajs
        if mods is None:
            data = [f"*/{t}/*.parquet" for t in trajs]
        else:
            data = [p for m in mods for t in trajs for p in (f"{m}/{t}/*.parquet", f"sim_*_{m}/{t}/*.parquet")]
        sim = True
    else:
        names = []
        data = [f"{m}/*/*.parquet" for m in (mods or REAL_MODALITIES)] if real else []
        data += [f"{g}_{m}/*/*.parquet" for g in groups for m in (mods or ["*"])]
        sim = bool(groups)
        if real:  # real calibration by name: the sim calibration files are many
            from huggingface_hub import hf_hub_download

            manifest = hf_hub_download(repo_id, MANIFEST, repo_type="dataset", revision=revision,
                                       local_dir=root)
            names = list(json.loads(Path(manifest).read_text()).get("trajectories", {}))

    side = ["calibration/rig.yaml", "calibration/clicks_*"]
    side += [f"{d}/{t}.yaml" for t in names for d in ("calibration", "metadata")]
    if sim and trajs is None:
        side += ["calibration/*", "metadata/*"]
    snapshot_download(
        repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=root,
        allow_patterns=META_PATTERNS + side + data,
        max_workers=max_workers,
    )
    return root
