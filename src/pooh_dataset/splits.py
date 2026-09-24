"""Train / val / test splits at trajectory granularity.

The source of truth is ``splits.json`` at the dataset root::

    {"version": 1,
     "train": ["trajectory_007", "trajectory_008"],
     "val":   [],
     "test":  ["trajectory_009"]}

Without it, trajectories are assigned by a stable hash of their name, so adding new
trajectories never moves an existing one to another split.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

__all__ = ["load_splits", "hash_split", "SPLIT_NAMES"]

SPLIT_NAMES = ("train", "val", "test")


def hash_split(name: str, ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)) -> str:
    h = int(hashlib.sha1(name.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < ratios[0]:
        return "train"
    return "val" if h < ratios[0] + ratios[1] else "test"


def load_splits(root: Path, trajectories: list[str]) -> dict[str, list[str]]:
    path = Path(root) / "splits.json"
    if path.exists():
        data = json.loads(path.read_text())
        splits = {k: [t for t in data.get(k, []) if t in trajectories] for k in SPLIT_NAMES}
        extra = {k: v for k, v in data.items() if k not in SPLIT_NAMES and isinstance(v, list)}
        splits.update(extra)
        return splits
    warnings.warn(
        "splits.json not found in the dataset; falling back to a hash-based trajectory split. "
        "Results are only comparable across people when the dataset ships splits.json.",
        stacklevel=3,
    )
    splits: dict[str, list[str]] = {k: [] for k in SPLIT_NAMES}
    for t in sorted(trajectories):
        splits[hash_split(t)].append(t)
    return splits
