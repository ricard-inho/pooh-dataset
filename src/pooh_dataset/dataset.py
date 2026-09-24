"""Entry point: a local copy of the dataset."""

from __future__ import annotations

import json
from collections.abc import Iterable
from functools import cached_property
from pathlib import Path

from .hub import DEFAULT_REPO_ID, default_root, download
from .objects import ObjectModel, load_object_models
from .splits import load_splits
from .trajectory import Trajectory

__all__ = ["PoohDataset"]


class PoohDataset:
    """A local dataset folder with the Hub layout ``<modality>/<trajectory>/*.parquet``.

    >>> ds = PoohDataset.download(modalities=["mocap", "realsense_color", "realsense_depth"])
    >>> traj = ds["trajectory_007"]
    >>> rgb = traj.frame("realsense_color", 0)
    """

    def __init__(self, root: str | Path | None = None, repo_id: str = DEFAULT_REPO_ID):
        self.repo_id = repo_id
        self.root = Path(root) if root is not None else default_root(repo_id)
        if not self.root.exists():
            raise FileNotFoundError(
                f"{self.root} does not exist; call PoohDataset.download(...) or `pooh download`"
            )
        self._trajectories: dict[str, Trajectory] = {}

    @classmethod
    def download(
        cls,
        repo_id: str = DEFAULT_REPO_ID,
        root: str | Path | None = None,
        trajectories: str | Iterable[str] | None = None,
        modalities: str | Iterable[str] | None = None,
        revision: str | None = None,
    ) -> PoohDataset:
        root = download(repo_id, root, trajectories, modalities, revision)
        return cls(root, repo_id)

    def __repr__(self) -> str:
        return f"PoohDataset({str(self.root)!r}, trajectories={len(self.trajectory_names)})"

    @cached_property
    def manifest(self) -> dict:
        path = self.root / "manifest.json"
        return json.loads(path.read_text()) if path.exists() else {}

    @cached_property
    def trajectory_names(self) -> list[str]:
        """Trajectories listed in the manifest, or found on disk if there is no manifest."""
        names = set(self.manifest.get("trajectories", {}))
        if not names:
            names = {t.name for m in self.root.iterdir() if m.is_dir() for t in m.iterdir() if t.is_dir()}
            names -= {"calibration", "models"}
        return sorted(names)

    @property
    def local_trajectory_names(self) -> list[str]:
        """Trajectories with at least one modality downloaded."""
        return [t for t in self.trajectory_names if self[t].modalities]

    def trajectory_info(self, name: str) -> dict:
        """Free-form metadata: ``domain`` (real / synthetic), ``task``, notes, ..."""
        info = dict(self.manifest.get("trajectory_info", {}).get(name, {}))
        info.setdefault("domain", "real")
        info["counts"] = self.manifest.get("trajectories", {}).get(name, {})
        return info

    def __getitem__(self, name: str) -> Trajectory:
        if name not in self._trajectories:
            if name not in self.trajectory_names:
                raise KeyError(f"unknown trajectory {name!r}; have {self.trajectory_names}")
            self._trajectories[name] = Trajectory(self.root, name, self.trajectory_info(name))
        return self._trajectories[name]

    def __iter__(self):
        return (self[t] for t in self.trajectory_names)

    def __len__(self) -> int:
        return len(self.trajectory_names)

    @cached_property
    def splits(self) -> dict[str, list[str]]:
        return load_splits(self.root, self.trajectory_names)

    def split(self, name: str | Iterable[str], domain: str | None = None) -> list[str]:
        """Trajectory names of a split ("train", "val", "test", "all" or an explicit list),
        optionally restricted to a domain ("real" / "synthetic")."""
        if isinstance(name, str):
            names = self.trajectory_names if name == "all" else self.splits.get(name)
            if names is None:
                raise KeyError(f"unknown split {name!r}; have {sorted(self.splits)}")
        else:
            names = list(name)
        if domain is not None:
            names = [t for t in names if self.trajectory_info(t)["domain"] == domain]
        return names

    @cached_property
    def objects(self) -> dict[str, ObjectModel]:
        """Object models from ``models/objects.yaml`` (empty until the CAD is uploaded)."""
        return load_object_models(self.root / "models")
