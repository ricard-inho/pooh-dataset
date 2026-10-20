"""Entry point: a local copy of the dataset."""

from __future__ import annotations

import json
from collections.abc import Iterable
from functools import cached_property
from pathlib import Path

from .hub import (
    DEFAULT_REPO_ID,
    MANIFEST,
    SIM_GROUPS,
    SIM_MANIFEST,
    default_root,
    domain_parts,
    download,
)
from .objects import ObjectModel, load_object_models
from .splits import load_splits, sim_run
from .trajectory import Trajectory

__all__ = ["PoohDataset"]

NOT_TRAJECTORY_DIRS = {"calibration", "metadata", "models"}


class PoohDataset:
    """A local dataset folder with the Hub layout: ``<modality>/<trajectory>/*.parquet`` for
    real recordings, ``<sim group>_<modality>/<trajectory>/*.parquet`` for simulated ones.

    >>> ds = PoohDataset.download(modalities=["mocap", "realsense_color", "realsense_depth"])
    >>> traj = ds["trajectory_007"]
    >>> rgb = traj.frame("realsense_color", 0)
    >>> sim = PoohDataset.download(domain="sim_3d", modalities=["rgb", "mocap"])
    >>> sim.split("all", domain="synthetic")
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
        domain: str = "real",
        split: str | None = None,
        task: str | None = None,
    ) -> PoohDataset:
        root = download(repo_id, root, trajectories, modalities, revision, domain=domain, split=split, task=task)
        return cls(root, repo_id)

    def __repr__(self) -> str:
        return f"PoohDataset({str(self.root)!r}, trajectories={len(self.trajectory_names)})"

    @cached_property
    def manifest(self) -> dict:
        """``manifest.json``: the real recordings (or, in a local conversion of the simulated
        data, the simulated ones)."""
        path = self.root / MANIFEST
        return json.loads(path.read_text()) if path.exists() else {}

    @cached_property
    def sim_manifest(self) -> dict:
        """``sim_manifest.json``: the simulated recordings on the Hub."""
        path = self.root / SIM_MANIFEST
        return json.loads(path.read_text()) if path.exists() else {}

    @cached_property
    def _entries(self) -> dict[str, dict]:
        """trajectory -> info (domain, group, counts, ...) from both manifests."""
        out = {}
        real_info = self.manifest.get("trajectory_info", {})
        hub_splits = self.manifest.get("splits", {})
        for name, entry in self.manifest.get("trajectories", {}).items():
            if isinstance(entry.get("counts"), dict):  # local conversion of simulated data
                out[name] = _sim_info(entry)
                continue
            info = dict(real_info.get(name, {}))
            info.setdefault("domain", "real")
            info["counts"] = entry
            if name in hub_splits:
                info["hub_split"] = hub_splits[name]
            out[name] = info
        for name, entry in self.sim_manifest.get("trajectories", {}).items():
            out[name] = _sim_info(entry)
        return out

    @cached_property
    def trajectory_names(self) -> list[str]:
        """Trajectories listed in the manifests, or found on disk if there is none."""
        names = set(self._entries)
        if not names:
            names = {t.name for m in self.root.iterdir()
                     if m.is_dir() and m.name not in NOT_TRAJECTORY_DIRS for t in m.iterdir() if t.is_dir()}
        return sorted(names)

    @property
    def local_trajectory_names(self) -> list[str]:
        """Trajectories with at least one modality downloaded."""
        return [t for t in self.trajectory_names if self[t].modalities]

    def trajectory_info(self, name: str) -> dict:
        """Free-form metadata: ``domain`` (real / synthetic), ``counts`` per modality,
        ``hub_split`` (the trajectory's split on the Hub viewer, if known); simulated
        trajectories add ``group`` (sim_3d / sim_zerog), ``scenario``, ``task`` (approach /
        orbit in sim_3d, docking / inspection in sim_zerog), ``run`` (the recording run: its
        episodes share a split), ``duration_s`` and the final goal errors."""
        info = dict(self._entries.get(name, {}))
        if not info:
            info = {"domain": "synthetic", "group": _disk_group(self.root, name)} \
                if _disk_group(self.root, name) else {"domain": "real"}
            info["counts"] = {}
        return info

    def __getitem__(self, name: str) -> Trajectory:
        if name not in self._trajectories:
            if name not in self.trajectory_names:
                raise KeyError(f"unknown trajectory {name!r}; have {self.trajectory_names}")
            info = self.trajectory_info(name)
            prefix = f"{info['group']}_" if info.get("group") else ""
            self._trajectories[name] = Trajectory(self.root, name, info, prefix=prefix)
        return self._trajectories[name]

    def __iter__(self):
        return (self[t] for t in self.trajectory_names)

    def __len__(self) -> int:
        return len(self.trajectory_names)

    @cached_property
    def splits(self) -> dict[str, list[str]]:
        """Split name -> trajectories, from ``splits.json`` (see :mod:`pooh_dataset.splits`)."""
        real = [t for t in self.trajectory_names if self.trajectory_info(t)["domain"] == "real"]
        return load_splits(self.root, self.trajectory_names, unlisted=real)

    def in_domain(self, name: str, domain: str | None) -> bool:
        """Whether trajectory ``name`` belongs to ``domain`` (see :data:`pooh_dataset.hub.DOMAINS`)."""
        if domain is None:
            return True
        real, groups = domain_parts(domain)
        info = self.trajectory_info(name)
        if info["domain"] == "real":
            return real
        if domain in ("synthetic", "all"):
            return True
        return info.get("group") in groups  # one simulation group

    def split(self, name: str | Iterable[str], domain: str | None = None, task: str | None = None) -> list[str]:
        """Trajectory names of a split ("train", "val", "test", "all" or an explicit list),
        optionally restricted to a domain ("real", "synthetic", "sim_3d", "sim_zerog") and a
        simulated task ("approach", "orbit", "docking", "inspection")."""
        if isinstance(name, str):
            names = self.trajectory_names if name == "all" else self.splits.get(name)
            if names is None:
                raise KeyError(f"unknown split {name!r}; have {sorted(self.splits)}")
        else:
            names = list(name)
        return [t for t in names if self.in_domain(t, domain)
                and (task is None or self.trajectory_info(t).get("task") == task)]

    def tasks(self, domain: str = "synthetic") -> dict[str, list[str]]:
        """Task (``<group>/<task>``, e.g. ``sim_zerog/docking``) -> trajectories."""
        out: dict[str, list[str]] = {}
        for t in self.split("all", domain):
            info = self.trajectory_info(t)
            if info.get("task") is not None:
                out.setdefault(f"{info.get('group') or info['domain']}/{info['task']}", []).append(t)
        return out

    @cached_property
    def hub_splits(self) -> dict[str, list[str]]:
        """Splits of the Hub viewer: groups of real trajectories (``--split`` of the real
        converter, default ``train``) and simulation scenarios (e.g. ``iss_earth``). They
        organise the data; :meth:`split` gives the train / val / test splits for training."""
        out: dict[str, list[str]] = {}
        for t in self.trajectory_names:
            s = self.trajectory_info(t).get("hub_split")
            if s is not None:
                out.setdefault(s, []).append(t)
        return out

    def scenarios(self, domain: str = "synthetic") -> dict[str, list[str]]:
        """Simulation scenario (e.g. ``orbital/iss/earth``) -> trajectories."""
        out: dict[str, list[str]] = {}
        for t in self.split("all", domain):
            s = self.trajectory_info(t).get("scenario")
            if s is not None:
                out.setdefault(s, []).append(t)
        return out

    @cached_property
    def objects(self) -> dict[str, ObjectModel]:
        """Object models from ``models/objects.yaml`` (empty until the CAD is uploaded)."""
        return load_object_models(self.root / "models")


def _sim_info(entry: dict) -> dict:
    info = {k: v for k, v in entry.items() if k != "source_dir"}
    info["domain"] = "synthetic"
    info.setdefault("group", None)
    if "source_dir" in entry:
        run = sim_run(entry["source_dir"])
        info.setdefault("task", run["task"])
        info["run"] = run["run"]
    if "scenario" in entry:
        # Hub split: the scenario without the group (orbital/iss/earth -> iss_earth)
        info["hub_split"] = "_".join(entry["scenario"].split("/")[1:])
    return info


def _disk_group(root: Path, name: str) -> str | None:
    for g in SIM_GROUPS:
        if any(p.is_dir() for p in root.glob(f"{g}_*/{name}")):
            return g
    return None
