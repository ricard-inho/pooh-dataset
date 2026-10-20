"""Train / val / test splits at trajectory granularity.

The source of truth is ``splits.json`` at the dataset root::

    {"version": 2,
     "train": ["trajectory_hera_000001", ...],
     "val":   [],
     "test":  ["trajectory_hera_000007", ...],
     "summary": {"sim_3d/approach": {"train": 400, "test": 100}, ...}}

Real trajectories that ``splits.json`` does not list are assigned by a stable hash of their
name, so adding new trajectories never moves an existing one to another split. Simulated
trajectories it does not list are in no split until :func:`update_sim_splits` adds them.

The simulated part is written by :func:`update_sim_splits` (``pooh splits update``, also run
by the sim publisher): append-only, so a trajectory never changes split. Its unit is the
recording *run* (one simulator process, a few episodes of the same scenario with the same
seed): all episodes of a run share a split, so no near-duplicate of an eval episode is
trained on. Runs are balanced per scenario config (``zerog_cubesat_arm_wall_docking``,
``orbital_iss_earth_orbit``, ...), so every group, task, target and background gets eval
runs: a new run goes to ``test`` when that keeps the config's share of test runs closest
to ``eval_fraction``.

This module only uses the standard library: the sim publisher loads it by path.
"""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from collections.abc import Iterable
from pathlib import Path

__all__ = ["EVAL_FRACTION", "SPLIT_NAMES", "hash_split", "load_splits", "sim_run", "update_sim_splits"]

SPLIT_NAMES = ("train", "val", "test")
#: share of the recording runs of each simulated scenario config that go to ``test``
EVAL_FRACTION = 0.2
VERSION = 2

# <config>_<scenario index>_<launch time>_<process index>, e.g.
# zerog_cubesat_arm_wall_docking_00012_20261017T101530_002
_RUN_TAG = re.compile(r"^(?P<config>.+)_\d{5}_(?P<time>\d{8}T\d{6})_\d{3}$")


def hash_split(name: str, ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)) -> str:
    h = int(hashlib.sha1(name.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < ratios[0]:
        return "train"
    return "val" if h < ratios[0] + ratios[1] else "test"


def sim_run(source_dir: str) -> dict:
    """Recording run of a simulated trajectory, from its raw directory name
    (``<run>_env<k>_ep<j>``): ``run``, ``config`` (scenario config without the index),
    ``task`` (its last word: approach / orbit in sim_3d, docking / inspection in sim_zerog)
    and ``time`` (launch time, sorts runs). Unknown names are a run of their own, task None."""
    run = source_dir.rsplit("_env", 1)[0]
    m = _RUN_TAG.match(run)
    if m is None:
        return {"run": run, "config": run, "task": None, "time": ""}
    config = m["config"]
    return {"run": run, "config": config, "task": config.rsplit("_", 1)[-1], "time": m["time"]}


def load_splits(root: Path, trajectories: list[str], unlisted: Iterable[str] = ()) -> dict[str, list[str]]:
    """Splits of ``trajectories`` from ``splits.json``; those of ``unlisted`` (the real
    trajectories) that it does not list, or all of them without the file, get
    :func:`hash_split`."""
    path = Path(root) / "splits.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    known = set(trajectories)
    splits = {k: [t for t in data.get(k, []) if t in known] for k in SPLIT_NAMES}
    splits.update({k: v for k, v in data.items() if k not in SPLIT_NAMES and k != "summary" and isinstance(v, list)})
    listed = {t for v in splits.values() for t in v}
    hashed = sorted(t for t in unlisted if t in known and t not in listed)
    if hashed:
        warnings.warn(
            f"{len(hashed)} real trajectories have no official split in splits.json; using a hash-based "
            "split for them. Results are only comparable across people with the same dataset revision.",
            stacklevel=3,
        )
    for t in hashed:
        splits[hash_split(t)].append(t)
    return splits


def update_sim_splits(splits: dict | None, sim_manifest: dict, eval_fraction: float = EVAL_FRACTION) -> dict:
    """``splits.json`` content with the simulated trajectories of ``sim_manifest`` that it
    does not list yet added (see the module docstring). Listed trajectories, real or
    simulated, never move. Returns a new dict; ``splits`` may be None (no file yet)."""
    old = dict(splits or {})
    assigned = {t: k for k in SPLIT_NAMES for t in old.get(k, [])}
    entries = sim_manifest.get("trajectories", {})
    runs: dict[str, dict] = {}
    for t, e in entries.items():
        r = sim_run(e.get("source_dir", t))
        run = runs.setdefault(r["run"], {**r, "group": e.get("group"), "trajectories": []})
        run["trajectories"].append(t)

    # Splits of the runs already assigned (a run is never split; the first listed episode
    # decides for a run whose episodes are new), counted per scenario config.
    run_split: dict[str, str] = {}
    per_config: dict[tuple, dict[str, int]] = {}
    new_runs = []
    for name, run in runs.items():
        key = (run["group"], run["config"])
        cnt = per_config.setdefault(key, {"train": 0, "test": 0})
        have = [assigned[t] for t in sorted(run["trajectories"]) if t in assigned]
        if have:
            run_split[name] = have[0]
            if have[0] in cnt:
                cnt[have[0]] += 1
        else:
            new_runs.append(name)

    for name in sorted(new_runs, key=lambda n: (runs[n]["time"], n)):
        cnt = per_config[(runs[name]["group"], runs[name]["config"])]
        n = cnt["train"] + cnt["test"] + 1
        split = "test" if int(eval_fraction * n + 0.5) > cnt["test"] else "train"
        cnt[split] += 1
        run_split[name] = split

    out = {k: v for k, v in old.items() if k not in SPLIT_NAMES and k not in ("version", "summary")}
    lists = {k: set(old.get(k, [])) for k in SPLIT_NAMES}
    for name, run in runs.items():
        for t in run["trajectories"]:
            if t not in assigned:
                lists[run_split[name]].add(t)
    summary: dict[str, dict[str, int]] = {}
    for k in SPLIT_NAMES:
        for t in lists[k]:
            if t in entries:
                run = runs[sim_run(entries[t].get("source_dir", t))["run"]]
                cell = summary.setdefault(f"{run['group']}/{run['task']}", {s: 0 for s in SPLIT_NAMES})
                cell[k] += 1
    out.update({
        "version": VERSION,
        "rule": (f"simulated: per recording run, {eval_fraction:.0%} of the runs of each scenario config in "
                 "test, append-only; real: not listed yet, hash of the trajectory name (pooh_dataset.splits)"),
        "eval_fraction": eval_fraction,
        **{k: sorted(lists[k]) for k in SPLIT_NAMES},
        "summary": dict(sorted(summary.items())),
    })
    return out
