"""splits.json: append-only simulated splits per recording run, hash fallback for real data."""

import json

import pytest
from conftest import SIM_TRAJ

from pooh_dataset.splits import hash_split, load_benchmark, load_splits, sim_run, update_sim_splits

CONFIGS = {"sim_3d": ["orbital_iss_earth_orbit", "orbital_cubesat_black_approach"],
           "sim_zerog": ["zerog_cubesat_arm_wall_docking", "zerog_satellite_mockup_floating_inspection"]}


def manifest(day: int, runs_per_config: int = 5, episodes: int = 3, start: int = 0) -> dict:
    """Sim manifest of one recording day: ``runs_per_config`` runs of each config."""
    trajs, n = {}, start
    for group, configs in CONFIGS.items():
        for config in configs:
            for r in range(runs_per_config):
                run = f"{config}_{r:05d}_202610{day:02d}T{r:02d}0000_000"
                for ep in range(episodes):
                    n += 1
                    trajs[f"trajectory_fixture_{n:06d}"] = {
                        "group": group, "scenario": "x", "source_dir": f"{run}_env000_ep{ep:04d}"}
    return {"trajectories": trajs}


def merged(*ms: dict) -> dict:
    return {"trajectories": {t: e for m in ms for t, e in m["trajectories"].items()}}


def test_sim_run():
    r = sim_run("zerog_cubesat_arm_wall_docking_00012_20261017T101530_002_env000_ep0001")
    assert r == {"run": "zerog_cubesat_arm_wall_docking_00012_20261017T101530_002",
                 "config": "zerog_cubesat_arm_wall_docking", "task": "docking", "time": "20261017T101530"}
    assert sim_run("orbital_iss_earth_orbit_00001_20261016T000000_000_env000_ep0002")["task"] == "orbit"
    assert sim_run("odd")["task"] is None


def run_of(m, t):
    return sim_run(m["trajectories"][t]["source_dir"])["run"]


def test_every_cell_gets_train_and_test_and_runs_stay_together():
    m = manifest(16)
    s = update_sim_splits(None, m)
    assert set(s["summary"]) == {"sim_3d/orbit", "sim_3d/approach", "sim_zerog/docking", "sim_zerog/inspection"}
    for cell in s["summary"].values():  # 5 runs x 3 episodes: 1 run in test
        assert cell == {"train": 12, "val": 0, "test": 3}
    split_of = {t: k for k in ("train", "test") for t in s[k]}
    runs: dict[str, set] = {}
    for t in m["trajectories"]:
        runs.setdefault(run_of(m, t), set()).add(split_of[t])
    assert all(len(v) == 1 for v in runs.values())


def test_append_only_across_days():
    day1 = manifest(16)
    s1 = update_sim_splits(None, day1)
    day2 = manifest(17, start=len(day1["trajectories"]))
    s2 = update_sim_splits(s1, merged(day1, day2))
    for k in ("train", "test"):
        assert set(s1[k]) <= set(s2[k])
    for cell in s2["summary"].values():  # 10 runs per cell: 2 in test
        assert cell["test"] == 6 and cell["train"] == 24
    assert update_sim_splits(s2, merged(day1, day2)) == s2  # idempotent
    # a changed fraction only affects new runs
    s3 = update_sim_splits(s2, merged(day1, day2), eval_fraction=0.5)
    assert s3["train"] == s2["train"] and s3["test"] == s2["test"]


def test_real_and_extra_keys_kept():
    s = update_sim_splits({"version": 1, "train": ["trajectory_007"], "test": ["trajectory_009"],
                           "custom": ["a"]}, manifest(16, runs_per_config=1))
    assert "trajectory_007" in s["train"] and "trajectory_009" in s["test"] and s["custom"] == ["a"]


def test_load_splits_fallbacks(tmp_path):
    (tmp_path / "splits.json").write_text(json.dumps({"version": 2, "train": ["sim_a"], "test": [],
                                                      "summary": {}}))
    with pytest.warns(UserWarning, match="real trajectories"):
        s = load_splits(tmp_path, ["sim_a", "sim_b", "trajectory_007"], unlisted=["trajectory_007"])
    assert s["train"][0] == "sim_a" and "summary" not in s
    assert "trajectory_007" in s[hash_split("trajectory_007")]
    assert all("sim_b" not in v for v in s.values())  # unlisted sim: no split


def test_dataset_tasks(ds):
    assert ds.trajectory_info(SIM_TRAJ)["task"] == "approach"
    assert ds.tasks() == {"sim_3d/approach": [SIM_TRAJ]}
    assert ds.split("all", "synthetic", task="approach") == [SIM_TRAJ]
    assert ds.split("all", "synthetic", task="docking") == []


def test_benchmark_v1_frozen():
    b = load_benchmark("v1")
    assert b == load_benchmark("benchmark_v1") and b["name"] == "benchmark_v1"
    assert b["revision"] == "8205a1129f1fbb64cf7a4d0fc342b0a9e5c90509"
    assert len(b["train"]) == 1762 and len(b["test"]) == 436 and not b["val"]
    assert not set(b["train"]) & set(b["test"])
    assert "trajectory_009" in b["test"] and {"trajectory_007", "trajectory_008"} <= set(b["train"])
    assert b["summary"]["sim_zerog/docking"] == {"train": 300, "val": 0, "test": 72}
    # the Hub's splits.json carries an untouched copy
    assert update_sim_splits(None, manifest(16))["benchmark_v1"] == b


def test_benchmark_split_and_download_pin(ds, monkeypatch):
    from pooh_dataset import hub

    assert ds.split("test", benchmark="v1") == []  # none of the fixture's trajectories is in it
    with pytest.raises(ValueError, match="pinned"):
        hub.download(root=ds.root, benchmark="v1", revision="main")
    with pytest.raises(ValueError, match="unknown benchmark"):
        load_benchmark("v99")


def test_benchmark_fixes_sim_assignments():
    b = load_benchmark("v1")
    sim = [t for t in b["train"] + b["test"] if t.startswith("trajectory_hera_")][:30]
    m = {"trajectories": {t: {"group": "sim_zerog", "scenario": "x",
                              "source_dir": f"zerog_cubesat_arm_wall_docking_00000_20261016T000000_{i:03d}_env000_ep0001"}
                          for i, t in enumerate(sim)}}
    s = update_sim_splits(None, m)
    for t in sim:  # each its own run: the balancing would put some in other splits
        assert (t in s["test"]) == (t in b["test"])


def test_dataset_default_benchmark(ds):
    from pooh_dataset import PoohDataset

    b = PoohDataset(ds.root, benchmark="v1")
    assert b.split("train") == [] and b.split("all") == []  # none of the fixture's trajectories is in v1
    assert ds.split("train") == ["trajectory_a"]  # no default benchmark
    with pytest.raises(ValueError):
        PoohDataset(ds.root, benchmark="v99")
