"""A tiny synthetic dataset with the exact Hub layout and analytically known labels.

Rig ("SensorStack", camera frame == body frame) moves along +x at 1 m/s, looking down +z.
A 0.2 m cube ("cubesat") sits still at (0, 0, 2). Depth is a constant 2 m wall.

Plus one simulated trajectory (``sim_3d_<modality>/trajectory_sim_000001``, simulation
calibration and metadata, ``sim_manifest.json``): the robot moves along +y at 1 m/s, its
camera (sensor frame = robot base frame, x forward) looks down world +x at a static target at
(5, 0, 0) in front of a 5 m wall.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from PIL import Image

T0 = 1_790_000_000_000_000_000
W, H, F = 80, 60, 60.0
N_FRAMES = 6
FRAME_DT = 100_000_000  # 100 ms
SPEED = 1.0  # m/s along +x
MOCAP_END = 740_000_000  # last mocap sample, ns after T0
GOAL_X = SPEED * MOCAP_END / 1e9 - 0.03  # rig ends 3 cm past the goal

INTR = {"width": W, "height": H, "distortion_model": "plumb_bob", "D": [0.0] * 5,
        "K": [F, 0, 39.5, 0, F, 29.5, 0, 0, 1], "source": "ros_camera_info"}


def _jpeg(arr):
    b = io.BytesIO()
    Image.fromarray(arr).save(b, format="JPEG", quality=95)
    return b.getvalue()


def _png16(arr):
    b = io.BytesIO()
    Image.fromarray(arr).save(b, format="PNG")
    return b.getvalue()


SIM_TRAJ = "trajectory_sim_000001"
SIM_W, SIM_H, SIM_F = 64, 48, 40.0
SIM_TARGET_X = 5.0


def _write(root, modality, traj, table, row_group_size=None, files=1):
    # every table on the Hub carries the trajectory name
    table = table.append_column("trajectory_id", pa.array([traj] * table.num_rows))
    d = root / modality / traj
    d.mkdir(parents=True, exist_ok=True)
    n = table.num_rows
    for k in range(files):
        part = table.slice(k * n // files, (k + 1) * n // files - k * n // files)
        pq.write_table(part, d / f"data-{k:05d}-of-{files:05d}.parquet", row_group_size=row_group_size)


def build(root, trajectories=("trajectory_a", "trajectory_b")):
    rng = np.random.default_rng(0)
    frame_ts = T0 + np.arange(N_FRAMES) * FRAME_DT
    for traj in trajectories:
        img = [_jpeg(rng.integers(0, 255, (H, W, 3), dtype=np.uint8)) for _ in frame_ts]
        _write(root, "firefly", traj, pa.table({
            "image": [{"bytes": b, "path": None} for b in img], "timestamp_ns": frame_ts}),
            row_group_size=4)
        dep = [_png16(np.full((H, W), 2000, np.uint16)) for _ in frame_ts]
        _write(root, "realsense_depth", traj, pa.table({
            "image": [{"bytes": b, "path": None} for b in dep], "timestamp_ns": frame_ts}))

        n_ev = 1000
        t = np.sort(rng.integers(T0, T0 + 500_000_000, n_ev)).astype(np.int64)
        _write(root, "events", traj, pa.table({
            "t": t, "x": rng.integers(0, W, n_ev).astype(np.uint16),
            "y": rng.integers(0, H, n_ev).astype(np.uint16),
            "p": rng.integers(0, 2, n_ev).astype(np.uint8)}), row_group_size=100, files=2)

        mt = T0 - 50_000_000 + np.arange(80) * 10_000_000
        assert mt[-1] == T0 + MOCAP_END  # 100 Hz, covers all frames
        rows = []
        for ti in mt:
            dt = (ti - T0) / 1e9
            rows.append(("SensorStack", ti, SPEED * dt, 0.0, 0.0))
            rows.append(("cubesat", ti, 0.0, 0.0, 2.0))
        b, ts, x, y, z = map(list, zip(*rows))
        n = len(rows)
        _write(root, "mocap", traj, pa.table({
            "body": b, "timestamp_ns": ts, "header_stamp_ns": ts, "bag_timestamp_ns": ts,
            "x": x, "y": y, "z": z, "qx": [0.0] * n, "qy": [0.0] * n, "qz": [0.0] * n,
            "qw": [1.0] * n}))

        goal = {"header": {"frame_id": "world"}, "pose": {
            "position": {"x": GOAL_X, "y": 0.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}}
        _write(root, "robot_messages", traj, pa.table({
            "topic": ["/pingu_low_level"] * 3 + ["/observation_formater_input"],
            "msgtype": ["std_msgs/msg/Float64MultiArray"] * 3 + ["geometry_msgs/msg/PoseStamped"],
            "timestamp_ns": [T0 + 2, T0, T0 + 1, T0 - 10], "header_stamp_ns": [None] * 4,
            "bag_timestamp_ns": [T0] * 4,
            "data": [json.dumps({"data": [float(i), 1.0]}) for i in (2, 0, 1)] + [json.dumps(goal)]}))

        cal = root / "calibration"
        cal.mkdir(exist_ok=True)
        (cal / f"{traj}.yaml").write_text(yaml.safe_dump({
            "firefly": INTR, "events": INTR,
            "realsense_depth": {**INTR, "depth_scale": 0.001}}))

    (root / "calibration" / "rig.yaml").write_text(yaml.safe_dump({"extrinsics": {
        "reference_body": "SensorStack",
        "cameras": {c: {"T_body_cam": np.eye(4).ravel().tolist()}
                    for c in ("firefly", "events", "realsense_depth")}}}))
    (root / "models").mkdir(exist_ok=True)
    (root / "models" / "objects.yaml").write_text(yaml.safe_dump(
        {"cubesat": {"box": [0.2, 0.2, 0.2], "mocap_body": "cubesat", "class_id": 1}}))
    (root / "manifest.json").write_text(json.dumps({
        "repo_id": "test/fake",
        "trajectories": {t: {"firefly": N_FRAMES} for t in trajectories},
        "trajectory_info": {"trajectory_b": {"domain": "synthetic"}}}))
    (root / "splits.json").write_text(json.dumps(
        {"version": 1, "train": ["trajectory_a"], "val": [], "test": ["trajectory_b"]}))
    build_sim(root)
    return root


def _png(arr):
    b = io.BytesIO()
    Image.fromarray(arr).save(b, format="PNG")
    return b.getvalue()


def build_sim(root, traj=SIM_TRAJ, group="sim_3d"):
    """A simulated trajectory in the Hub layout (timestamps start at the reset: 0)."""
    frame_ts = np.arange(N_FRAMES, dtype=np.int64) * FRAME_DT
    rgb = [_png(np.full((SIM_H, SIM_W, 3), 80 + 10 * i, np.uint8)) for i in range(N_FRAMES)]
    depth = [_png16(np.full((SIM_H, SIM_W), int(SIM_TARGET_X * 1000), np.uint16)) for _ in frame_ts]
    sem = np.zeros((SIM_H, SIM_W), np.uint8)
    sem[SIM_H // 2 - 4:SIM_H // 2 + 4, SIM_W // 2 - 4:SIM_W // 2 + 4] = 2
    for mod, imgs in (("rgb", rgb), ("depth", depth), ("semantic", [_png(sem)] * N_FRAMES)):
        _write(root, f"{group}_{mod}", traj, pa.table({
            "image": [{"bytes": b, "path": None} for b in imgs], "timestamp_ns": frame_ts,
            "frame_index": np.arange(N_FRAMES)}))
    n_ev = 500
    rng = np.random.default_rng(1)
    _write(root, f"{group}_events", traj, pa.table({
        "t": np.sort(rng.integers(0, 500_000_000, n_ev)).astype(np.int64),
        "x": rng.integers(0, SIM_W, n_ev).astype(np.uint16),
        "y": rng.integers(0, SIM_H, n_ev).astype(np.uint16),
        "p": rng.integers(0, 2, n_ev).astype(np.uint8)}))
    mt = np.arange(0, 700_000_000, 10_000_000, dtype=np.int64)
    rows = []
    for ti in mt:
        rows.append(("robot", ti, 0.0, SPEED * ti / 1e9, 0.0))
        rows.append(("target", ti, SIM_TARGET_X, 0.0, 0.0))
    b, ts, x, y, z = map(list, zip(*rows))
    n = len(rows)
    zero = [0.0] * n
    _write(root, f"{group}_mocap", traj, pa.table({
        "timestamp_ns": ts, "body": b, "x": x, "y": y, "z": z, "qx": zero, "qy": zero, "qz": zero,
        "qw": [1.0] * n, "vx": zero, "vy": [SPEED if bb == "robot" else 0.0 for bb in b], "vz": zero,
        "wx": zero, "wy": zero, "wz": zero}))
    _write(root, f"{group}_imu", traj, pa.table({
        "timestamp_ns": mt, **{f"angular_velocity_{a}": np.zeros(len(mt)) for a in "xyz"},
        **{f"linear_acceleration_{a}": np.zeros(len(mt)) for a in "xyz"}}))
    _write(root, f"{group}_lidar", traj, pa.table({
        "scan_index": [0], "timestamp_ns": [0], "x": [[SIM_TARGET_X]], "y": [[0.0]], "z": [[0.0]],
        "distance": [[SIM_TARGET_X]], "channel": pa.array([[3]], pa.list_(pa.uint16()))}))
    steps = np.arange(4)
    _write(root, f"{group}_steps", traj, pa.table({
        "timestamp_ns": steps * 100_000_000, "step": steps, "action": [[0.0, 1.0]] * 4,
        "observation": [[1.0, 2.0, 3.0]] * 4, "reward": np.ones(4), "reward_terms": [[0.5, 0.5]] * 4,
        "goal_position_error": [1.0, 0.5, 0.1, 0.02], "goal_orientation_error": [0.1, 0.05, 0.02, 0.01],
        "task_state": ["{}"] * 4, "terminated": [False] * 4, "truncated": [False, False, False, True]}))

    mount = {"parent": "robot base", "translation_m": [0.0, 0.0, 0.0], "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]}
    cam = {"source": "simulation", "width": SIM_W, "height": SIM_H, "distortion_model": "none",
           "D": [0.0] * 5, "K": [SIM_F, 0, (SIM_W - 1) / 2, 0, SIM_F, (SIM_H - 1) / 2, 0, 0, 1],
           "T_base_sensor": mount}
    (root / "calibration").mkdir(exist_ok=True)
    (root / "calibration" / f"{traj}.yaml").write_text(yaml.safe_dump({
        "rgb": cam, "depth": {**cam, "unit": "mm", "scale_m": 0.001}, "semantic": cam,
        "events": {**cam, "emulator": "v2e"}, "lidar": {"source": "simulation", "T_base_sensor": mount}}))
    (root / "metadata").mkdir(exist_ok=True)
    (root / "metadata" / f"{traj}.yaml").write_text(yaml.safe_dump({
        "source": "simulation", "semantic_classes": {"BACKGROUND": 0, "UNLABELLED": 1, "target": 2}}))
    counts = {"rgb": N_FRAMES, "depth": N_FRAMES, "semantic": N_FRAMES, "events": n_ev, "mocap": n,
              "imu": len(mt), "lidar": 1, "steps": 4}
    (root / "sim_manifest.json").write_text(json.dumps({"license": "cc-by-4.0", "groups": [group], "trajectories": {
        traj: {"counts": counts, "group": group, "scenario": "orbital/cubesat/black", "duration_s": 0.7,
               "goal_position_error_m": 0.02, "goal_orientation_error_deg": 0.6,
               "source_dir": "orbital_cubesat_black_approach_00000_20261015T234104_001_env000_ep0002"}}}))


@pytest.fixture(scope="session")
def fake_root(tmp_path_factory):
    return build(tmp_path_factory.mktemp("pooh"))


@pytest.fixture()
def ds(fake_root):
    from pooh_dataset import PoohDataset

    return PoohDataset(fake_root, repo_id="test/fake")
