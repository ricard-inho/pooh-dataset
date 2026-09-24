"""A tiny synthetic dataset with the exact Hub layout and analytically known labels.

Rig ("SensorStack", camera frame == body frame) moves along +x at 1 m/s, looking down +z.
A 0.2 m cube ("cubesat") sits still at (0, 0, 2). Depth is a constant 2 m wall.
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


def _write(root, modality, traj, table, row_group_size=None, files=1):
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
    return root


@pytest.fixture(scope="session")
def fake_root(tmp_path_factory):
    return build(tmp_path_factory.mktemp("pooh"))


@pytest.fixture()
def ds(fake_root):
    from pooh_dataset import PoohDataset

    return PoohDataset(fake_root, repo_id="test/fake")
