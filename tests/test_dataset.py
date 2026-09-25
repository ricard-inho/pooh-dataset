import numpy as np
import pytest
from conftest import FRAME_DT, N_FRAMES, SPEED, T0, H, W

from pooh_dataset import MissingCalibrationError
from pooh_dataset.events import event_histogram, time_surface, voxel_grid
from pooh_dataset.labels import (
    mask_to_bbox,
    nearest_indices,
    object_pose_in_camera,
    render_mask,
)
from pooh_dataset.splits import hash_split


def test_structure(ds):
    assert ds.trajectory_names == ["trajectory_a", "trajectory_b"]
    traj = ds["trajectory_a"]
    assert "firefly" in traj.modalities and "events" in traj.modalities
    assert ds.split("train") == ["trajectory_a"]
    assert ds.split("all", domain="synthetic") == ["trajectory_b"]
    assert ds.trajectory_info("trajectory_a")["domain"] == "real"


def test_frames_across_row_groups(ds):
    traj = ds["trajectory_a"]
    ts = traj.timestamps("firefly")
    assert len(ts) == N_FRAMES and ts[0] == T0
    for i in (0, 3, 4, 5, -1):  # row groups of 4 -> index 4 lives in the second group
        assert traj.frame("firefly", i).shape == (H, W, 3)
    assert traj.frame_bytes("firefly", 0)[:2] == b"\xff\xd8"
    d = traj.depth(0)
    assert d.dtype == np.float32 and np.allclose(d, 2.0)
    with pytest.raises(IndexError):
        traj.frame("firefly", N_FRAMES)


def test_events_window(ds):
    traj = ds["trajectory_a"]
    lo, hi = traj.time_range("events")
    assert T0 <= lo < hi < T0 + 500_000_000
    ev = traj.events(T0 + 100_000_000, T0 + 200_000_000)
    assert len(ev) > 0 and ev.t.min() >= T0 + 100_000_000 and ev.t.max() < T0 + 200_000_000
    assert np.all(np.diff(ev.t) >= 0)
    assert event_histogram(ev, H, W).sum() == len(ev)
    vg = voxel_grid(ev, 5, H, W, T0 + 100_000_000, T0 + 200_000_000)
    assert vg.shape == (5, H, W)
    assert np.abs(vg).sum() <= len(ev) + 1e-3  # each event spreads a total weight of 1
    ts = time_surface(ev, T0 + 200_000_000, height=H, width=W)
    assert ts.max() <= 1 and ts.min() >= 0


def test_messages_sorted(ds):
    ts, arr = ds["trajectory_a"].message_array("/pingu_low_level")
    assert np.all(np.diff(ts) > 0) and arr[:, 0].tolist() == [0, 1, 2]


def test_nearest():
    idx, ok = nearest_indices(np.array([0, 10, 20]), np.array([-5, 4, 6, 19, 100]), tolerance_ns=5)
    assert idx.tolist() == [0, 0, 1, 2, 2] and ok.tolist() == [True, True, True, True, False]


def test_object_pose_and_mask(ds):
    traj = ds["trajectory_a"]
    cube = ds.objects["cubesat"]
    T, ok = object_pose_in_camera(traj, "firefly", cube, T0 + FRAME_DT)
    assert ok and np.allclose(T[:3, 3], [-SPEED * FRAME_DT / 1e9, 0, 2.0], atol=1e-9)
    intr = traj.calibration.intrinsics("firefly")
    T0_, _ = object_pose_in_camera(traj, "firefly", cube, T0)
    box = mask_to_bbox(render_mask(cube, T0_, intr))
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    assert abs(cx - 40) <= 1 and abs(cy - 30) <= 1
    half = 60 * 0.1 / 1.9  # front face at z = 1.9
    assert abs((box[2] - box[0]) - 2 * half) <= 2


def test_missing_calibration_message(ds):
    with pytest.raises(MissingCalibrationError, match="DATA_REQUIREMENTS"):
        ds["trajectory_a"].calibration.T_body_cam("realsense_color")


def test_hash_split_stable():
    assert hash_split("trajectory_007") == hash_split("trajectory_007")
    assert {hash_split(f"t{i}") for i in range(100)} == {"train", "val", "test"}


def test_fill_mask_holes():
    from pooh_dataset.labels import fill_mask_holes

    m = np.zeros((20, 30), bool)
    m[3:15, 4:20] = True
    m[6:9, 8:12] = False  # enclosed hole -> filled
    m[10:15, 15:18] = False  # notch open to the outside -> kept
    f = fill_mask_holes(m)
    assert f[6:9, 8:12].all() and not f[12:15, 15:18].any() and f.sum() == m.sum() + 12
