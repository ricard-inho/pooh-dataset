import numpy as np

from pooh_dataset.calibration import CameraIntrinsics
from pooh_dataset.geometry import (
    interpolate_poses,
    invert_pose,
    make_pose,
    max_monotonic_radius,
    project_points,
    quat_to_rotmat,
    rotmat_to_quat,
    slerp,
)
from pooh_dataset.imaging import CameraView


def test_quat_roundtrip():
    rng = np.random.default_rng(1)
    q = rng.normal(size=(50, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 3] < 0] *= -1
    assert np.allclose(rotmat_to_quat(quat_to_rotmat(q)), q, atol=1e-9)


def test_invert_pose():
    T = make_pose([1.0, 2.0, 3.0], [0.1, 0.2, 0.3, 0.9])
    assert np.allclose(T @ invert_pose(T), np.eye(4))


def test_slerp_midpoint():
    q0 = np.array([0, 0, 0, 1.0])
    q1 = np.array([0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)])  # 90 deg about z
    q = slerp(q0, q1, 0.5)
    assert np.allclose(q, [0, 0, np.sin(np.pi / 8), np.cos(np.pi / 8)])


def test_interpolate_and_gaps():
    t = np.array([0, 10, 20, 200])
    pos = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], float)
    quat = np.tile([0, 0, 0, 1.0], (4, 1))
    T, ok = interpolate_poses(t, pos, quat, np.array([5, 15, 100, 300]), max_gap_ns=50)
    assert np.allclose(T[0, :3, 3], [0.5, 0, 0]) and np.allclose(T[1, :3, 3], [1.5, 0, 0])
    assert ok.tolist() == [True, True, False, False]


def test_projection_and_distortion_fold():
    K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1.0]])
    uv, ok = project_points(np.array([[0, 0, 1.0], [0.1, 0, 1], [0, 0, -1]]), K)
    assert np.allclose(uv[:2], [[50, 50], [60, 50]]) and ok.tolist() == [True, True, False]
    D = np.array([-0.23, 0.18, 0, 0, -0.14])  # like the Firefly
    r = max_monotonic_radius(D)
    assert 0.5 < r < 5
    _, ok = project_points(np.array([[r * 1.5, 0, 1.0]]), K, D)
    assert not ok[0]


def test_undistort_view_identity_without_distortion():
    intr = CameraIntrinsics("c", 40, 30, np.array([[30, 0, 19.5], [0, 30, 14.5], [0, 0, 1.0]]))
    img = np.random.default_rng(0).integers(0, 255, (30, 40, 3), dtype=np.uint8)
    v = CameraView(intr, undistort=True)
    assert v.is_identity and v(img) is img
    small = CameraView(intr, size=(20, 15))
    assert small(img).shape == (15, 20, 3)
    assert np.isclose(small.intrinsics.K[0, 0], 15) and np.isclose(small.intrinsics.K[0, 2], 9.5)


def test_undistort_roundtrip_center():
    D = np.array([-0.2, 0.05, 0, 0, 0])
    intr = CameraIntrinsics("c", 64, 48, np.array([[50, 0, 31.5], [0, 50, 23.5], [0, 0, 1.0]]), D)
    v = CameraView(intr, undistort=True)
    img = np.zeros((48, 64), np.uint8)
    img[20:28, 28:36] = 255  # a blob at the principal point is barely moved
    out = v(img)
    assert out[23, 31] == 255 and not v.intrinsics.has_distortion
