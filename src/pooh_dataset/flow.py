"""Optical-flow ground truth from depth + mocap.

Scene points are back-projected from a RealSense depth frame, moved with the world (static
background) or with the tracked object they belong to, and projected into the target camera
at ``t0`` and ``t1``. The difference of the two projections is the flow. This works for any
calibrated camera, including the event camera (flow between ``t0`` and ``t1`` for the events
in that window).

Limitations: untracked moving things (arms, people) get background motion, and pixels not
covered by the depth sensor are invalid. Use the returned ``valid`` mask.
"""

from __future__ import annotations

import numpy as np

from .calibration import CameraIntrinsics
from .geometry import distort_normalized, invert_pose, transform_points
from .labels import camera_pose, nearest_indices, object_pose, render_instances
from .objects import ObjectModel
from .trajectory import Trajectory

__all__ = ["backproject_depth", "flow_from_world_points", "optical_flow"]


def _undistort_normalized(xy_d: np.ndarray, D: np.ndarray, iters: int = 10) -> np.ndarray:
    xy = xy_d.copy()
    for _ in range(iters):
        err = distort_normalized(xy, D) - xy_d
        xy -= err
    return xy


def backproject_depth(depth_m: np.ndarray, intr: CameraIntrinsics, stride: int = 1):
    """Depth image (metres, 0 = invalid) -> (points (N, 3) in the camera frame, pixels (N, 2))."""
    vs, us = np.nonzero(depth_m[::stride, ::stride] > 0)
    us, vs = us * stride, vs * stride
    z = depth_m[vs, us].astype(np.float64)
    K = intr.K
    y = (vs - K[1, 2]) / K[1, 1]
    x = (us - K[0, 2] - K[0, 1] * y) / K[0, 0]
    xy = np.stack([x, y], -1)
    if intr.has_distortion:
        xy = _undistort_normalized(xy, intr.D)
    pts = np.column_stack([xy * z[:, None], z])
    return pts, np.stack([us, vs], -1)


def flow_from_world_points(
    X_w0: np.ndarray, X_w1: np.ndarray, T_w_c0: np.ndarray, T_w_c1: np.ndarray, intr: CameraIntrinsics
) -> tuple[np.ndarray, np.ndarray]:
    """Forward flow at ``t0`` of world points that move from ``X_w0`` to ``X_w1``.

    Returns ``flow`` (H, W, 2) float32 (u, v displacement) and ``valid`` (H, W) bool. When
    several points land on one pixel the nearest one wins (z-buffer).
    """
    p0 = transform_points(invert_pose(T_w_c0), X_w0)
    p1 = transform_points(invert_pose(T_w_c1), X_w1)
    uv0, ok0 = intr.project(p0)
    uv1, ok1 = intr.project(p1)
    pix = np.rint(uv0).astype(np.int64)
    ok = ok0 & ok1 & (pix[:, 0] >= 0) & (pix[:, 0] < intr.width) & (pix[:, 1] >= 0) & (pix[:, 1] < intr.height)
    pix, d, z = pix[ok], (uv1 - uv0)[ok], p0[ok, 2]
    order = np.argsort(-z)  # far first, near written last
    flat = pix[order, 1] * intr.width + pix[order, 0]
    flow = np.zeros((intr.height * intr.width, 2), np.float32)
    valid = np.zeros(intr.height * intr.width, bool)
    flow[flat] = d[order]
    valid[flat] = True
    return flow.reshape(intr.height, intr.width, 2), valid.reshape(intr.height, intr.width)


def optical_flow(
    traj: Trajectory,
    camera: str,
    t0_ns: int,
    t1_ns: int,
    objects: list[ObjectModel] = (),
    depth_modality: str = "realsense_depth",
    depth_index: int | None = None,
    intr: CameraIntrinsics | None = None,
    depth_tolerance_ns: int = 50_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Dense flow GT for ``camera`` from ``t0_ns`` to ``t1_ns``.

    ``intr`` overrides the camera's intrinsics (e.g. an undistorted / resized
    :class:`~pooh_dataset.imaging.CameraView`). Returns ``(flow, valid)``.
    """
    intr = intr or traj.calibration.intrinsics(camera)
    d_intr = traj.calibration.intrinsics(depth_modality)
    if depth_index is None:
        depth_index, ok = nearest_indices(traj.timestamps(depth_modality), t0_ns, depth_tolerance_ns)
        if not ok:
            return np.zeros((intr.height, intr.width, 2), np.float32), np.zeros((intr.height, intr.width), bool)
    t_d = int(traj.timestamps(depth_modality)[depth_index])
    pts_d, pix_d = backproject_depth(traj.depth(depth_index, depth_modality), d_intr)

    T_w_d, ok_d = camera_pose(traj, depth_modality, t_d)
    T_w_c0, ok0 = camera_pose(traj, camera, t0_ns)
    T_w_c1, ok1 = camera_pose(traj, camera, t1_ns)
    if not (ok_d and ok0 and ok1):
        return np.zeros((intr.height, intr.width, 2), np.float32), np.zeros((intr.height, intr.width), bool)

    X_wd = transform_points(T_w_d, pts_d)
    X_w0, X_w1 = X_wd.copy(), X_wd.copy()
    if objects:
        T_d_w = invert_pose(T_w_d)
        poses = [object_pose(traj, o, t) for o in objects for t in (t_d, t0_ns, t1_ns)]
        in_depth = [(o, T_d_w @ poses[3 * i][0]) for i, o in enumerate(objects)]
        inst, _ = render_instances(in_depth, d_intr)
        ids = inst[pix_d[:, 1], pix_d[:, 0]]
        for i in range(len(objects)):
            (T_od, v_d), (T_o0, v0), (T_o1, v1) = poses[3 * i : 3 * i + 3]
            sel = ids == i + 1
            if not sel.any():
                continue
            if not (v_d and v0 and v1):  # object untracked: its pixels have unknown motion
                X_w0[sel] = np.nan
                continue
            X_obj = transform_points(invert_pose(T_od), X_wd[sel])
            X_w0[sel] = transform_points(T_o0, X_obj)
            X_w1[sel] = transform_points(T_o1, X_obj)
    keep = np.isfinite(X_w0).all(1)
    return flow_from_world_points(X_w0[keep], X_w1[keep], T_w_c0, T_w_c1, intr)
