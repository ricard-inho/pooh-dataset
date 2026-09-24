"""Rigid-body math: quaternions (x, y, z, w order, as stored in the dataset), 4x4 poses,
pose interpolation and pinhole + plumb_bob (radial-tangential) projection.

Naming convention: ``T_a_b`` maps points expressed in frame ``b`` into frame ``a``
(i.e. it is the pose of ``b`` in ``a``).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "quat_to_rotmat",
    "rotmat_to_quat",
    "make_pose",
    "invert_pose",
    "transform_points",
    "slerp",
    "interpolate_poses",
    "project_points",
    "distort_normalized",
    "max_monotonic_radius",
]


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """(..., 4) xyzw quaternions -> (..., 3, 3) rotation matrices."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = np.moveaxis(q, -1, 0)
    R = np.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    )  # fmt: skip
    return R.reshape(q.shape[:-1] + (3, 3))


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) rotation matrices -> (..., 4) xyzw quaternions with w >= 0."""
    R = np.asarray(R, dtype=np.float64)
    flat = R.reshape(-1, 3, 3)
    out = np.empty((flat.shape[0], 4))
    for i, m in enumerate(flat):
        tr = np.trace(m)
        if tr > 0:
            s = 2.0 * np.sqrt(tr + 1.0)
            q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, s / 4]
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            q = [s / 4, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            q = [(m[0, 1] + m[1, 0]) / s, s / 4, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, s / 4, (m[1, 0] - m[0, 1]) / s]
        q = np.asarray(q)
        out[i] = q if q[3] >= 0 else -q
    return out.reshape(R.shape[:-2] + (4,))


def make_pose(position: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    """Build (..., 4, 4) homogeneous poses from positions (..., 3) and xyzw quaternions."""
    position = np.asarray(position, dtype=np.float64)
    R = quat_to_rotmat(quat_xyzw)
    T = np.zeros(R.shape[:-2] + (4, 4))
    T[..., :3, :3] = R
    T[..., :3, 3] = position
    T[..., 3, 3] = 1.0
    return T


def invert_pose(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Rt = np.swapaxes(R, -1, -2)
    out = np.zeros_like(T)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, t)
    out[..., 3, 3] = 1.0
    return out


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a (4, 4) pose to (N, 3) points, or per-point (N, 4, 4) poses to (N, 3) points."""
    T = np.asarray(T, dtype=np.float64)
    pts = np.asarray(pts, dtype=np.float64)
    if T.ndim == 2:
        return pts @ T[:3, :3].T + T[:3, 3]
    return np.einsum("nij,nj->ni", T[:, :3, :3], pts) + T[:, :3, 3]


def slerp(q0: np.ndarray, q1: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Batched spherical interpolation between xyzw quaternions (shortest path)."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64).copy()
    alpha = np.asarray(alpha, dtype=np.float64)[..., None]
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    near = sin_theta < 1e-6
    safe = np.where(near, 1.0, sin_theta)
    w0 = np.where(near, 1 - alpha, np.sin((1 - alpha) * theta) / safe)
    w1 = np.where(near, alpha, np.sin(alpha * theta) / safe)
    q = w0 * q0 + w1 * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def interpolate_poses(
    t_src: np.ndarray,
    positions: np.ndarray,
    quats_xyzw: np.ndarray,
    t_query: np.ndarray | int,
    max_gap_ns: int | None = 50_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a pose track (linear position, slerp rotation) at query timestamps.

    Returns ``(T, valid)`` with ``T`` of shape (M, 4, 4). A query is invalid when it falls
    outside the track or when the two bracketing samples are more than ``max_gap_ns`` apart
    (a tracking dropout); invalid poses are filled with NaN.
    """
    scalar = np.ndim(t_query) == 0
    t_query = np.atleast_1d(np.asarray(t_query, dtype=np.int64))
    t_src = np.asarray(t_src, dtype=np.int64)
    order = np.argsort(t_src, kind="stable")
    t_src, positions, quats_xyzw = t_src[order], positions[order], quats_xyzw[order]

    if len(t_src) == 0:
        T = np.full((len(t_query), 4, 4), np.nan)
        valid = np.zeros(len(t_query), dtype=bool)
        return (T[0], valid[0]) if scalar else (T, valid)

    hi = np.clip(np.searchsorted(t_src, t_query, side="left"), 1, max(len(t_src) - 1, 1))
    lo = hi - 1
    if len(t_src) == 1:
        lo = hi = np.zeros_like(t_query)
    t0, t1 = t_src[lo], t_src[hi]
    span = (t1 - t0).astype(np.float64)
    alpha = np.where(span > 0, (t_query - t0) / np.where(span > 0, span, 1.0), 0.0)

    valid = (t_query >= t_src[0]) & (t_query <= t_src[-1])
    if max_gap_ns is not None:
        valid &= (t1 - t0) <= max_gap_ns
    alpha = np.clip(alpha, 0.0, 1.0)

    pos = positions[lo] + alpha[:, None] * (positions[hi] - positions[lo])
    quat = slerp(quats_xyzw[lo], quats_xyzw[hi], alpha)
    T = make_pose(pos, quat)
    T[~valid] = np.nan
    return (T[0], bool(valid[0])) if scalar else (T, valid)


def distort_normalized(xy: np.ndarray, D: np.ndarray | None) -> np.ndarray:
    """Apply plumb_bob / radtan distortion (k1, k2, p1, p2[, k3]) to normalized coords (N, 2)."""
    if D is None or len(D) == 0 or not np.any(D):
        return xy
    d = np.zeros(5)
    d[: min(5, len(D))] = np.asarray(D, dtype=np.float64)[:5]
    k1, k2, p1, p2, k3 = d
    x, y = xy[:, 0], xy[:, 1]
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([xd, yd], axis=-1)


def max_monotonic_radius(D: np.ndarray | None, r_limit: float = 5.0) -> float:
    """Largest normalized radius for which radial distortion is still monotonic.

    Beyond it the polynomial folds back and far-off-image points would project *into* the
    image; such points are treated as invalid.
    """
    if D is None or len(D) == 0 or not np.any(D):
        return np.inf
    d = np.zeros(5)
    d[: min(5, len(D))] = np.asarray(D, dtype=np.float64)[:5]
    k1, k2, _, _, k3 = d
    r = np.linspace(0, r_limit, 20001)
    r2 = r * r
    deriv = 1 + 3 * k1 * r2 + 5 * k2 * r2**2 + 7 * k3 * r2**3
    bad = np.nonzero(deriv <= 0)[0]
    return float(r[bad[0]]) if len(bad) else np.inf


def project_points(
    pts_cam: np.ndarray,
    K: np.ndarray,
    D: np.ndarray | None = None,
    min_depth: float = 1e-3,
    r_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project (N, 3) camera-frame points. Returns (uv (N, 2), valid (N,)).

    ``valid`` is False for points behind the camera or outside the monotonic region of the
    distortion model; their uv are still returned but meaningless.
    """
    pts_cam = np.asarray(pts_cam, dtype=np.float64).reshape(-1, 3)
    z = pts_cam[:, 2]
    valid = z > min_depth
    zs = np.where(valid, z, 1.0)
    xy = pts_cam[:, :2] / zs[:, None]
    if D is not None and np.any(D):
        if r_max is None:
            r_max = max_monotonic_radius(D)
        valid &= np.hypot(xy[:, 0], xy[:, 1]) < r_max
        xy = distort_normalized(xy, D)
    K = np.asarray(K, dtype=np.float64)
    u = K[0, 0] * xy[:, 0] + K[0, 1] * xy[:, 1] + K[0, 2]
    v = K[1, 1] * xy[:, 1] + K[1, 2]
    return np.stack([u, v], axis=-1), valid
