"""Ground-truth labels derived from motion capture: camera / object poses, instance masks
and bounding boxes. Everything needs the rig extrinsics and object models described in
``DATA_REQUIREMENTS.md``.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .calibration import CameraIntrinsics, MissingCalibrationError
from .geometry import invert_pose, transform_points
from .objects import ObjectModel
from .trajectory import Trajectory

__all__ = [
    "nearest_indices",
    "camera_pose",
    "object_pose",
    "object_pose_in_camera",
    "render_mask",
    "render_instances",
    "mask_to_bbox",
    "projected_bbox",
]


def nearest_indices(ref_ts: np.ndarray, query_ts, tolerance_ns: int | None = None):
    """For each query timestamp, the index of the nearest ``ref_ts`` entry.

    Returns ``(indices, valid)``; ``valid`` is False where the nearest sample is further
    than ``tolerance_ns`` away.
    """
    ref_ts = np.asarray(ref_ts, dtype=np.int64)
    scalar = np.ndim(query_ts) == 0
    q = np.atleast_1d(np.asarray(query_ts, dtype=np.int64))
    order = np.argsort(ref_ts, kind="stable")
    s = ref_ts[order]
    hi = np.clip(np.searchsorted(s, q), 1, len(s) - 1) if len(s) > 1 else np.zeros_like(q)
    lo = np.maximum(hi - 1, 0)
    pick = np.where(np.abs(s[hi] - q) < np.abs(s[lo] - q), hi, lo)
    valid = np.ones(len(q), bool) if tolerance_ns is None else np.abs(s[pick] - q) <= tolerance_ns
    idx = order[pick]
    return (int(idx[0]), bool(valid[0])) if scalar else (idx, valid)


# ------------------------------------------------------------------- poses


def camera_pose(traj: Trajectory, camera: str, t_ns, max_gap_ns: int = 50_000_000):
    """T_world_cam at ``t_ns`` from the reference body's mocap track and the extrinsics."""
    calib = traj.calibration
    T_body_cam = calib.T_body_cam(camera)
    T_world_body, valid = traj.body_pose(calib.reference_body, t_ns, max_gap_ns)
    return T_world_body @ T_body_cam, valid


def object_pose(traj: Trajectory, obj: ObjectModel, t_ns, max_gap_ns: int = 50_000_000):
    """T_world_model at ``t_ns``."""
    T_world_body, valid = traj.body_pose(obj.mocap_body, t_ns, max_gap_ns)
    return T_world_body @ obj.T_body_model, valid


def object_pose_in_camera(traj: Trajectory, camera: str, obj: ObjectModel, t_ns,
                          max_gap_ns: int = 50_000_000):
    """T_cam_model at ``t_ns``: the 6D pose label of ``obj`` as seen by ``camera``."""
    T_wc, v1 = camera_pose(traj, camera, t_ns, max_gap_ns)
    T_wo, v2 = object_pose(traj, obj, t_ns, max_gap_ns)
    return invert_pose(T_wc) @ T_wo, np.logical_and(v1, v2)


# --------------------------------------------------------------- rendering


def render_mask(obj: ObjectModel, T_cam_model: np.ndarray, intr: CameraIntrinsics) -> np.ndarray:
    """(H, W) bool silhouette of ``obj``: union of its projected triangles.

    Exact for a closed mesh as long as the object is fully in front of the camera; triangles
    with a vertex behind the camera (or outside the distortion model's valid radius) are
    dropped. Occlusion by *other* objects is handled by :func:`render_instances`.
    """
    if not np.all(np.isfinite(T_cam_model)):
        return np.zeros((intr.height, intr.width), bool)
    pts = transform_points(T_cam_model, obj.vertices)
    uv, ok = intr.project(pts)
    tri_ok = ok[obj.faces].all(axis=1)
    if not tri_ok.any():
        return np.zeros((intr.height, intr.width), bool)
    img = Image.new("1", (intr.width, intr.height), 0)
    draw = ImageDraw.Draw(img)
    for tri in uv[obj.faces[tri_ok]]:
        draw.polygon([tuple(p) for p in tri], fill=1)
    return np.asarray(img, dtype=bool)


def render_instances(
    objects: list[tuple[ObjectModel, np.ndarray]], intr: CameraIntrinsics
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Render several objects with painter's-order occlusion (by distance to the camera).

    Returns ``(instance_map, visible_masks)``: an int32 (H, W) map holding ``1 + index`` into
    ``objects`` (0 = background) and the per-object *visible* masks.
    """
    inst = np.zeros((intr.height, intr.width), np.int32)
    depth = [
        np.linalg.norm(T[:3, 3]) if np.all(np.isfinite(T)) else np.inf for _, T in objects
    ]
    for i in np.argsort(depth)[::-1]:  # far to near, nearer objects overwrite
        m = render_mask(objects[i][0], objects[i][1], intr)
        inst[m] = i + 1
    return inst, [inst == i + 1 for i in range(len(objects))]


def mask_to_bbox(mask: np.ndarray) -> np.ndarray | None:
    """Tight [x0, y0, x1, y1] (pixel edges, exclusive max) of a boolean mask, or None."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float64)


def projected_bbox(obj: ObjectModel, T_cam_model: np.ndarray, intr: CameraIntrinsics,
                   clip: bool = True) -> np.ndarray | None:
    """[x0, y0, x1, y1] of the projected model vertices (amodal box), or None if not in view."""
    if not np.all(np.isfinite(T_cam_model)):
        return None
    uv, ok = intr.project(transform_points(T_cam_model, obj.vertices))
    if not ok.any():
        return None
    uv = uv[ok]
    box = np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()])
    if clip:
        box = np.clip(box, 0, [intr.width, intr.height, intr.width, intr.height])
        if box[2] <= box[0] or box[3] <= box[1]:
            return None
    return box


def require_objects(dataset_objects: dict, names: list[str] | None) -> list[ObjectModel]:
    if not dataset_objects:
        raise MissingCalibrationError(
            "the dataset has no object models yet (models/objects.yaml + CAD mesh)"
        )
    names = names or sorted(dataset_objects)
    missing = [n for n in names if n not in dataset_objects]
    if missing:
        raise KeyError(f"unknown object(s) {missing}; have {sorted(dataset_objects)}")
    return [dataset_objects[n] for n in names]
