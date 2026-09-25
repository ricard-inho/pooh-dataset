"""Camera extrinsics + object-model alignment from clicked object keypoints.

Setup: the cameras are rigid on a mocap-tracked rig (``SensorStack``) and an object with a
CAD model is tracked by its own mocap body (``cubesat``). In a handful of frames a person
clicks the visible keypoints (e.g. cube corners): in any order, without saying which is
which. The solver finds

* ``T_body_cam``: the camera optical frame in the rig's mocap body frame, and
* a correction of ``T_body_model``: rotation about the object's vertical axis (not
  observable from mocap alone) plus a small translation,

by minimising the distance of every click to the nearest projected keypoint. A global
search over camera orientations provides the initial guess, so no prior extrinsics are
needed. Symmetric objects (a cube) are only recovered up to their symmetry, which does not
change masks or boxes.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .calibration import CameraIntrinsics
from .dataset import PoohDataset
from .geometry import (
    invert_pose,
    project_points,
    rotmat_to_quat,
    rotmat_to_rotvec,
    rotvec_to_rotmat,
    transform_points,
)
from .objects import ObjectModel

__all__ = [
    "Observation",
    "CalibrationResult",
    "solve",
    "select_frames",
    "observations_from_clicks",
    "write_outputs",
    "render_overlay",
    "align_markers",
    "read_motive_markers",
]

MISS_PX = 200.0  # residual for a click when no keypoint projects in front of the camera


@dataclass
class Observation:
    trajectory: str
    index: int
    t_ns: int
    points: np.ndarray  # (M, 2) clicked pixels
    T_world_body: np.ndarray  # rig body (e.g. SensorStack) at t
    T_world_obj: np.ndarray  # object's mocap body at t
    intr: CameraIntrinsics


@dataclass
class CalibrationResult:
    T_body_cam: np.ndarray
    T_body_model: np.ndarray
    rms_px: float
    per_frame_rms_px: list[float]
    click_errors_px: list[np.ndarray]
    yaw_correction_deg: float
    translation_correction_m: np.ndarray
    n_frames: int = 0
    tilt_correction_deg: float = 0.0
    n_clicks: int = 0
    notes: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ model


def _yaw_axis(obs: list[Observation], T_body_model: np.ndarray) -> np.ndarray:
    """Model axis closest to world up (the axis we rotate about)."""
    R = np.mean([(o.T_world_obj @ T_body_model)[:3, :3] for o in obs], axis=0)
    i = int(np.argmax(np.abs(R[2])))
    axis = np.zeros(3)
    axis[i] = np.sign(R[2, i])
    return axis


def _model_correction(yaw: float, d: np.ndarray, axis: np.ndarray, tilt=(0.0, 0.0)) -> np.ndarray:
    """Rotation about the model's vertical ``axis`` (yaw) plus about the two axes orthogonal
    to it (tilt), then a translation ``d`` (model frame)."""
    e1 = np.roll(np.abs(axis), 1)  # the two model axes orthogonal to the (signed unit) yaw axis
    e2 = np.cross(axis, e1)
    T = np.eye(4)
    T[:3, :3] = rotvec_to_rotmat(axis * yaw + e1 * tilt[0] + e2 * tilt[1])
    T[:3, 3] = d
    return T


N_PARAMS = 12  # camera rotvec (3) + t (3) | model yaw (1) + t (3) + tilt (2)


def _unpack(p: np.ndarray, T_model0: np.ndarray, axis: np.ndarray):
    T_bc = np.eye(4)
    T_bc[:3, :3] = rotvec_to_rotmat(p[:3])
    T_bc[:3, 3] = p[3:6]
    T_bm = T_model0 @ _model_correction(p[6], p[7:10], axis, p[10:12])
    return T_bc, T_bm


def predict(T_body_cam, T_body_model, o: Observation, keypoints: np.ndarray):
    """Projected keypoints (K, 2) and validity for one observation."""
    T_cam_model = invert_pose(T_body_cam) @ invert_pose(o.T_world_body) @ o.T_world_obj @ T_body_model
    return project_points(transform_points(T_cam_model, keypoints), o.intr.K, o.intr.D)


def _assign(uv, ok, clicks):
    if not ok.any():
        return None
    d = np.linalg.norm(clicks[:, None, :] - uv[None, ok, :], axis=-1)
    return np.nonzero(ok)[0][np.argmin(d, axis=1)]


def _residuals(p, obs, keypoints, T_model0, axis, assignment, prior_px_per_m):
    T_bc, T_bm = _unpack(p, T_model0, axis)
    res = []
    for o, a in zip(obs, assignment):
        uv, ok = predict(T_bc, T_bm, o, keypoints)
        if a is None or not ok[a].all():
            res.append(np.full(o.points.size, MISS_PX))
        else:
            res.append((uv[a] - o.points).ravel())
    res.append(p[7:10] * prior_px_per_m)
    return np.concatenate(res)


def _assignments(p, obs, keypoints, T_model0, axis):
    T_bc, T_bm = _unpack(p, T_model0, axis)
    return [_assign(*predict(T_bc, T_bm, o, keypoints), o.points) for o in obs]


def _lm(p, obs, keypoints, T_model0, axis, free, prior_px_per_m, iters=60):
    """Levenberg-Marquardt with nearest-keypoint re-assignment every iteration."""
    lam = 1e-2
    a = _assignments(p, obs, keypoints, T_model0, axis)
    r = _residuals(p, obs, keypoints, T_model0, axis, a, prior_px_per_m)
    cost = r @ r
    for _ in range(iters):
        J = np.zeros((len(r), len(free)))
        for j, k in enumerate(free):
            h = 1e-6 if k < 3 or k in (6, 10, 11) else 1e-5
            dp = np.zeros_like(p)
            dp[k] = h
            J[:, j] = (_residuals(p + dp, obs, keypoints, T_model0, axis, a, prior_px_per_m)
                       - _residuals(p - dp, obs, keypoints, T_model0, axis, a, prior_px_per_m)) / (2 * h)
        g = J.T @ r
        H = J.T @ J
        improved = False
        for _ in range(10):
            step = np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-9), -g)
            p_new = p.copy()
            p_new[free] += step
            a_new = _assignments(p_new, obs, keypoints, T_model0, axis)
            r_new = _residuals(p_new, obs, keypoints, T_model0, axis, a_new, prior_px_per_m)
            if r_new @ r_new < cost:
                p, a, r, cost = p_new, a_new, r_new, r_new @ r_new
                lam = max(lam / 3, 1e-7)
                improved = True
                break
            lam *= 5
        if not improved or np.linalg.norm(step) < 1e-9:
            break
    return p, cost


def _random_rotations(n: int, seed: int) -> np.ndarray:
    q = np.random.default_rng(seed).normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q.T
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], -1).reshape(n, 3, 3)  # fmt: skip


def _coarse_search(obs, keypoints, T_model0, axis, n_rot, seed, top_k):
    """Score random camera orientations (camera at the rig origin) x a yaw grid."""
    Rs = _random_rotations(n_rot, seed)
    candidates = []
    for yaw in np.radians([0.0, 22.5, 45.0, 67.5]):
        T_bm = T_model0 @ _model_correction(yaw, np.zeros(3), axis)
        score = np.zeros(n_rot)
        for o in obs:
            X_body = transform_points(invert_pose(o.T_world_body) @ o.T_world_obj @ T_bm, keypoints)
            X_cam = np.einsum("nji,kj->nki", Rs, X_body)  # R^T x for every rotation
            uv, ok = project_points(X_cam.reshape(-1, 3), o.intr.K, o.intr.D)
            uv = uv.reshape(n_rot, len(keypoints), 2)
            ok = ok.reshape(n_rot, len(keypoints))
            d = np.linalg.norm(o.points[None, :, None, :] - uv[:, None, :, :], axis=-1)
            d = np.where(ok[:, None, :], d, np.inf).min(axis=-1)
            score += np.minimum(d, MISS_PX).sum(axis=1)
        for i in np.argsort(score)[:top_k]:
            candidates.append((score[i], rotmat_to_rotvec(Rs[i]), yaw))
    candidates.sort(key=lambda c: c[0])
    return candidates[:top_k]


def solve(
    obs: list[Observation],
    obj: ObjectModel,
    refine_model: bool = True,
    refine_tilt: bool = False,
    n_rotations: int = 4000,
    top_k: int = 8,
    seed: int = 0,
    prior_px_per_m: float = 100.0,
) -> CalibrationResult:
    """Estimate T_body_cam (and refine obj.T_body_model) from clicked keypoints."""
    obs = [o for o in obs if len(o.points)]
    if len(obs) < 2:
        raise ValueError("need clicks in at least 2 frames (10 recommended)")
    keypoints = obj.click_points
    T_model0 = obj.T_body_model
    axis = _yaw_axis(obs, T_model0)
    best = None
    for _, rv, yaw in _coarse_search(obs, keypoints, T_model0, axis, n_rotations, seed, top_k):
        p = np.zeros(N_PARAMS)
        p[:3], p[6] = rv, yaw
        free = list(range(6)) + ([6] if refine_model else [])
        p, cost = _lm(p, obs, keypoints, T_model0, axis, free, prior_px_per_m)
        if refine_model:
            full = list(range(10)) + ([10, 11] if refine_tilt else [])
            p, cost = _lm(p, obs, keypoints, T_model0, axis, full, prior_px_per_m)
        if best is None or cost < best[1]:
            best = (p, cost)
    p = best[0]
    T_bc, T_bm = _unpack(p, T_model0, axis)
    errs, per_frame = [], []
    for o in obs:
        uv, ok = predict(T_bc, T_bm, o, keypoints)
        a = _assign(uv, ok, o.points)
        e = np.full(len(o.points), np.inf) if a is None else np.linalg.norm(uv[a] - o.points, axis=1)
        errs.append(e)
        per_frame.append(float(np.sqrt(np.mean(e**2))))
    all_e = np.concatenate(errs)
    res = CalibrationResult(
        T_body_cam=T_bc, T_body_model=T_bm, rms_px=float(np.sqrt(np.mean(all_e**2))),
        per_frame_rms_px=per_frame, click_errors_px=errs,
        yaw_correction_deg=float(np.degrees((p[6] + np.pi / 4) % (np.pi / 2) - np.pi / 4)),
        translation_correction_m=p[7:10].copy(), n_frames=len(obs), n_clicks=len(all_e),
        tilt_correction_deg=float(np.degrees(np.linalg.norm(p[10:12]))),
    )
    worst = int(np.argmax(per_frame))
    if per_frame[worst] > max(3 * np.median(per_frame), 5.0):
        o = obs[worst]
        res.notes.append(f"frame {o.trajectory}#{o.index} fits badly ({per_frame[worst]:.1f} px): "
                         "re-click it or drop it")
    if len({o.trajectory for o in obs}) == 1:
        res.notes.append("all frames come from one trajectory; clicking frames from several "
                         "trajectories makes the result more reliable")
    return res


# ------------------------------------------------------------- dataset io


def select_frames(ds: PoohDataset, camera: str, obj: ObjectModel, rig_body: str, n: int,
                  trajectories: list[str] | None = None, step: int = 3,
                  exclude: set[tuple[str, int]] = frozenset()):
    """Frames where rig and object are tracked, ordered for diversity of the rig-to-object
    pose (farthest-point sampling). Returns [(trajectory, index, t_ns), ...] (up to 3n).

    ``exclude`` = already-clicked (trajectory, index) frames: they are never returned, and new
    candidates are picked to be far from them."""
    cands, feats, seen = [], [], []
    for name in trajectories or ds.local_trajectory_names:
        traj = ds[name]
        if not (obj.present_in(name) and traj.has(camera) and traj.has("mocap")):
            continue
        if rig_body not in traj.mocap_bodies or obj.mocap_body not in traj.mocap_bodies:
            continue
        ts = traj.timestamps(camera)
        idx = np.arange(0, len(ts), step)
        idx = np.union1d(idx, [i for t, i in exclude if t == name and i < len(ts)]).astype(int)
        T_r, ok_r = traj.body_pose(rig_body, ts[idx])
        T_o, ok_o = traj.body_pose(obj.mocap_body, ts[idx])
        for k in np.nonzero(ok_r & ok_o)[0]:
            rel = invert_pose(T_r[k]) @ T_o[k]
            f = np.concatenate([rel[:3, 3], 0.3 * rotmat_to_rotvec(rel[:3, :3])])
            if (name, int(idx[k])) in exclude:
                seen.append(f)
            else:
                cands.append((name, int(idx[k]), int(ts[idx[k]])))
                feats.append(f)
    if not cands:
        return []
    feats = np.asarray(feats)
    if seen:  # start far from what was already clicked
        dist = np.min(np.linalg.norm(feats[:, None] - np.asarray(seen)[None], axis=-1), axis=1)
        order = []
    else:
        order = [0]
        dist = np.linalg.norm(feats - feats[0], axis=1)
    while len(order) < min(3 * n, len(cands)):
        i = int(np.argmax(dist))
        order.append(i)
        dist = np.minimum(dist, np.linalg.norm(feats - feats[i], axis=1))
        dist[order] = -1.0
    return [cands[i] for i in order]


def observations_from_clicks(ds: PoohDataset, clicks: dict, obj: ObjectModel) -> list[Observation]:
    obs = []
    rig_body, camera = clicks["rig_body"], clicks["camera"]
    for c in clicks["frames"]:
        if not c["points"]:
            continue
        traj = ds[c["trajectory"]]
        T_r, ok_r = traj.body_pose(rig_body, c["t_ns"])
        T_o, ok_o = traj.body_pose(obj.mocap_body, c["t_ns"])
        if not (ok_r and ok_o):
            continue
        obs.append(Observation(c["trajectory"], c["index"], c["t_ns"], np.asarray(c["points"], float),
                               T_r, T_o, traj.calibration.intrinsics(camera)))
    return obs


def _T_to_yaml(T: np.ndarray) -> dict:
    return {"translation": [round(float(x), 6) for x in T[:3, 3]],
            "rotation_xyzw": [round(float(x), 6) for x in rotmat_to_quat(T[:3, :3])]}


def write_outputs(ds: PoohDataset, res: CalibrationResult, camera: str, obj: ObjectModel,
                  rig_body: str, out_dir: Path, also: list[str] = (), refine_model: bool = True) -> list[Path]:
    """Write ``calibration/rig.yaml`` and ``models/objects.yaml`` under ``out_dir``, laid out
    like the Hub repo (so the folder can be uploaded as-is). Existing entries are kept."""
    out_dir = Path(out_dir)
    today = _dt.date.today().isoformat()
    written = []

    rig_path = out_dir / "calibration" / "rig.yaml"
    src = rig_path if rig_path.exists() else ds.root / "calibration" / "rig.yaml"
    rig = (yaml.safe_load(src.read_text()) or {}) if src.exists() else {}
    ext = rig.setdefault("extrinsics", {})
    ext["reference_body"] = rig_body
    cams = ext.setdefault("cameras", {})
    for cam in [camera, *also]:
        cams[cam] = {"T_body_cam": _T_to_yaml(res.T_body_cam),
                     "source": f"pooh calibrate {today}: {res.n_clicks} clicks in {res.n_frames} "
                               f"{camera} frames, RMS {res.rms_px:.2f} px"
                               + ("" if cam == camera else f" (copied from {camera})")}
    rig_path.parent.mkdir(parents=True, exist_ok=True)
    rig_path.write_text("# Rig extrinsics: T_body_cam = pose of each camera optical frame in the "
                        "reference mocap body frame.\n" + yaml.safe_dump(rig, sort_keys=False))
    written.append(rig_path)

    if refine_model:
        obj_path = out_dir / "models" / "objects.yaml"
        src = obj_path if obj_path.exists() else ds.root / "models" / "objects.yaml"
        objs = yaml.safe_load(src.read_text()) or {}
        entry = objs.setdefault(obj.name, {})
        entry["T_body_model"] = _T_to_yaml(res.T_body_model)
        entry["keypoints"] = np.round(obj.click_points, 6).tolist()
        header = (
            "# Object models for mask / 6D-pose ground truth.\n"
            "# T_body_model maps CAD (model) coords into the OptiTrack rigid body's frame:\n"
            "#   p_body = R(rotation_xyzw) @ p_model + translation      (metres)\n"
            f"# {obj.name}.T_body_model refined by `pooh calibrate` on {today} from {res.n_clicks} "
            f"clicked keypoints in {res.n_frames} {camera} frames (RMS {res.rms_px:.2f} px);\n"
            f"#   correction vs. the previous estimate: yaw {res.yaw_correction_deg:+.1f} deg, "
            f"translation {np.round(res.translation_correction_m * 1000, 1).tolist()} mm.\n"
        )
        obj_path.parent.mkdir(parents=True, exist_ok=True)
        obj_path.write_text(header + yaml.safe_dump(objs, sort_keys=False))
        written.append(obj_path)
    return written


def write_object_transform(ds: PoohDataset, obj_name: str, T_body_model: np.ndarray, out_dir: Path,
                           note: str, extra: dict | None = None) -> Path:
    """Update one object's ``T_body_model`` in ``<out_dir>/models/objects.yaml``."""
    obj_path = Path(out_dir) / "models" / "objects.yaml"
    src = obj_path if obj_path.exists() else ds.root / "models" / "objects.yaml"
    objs = yaml.safe_load(src.read_text()) or {}
    entry = objs.setdefault(obj_name, {})
    entry["T_body_model"] = _T_to_yaml(T_body_model)
    entry.update(extra or {})
    header = ("# Object models for mask / 6D-pose ground truth.\n"
              "# T_body_model maps CAD (model) coords into the OptiTrack rigid body's frame:\n"
              "#   p_body = R(rotation_xyzw) @ p_model + translation      (metres)\n"
              f"# {obj_name}.T_body_model: {note}\n")
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    obj_path.write_text(header + yaml.safe_dump(objs, sort_keys=False))
    return obj_path


def save_clicks(path: Path, clicks: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clicks, indent=1))


def load_clicks(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def render_overlay(image: np.ndarray, o: Observation, res: CalibrationResult, obj: ObjectModel) -> np.ndarray:
    """RGB image with the rendered mask outline (green), predicted keypoints (yellow) and the
    clicks (red)."""
    from PIL import Image, ImageDraw

    from .labels import render_mask

    T_cam_model = invert_pose(res.T_body_cam) @ invert_pose(o.T_world_body) @ o.T_world_obj @ res.T_body_model
    mask = render_mask(obj, T_cam_model, o.intr)
    inner = mask.copy()
    inner[1:] &= mask[:-1]
    inner[:-1] &= mask[1:]
    inner[:, 1:] &= mask[:, :-1]
    inner[:, :-1] &= mask[:, 1:]
    out = image.copy() if image.ndim == 3 else np.repeat(image[..., None], 3, -1)
    edge = mask & ~inner
    out[edge] = (0, 255, 0)
    img = Image.fromarray(out)
    draw = ImageDraw.Draw(img)
    r = max(2, o.intr.width // 200)
    uv, ok = predict(res.T_body_cam, res.T_body_model, o, obj.click_points)
    for (u, v), v_ok in zip(uv, ok):
        if v_ok:
            draw.line([(u - r, v), (u + r, v)], fill=(255, 220, 0), width=2)
            draw.line([(u, v - r), (u, v + r)], fill=(255, 220, 0), width=2)
    for u, v in o.points:
        draw.ellipse([u - r, v - r, u + r, v + r], outline=(255, 40, 40), width=2)
    return np.asarray(img)


# --------------------------------------------------- mocap markers -> CAD


#: Motive stores rigid bodies Y-up; the ROS mocap stream (Z-up) reports the same body with the
#: standard conversion (x, y, z)_ros = (x, -z, y)_motive. Verified on the cubesat body (markers on
#: its bottom face): the fit then puts the cube above its markers, tilted ~23 deg as in the lab.
MOTIVE_TO_ROS_BODY = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)


def read_motive_markers(path: Path) -> np.ndarray:
    """(N, 3) marker positions (metres, body frame, relative to the pivot) from a Motive
    rigid-body export (``.motive`` XML)."""
    import re

    txt = Path(path).read_text()
    pos = re.findall(r"<position>([^<]+)</position>", txt)
    if not pos:
        raise ValueError(f"{path}: no <position> entries (is this a Motive rigid body export?)")
    return np.array([[float(v) for v in p.split(",")] for p in pos])


def _kabsch(A: np.ndarray, B: np.ndarray):
    """Proper rotation R and translation t minimising |R A + t - B|."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def align_markers(motive_markers: np.ndarray, cad_markers: np.ndarray,
                  motive_to_body: np.ndarray = MOTIVE_TO_ROS_BODY):
    """T_body_model from marker positions known in both the mocap body frame and the CAD.

    ``cad_markers`` (K, 3) may be fewer than the Motive markers (e.g. only the corner LEDs
    were measured): the best-fitting subset and correspondence is searched. Returns
    ``(T_body_model, residuals_m, motive_indices)``.
    """
    import itertools

    motive_markers = np.asarray(motive_markers, float)
    cad_markers = np.asarray(cad_markers, float)
    k = len(cad_markers)
    if k < 3:
        raise ValueError("need at least 3 CAD marker positions")
    best = None
    for subset in itertools.combinations(range(len(motive_markers)), k):
        for perm in itertools.permutations(subset):
            B = motive_markers[list(perm)]
            R, t = _kabsch(cad_markers, B)
            r = np.linalg.norm(cad_markers @ R.T + t - B, axis=1)
            if best is None or r.max() < best[0].max():
                best = (r, perm, R, t)
    r, perm, R, t = best
    T = np.eye(4)
    T[:3, :3] = motive_to_body @ R
    T[:3, 3] = motive_to_body @ t
    return T, r, list(perm)
