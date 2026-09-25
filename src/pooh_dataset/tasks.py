"""Framework-agnostic task datasets. Each item is a dict of numpy arrays; wrap with
:mod:`pooh_dataset.torch` for PyTorch or iterate directly.

All tasks split by trajectory (see :mod:`pooh_dataset.splits`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dataset import PoohDataset
from .events import voxel_grid
from .flow import optical_flow
from .geometry import invert_pose, transform_points
from .imaging import CameraView
from .labels import (
    camera_pose,
    mask_to_bbox,
    object_pose,
    object_pose_in_camera,
    render_instances,
    require_objects,
)
from .objects import ObjectModel
from .trajectory import Trajectory

__all__ = ["FrameRef", "PoseEstimation", "ObjectDetection", "Segmentation", "OpticalFlow"]


@dataclass(frozen=True)
class FrameRef:
    trajectory: str
    camera: str
    index: int  # frame row index (for events: index of the reference timestamp)
    t_ns: int


class _FrameTask:
    needs_objects = True

    def __init__(
        self,
        dataset: PoohDataset,
        split: str | list[str] = "train",
        camera: str = "firefly",
        objects: list[str] | None = None,
        stride: int = 1,
        undistort: bool = False,
        image_size: tuple[int, int] | None = None,
        require_visible: bool = True,
        domain: str | None = None,
        max_gap_ns: int = 50_000_000,
    ):
        self.dataset = dataset
        self.camera = camera
        self.stride = stride
        self.undistort = undistort
        self.image_size = image_size
        self.max_gap_ns = max_gap_ns
        self.trajectories = [t for t in dataset.split(split, domain) if dataset[t].has(self._index_modality)]
        if self.needs_objects:
            self.objects: list[ObjectModel] = require_objects(dataset.objects, objects)
        else:
            self.objects = [dataset.objects[n] for n in (objects or sorted(dataset.objects))]
        self._views: dict[str, CameraView] = {}
        self.refs = [r for t in self.trajectories for r in self._index_trajectory(dataset[t], require_visible)]

    @property
    def _index_modality(self) -> str:
        return self.camera

    def __len__(self) -> int:
        return len(self.refs)

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def view(self, traj: Trajectory) -> CameraView:
        if traj.name not in self._views:
            intr = traj.calibration.intrinsics(self.camera)
            self._views[traj.name] = CameraView(intr, self.undistort, self.image_size)
        return self._views[traj.name]

    def objects_in(self, traj: Trajectory) -> list[ObjectModel]:
        """The task's objects that were present (listed and tracked) in ``traj``."""
        bodies = set(traj.mocap_bodies) if traj.has("mocap") else set()
        return [o for o in self.objects if o.present_in(traj.name) and o.mocap_body in bodies]

    def _index_trajectory(self, traj: Trajectory, require_visible: bool) -> list[FrameRef]:
        ts = traj.timestamps(self._index_modality)
        idx = np.arange(0, len(ts), self.stride)
        if require_visible and self.needs_objects and not self.objects_in(traj):
            return []  # none of the requested objects were in this recording
        if require_visible and self.objects_in(traj):
            keep = np.zeros(len(idx), bool)
            intr = self.view(traj).intrinsics
            T_wc, ok_c = camera_pose(traj, self.camera, ts[idx], self.max_gap_ns)
            T_cw = invert_pose(T_wc)
            for obj in self.objects_in(traj):
                T_wo, ok_o = object_pose(traj, obj, ts[idx], self.max_gap_ns)
                T_co = T_cw @ T_wo
                for k in np.nonzero(ok_c & ok_o)[0]:
                    uv, ok = intr.project(transform_points(T_co[k], obj.corners))
                    if ok.any() and _overlaps(uv[ok], intr.width, intr.height):
                        keep[k] = True
            idx = idx[keep]
        return [FrameRef(traj.name, self.camera, int(i), int(ts[i])) for i in idx]

    def _base(self, ref: FrameRef) -> tuple[Trajectory, dict]:
        traj = self.dataset[ref.trajectory]
        view = self.view(traj)
        sample = {
            "image": view(traj.frame(ref.camera, ref.index)),
            "K": view.intrinsics.K.copy(),
            "D": view.intrinsics.D.copy(),
            "trajectory": ref.trajectory,
            "camera": ref.camera,
            "index": ref.index,
            "t_ns": ref.t_ns,
        }
        return traj, sample

    def _poses(self, traj: Trajectory, ref: FrameRef):
        T, ok = [], []
        for obj in self.objects_in(traj):
            T_co, v = object_pose_in_camera(traj, ref.camera, obj, ref.t_ns, self.max_gap_ns)
            T.append(T_co)
            ok.append(bool(v))
        return np.stack(T) if T else np.zeros((0, 4, 4)), np.asarray(ok, bool)


def _overlaps(uv: np.ndarray, w: int, h: int) -> bool:
    return uv[:, 0].max() > 0 and uv[:, 0].min() < w and uv[:, 1].max() > 0 and uv[:, 1].min() < h


class PoseEstimation(_FrameTask):
    """6D object pose: image + intrinsics + T_cam_model per object.

    Item keys: ``image`` (H, W, 3) uint8, ``K`` (3, 3), ``D`` (5,), ``poses`` (N, 4, 4)
    (metres), ``object_ids`` (N,), ``names``, ``valid`` (N,) (mocap tracked), plus
    ``trajectory`` / ``index`` / ``t_ns`` for traceability.
    """

    def __getitem__(self, i: int) -> dict:
        ref = self.refs[i]
        traj, s = self._base(ref)
        s["poses"], s["valid"] = self._poses(traj, ref)
        objs = self.objects_in(traj)
        s["object_ids"] = np.array([o.class_id for o in objs], np.int64)
        s["names"] = [o.name for o in objs]
        return s


class ObjectDetection(_FrameTask):
    """Boxes (+ instance masks) of the tracked objects, rendered from mocap + CAD.

    Item keys: ``image``, ``boxes`` (N, 4) xyxy of the *visible* part, ``labels`` (N,)
    class ids, ``masks`` (N, H, W) bool (if ``masks=True``), ``poses`` (N, 4, 4), ``names``.
    Objects that are occluded / out of view are dropped, so N varies per frame.
    """

    def __init__(self, *args, masks: bool = True, min_visible_px: int = 20, **kw):
        self.with_masks = masks
        self.min_visible_px = min_visible_px
        super().__init__(*args, **kw)

    def _labels(self, traj: Trajectory, ref: FrameRef, s: dict) -> dict:
        poses, valid = self._poses(traj, ref)
        intr = self.view(traj).intrinsics
        items = [(o, T) for o, T, v in zip(self.objects_in(traj), poses, valid) if v]
        inst, vis = render_instances(items, intr)
        boxes, labels, masks, names, kept_poses = [], [], [], [], []
        for (obj, T), m in zip(items, vis):
            if m.sum() < self.min_visible_px:
                continue
            boxes.append(mask_to_bbox(m))
            labels.append(obj.class_id)
            masks.append(m)
            names.append(obj.name)
            kept_poses.append(T)
        s["boxes"] = np.array(boxes, np.float32).reshape(-1, 4)
        s["labels"] = np.array(labels, np.int64)
        s["names"] = names
        s["poses"] = np.array(kept_poses).reshape(-1, 4, 4)
        s["_instance_map"] = inst
        if self.with_masks:
            s["masks"] = np.array(masks, bool).reshape(-1, intr.height, intr.width)
        return s

    def __getitem__(self, i: int) -> dict:
        ref = self.refs[i]
        traj, s = self._base(ref)
        s = self._labels(traj, ref, s)
        s.pop("_instance_map")
        return s


class Segmentation(ObjectDetection):
    """Semantic + instance segmentation. Item keys: ``image``, ``semantic`` (H, W) int64
    (0 = background, else class id), ``instances`` (H, W) int64 (0 = background, else
    1..N matching ``labels``), ``labels``, ``names``, ``boxes``."""

    def __getitem__(self, i: int) -> dict:
        ref = self.refs[i]
        traj, s = self._base(ref)
        s = self._labels(traj, ref, s)
        s.pop("_instance_map")
        masks = s.pop("masks") if self.with_masks else None
        h, w = s["image"].shape[:2]
        sem = np.zeros((h, w), np.int64)
        inst = np.zeros((h, w), np.int64)
        if masks is not None:
            for k, (m, c) in enumerate(zip(masks, s["labels"]), start=1):
                sem[m] = c
                inst[m] = k
        s["semantic"], s["instances"] = sem, inst
        return s


class OpticalFlow(_FrameTask):
    """Dense optical-flow GT from RealSense depth + mocap (static scene + tracked objects).

    * Frame cameras (``firefly``, ``realsense_color``, ``realsense_depth``): pairs of frames
      ``(i, i + frame_gap)``; item keys ``image0``, ``image1``, ``flow`` (H, W, 2), ``valid``.
    * ``camera="events"``: windows ``[t0, t0 + window_ns)`` starting at every depth frame;
      item keys ``events`` (voxel grid, ``event_bins`` x H x W), ``flow``, ``valid``,
      ``t0_ns``, ``t1_ns``.
    """

    needs_objects = False

    def __init__(self, dataset: PoohDataset, split="train", camera: str = "realsense_color",
                 objects: list[str] | None = None, frame_gap: int = 1,
                 window_ns: int = 50_000_000, event_bins: int = 5, **kw):
        if camera == "events" and (kw.get("undistort") or kw.get("image_size")):
            raise ValueError("undistort / image_size are not supported for the event camera")
        self.frame_gap = frame_gap
        self.window_ns = window_ns
        self.event_bins = event_bins
        kw.setdefault("require_visible", False)
        super().__init__(dataset, split, camera, objects, **kw)

    @property
    def _index_modality(self) -> str:
        return "realsense_depth" if self.camera == "events" else self.camera

    def _index_trajectory(self, traj: Trajectory, require_visible: bool) -> list[FrameRef]:
        refs = super()._index_trajectory(traj, require_visible)
        if self.camera == "events":
            lo, hi = traj.time_range("events")
            return [r for r in refs if lo <= r.t_ns and r.t_ns + self.window_ns <= hi]
        n = len(traj.timestamps(self.camera))
        return [r for r in refs if r.index + self.frame_gap < n]

    def __getitem__(self, i: int) -> dict:
        ref = self.refs[i]
        traj = self.dataset[ref.trajectory]
        view = self.view(traj)
        if self.camera == "events":
            t0, t1 = ref.t_ns, ref.t_ns + self.window_ns
            intr = view.intrinsics
            ev = traj.events(t0, t1)
            s = {"events": voxel_grid(ev, self.event_bins, intr.height, intr.width, t0, t1)}
            depth_index = ref.index
        else:
            j = ref.index + self.frame_gap
            t0, t1 = ref.t_ns, int(traj.timestamps(self.camera)[j])
            nearest = self.camera == "realsense_depth"
            s = {
                "image0": view(traj.frame(self.camera, ref.index), nearest),
                "image1": view(traj.frame(self.camera, j), nearest),
            }
            depth_index = ref.index if self.camera == "realsense_depth" else None
        flow, valid = optical_flow(traj, self.camera, t0, t1, self.objects_in(traj),
                                   depth_index=depth_index, intr=view.intrinsics)
        s.update(flow=flow, valid=valid, K=view.intrinsics.K.copy(), trajectory=ref.trajectory,
                 camera=self.camera, t0_ns=t0, t1_ns=t1)
        return s
