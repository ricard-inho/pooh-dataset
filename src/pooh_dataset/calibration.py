"""Camera intrinsics, rig extrinsics and per-sensor time offsets.

Calibration is read from ``calibration/rig.yaml`` (shared defaults, optional) merged with
``calibration/<trajectory>.yaml`` (per-trajectory override). See ``DATA_REQUIREMENTS.md``
for the full schema.

Simulated trajectories have their own format (no rig.yaml): each sensor has
``T_base_sensor`` (sensor frame x forward, y left, z up, in the robot base frame). It is read
as extrinsics on the ``robot`` mocap body, with cameras turned into their optical frame
(z forward, x right, y down), and depth ``scale_m`` as ``depth_scale``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .geometry import make_pose, max_monotonic_radius, project_points

__all__ = ["CameraIntrinsics", "Calibration", "MissingCalibrationError", "parse_transform"]

REQUIREMENTS_HINT = (
    "See https://github.com/ricard-inho/pooh-dataset/blob/main/DATA_REQUIREMENTS.md for the calibration "
    "files the dataset needs to provide, and run `pooh check` to see what is missing."
)
RESERVED_KEYS = {"extrinsics", "time_offsets_ns"}
#: mocap body the simulated sensors are mounted on
SIM_REFERENCE_BODY = "robot"
#: optical frame (z forward, x right, y down) in a sensor frame (x forward, y left, z up)
T_SENSOR_OPTICAL = np.array([[0.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0]])


class MissingCalibrationError(RuntimeError):
    """Raised when a label needs calibration (intrinsics / extrinsics / models) the dataset
    does not provide yet."""

    def __init__(self, msg: str):
        super().__init__(f"{msg}\n{REQUIREMENTS_HINT}")


def parse_transform(value: Any) -> np.ndarray:
    """Parse a 4x4 transform given as 16 row-major numbers, a nested 4x4 list, or a mapping
    ``{translation: [x, y, z], rotation_xyzw: [qx, qy, qz, qw]}`` (``translation_m`` also
    accepted)."""
    if isinstance(value, dict):
        t = value.get("translation", value.get("translation_m", [0.0, 0.0, 0.0]))
        q = value.get("rotation_xyzw", value.get("quaternion_xyzw", [0.0, 0.0, 0.0, 1.0]))
        return make_pose(np.asarray(t, float), np.asarray(q, float))
    T = np.asarray(value, dtype=np.float64).reshape(4, 4)
    if not np.allclose(T[3], [0, 0, 0, 1]):
        raise ValueError(f"not a homogeneous transform: last row is {T[3]}")
    return T


@dataclass(frozen=True, eq=False)
class CameraIntrinsics:
    name: str
    width: int
    height: int
    K: np.ndarray
    D: np.ndarray = field(default_factory=lambda: np.zeros(5))
    distortion_model: str = "plumb_bob"
    source: str = "unknown"
    frame_id: str | None = None
    depth_scale: float | None = None  # metres per raw unit, depth cameras only

    @classmethod
    def from_dict(cls, name: str, d: dict) -> CameraIntrinsics:
        D = np.asarray(d.get("D") or [0.0] * 5, dtype=np.float64)
        model = d.get("distortion_model", "plumb_bob")
        if model not in ("plumb_bob", "rational_polynomial", "radtan", "none", "") and np.any(D):
            raise ValueError(f"{name}: unsupported distortion model {model!r}")
        return cls(
            name=name,
            width=int(d["width"]),
            height=int(d["height"]),
            K=np.asarray(d["K"], dtype=np.float64).reshape(3, 3),
            D=D,
            distortion_model=model,
            source=str(d.get("source", "unknown")),
            frame_id=d.get("frame_id"),
            depth_scale=d.get("depth_scale", d.get("scale_m")),
        )

    @property
    def size(self) -> tuple[int, int]:
        """(width, height)"""
        return self.width, self.height

    @property
    def is_supplement(self) -> bool:
        """True when the intrinsics did not come from the driver's live camera_info."""
        return self.source.startswith("supplement")

    @property
    def has_distortion(self) -> bool:
        return bool(np.any(self.D))

    def project(self, pts_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return project_points(pts_cam, self.K, self.D, r_max=self._r_max)

    @cached_property
    def _r_max(self) -> float:
        return max_monotonic_radius(self.D)

    def resized(self, width: int, height: int) -> CameraIntrinsics:
        """Intrinsics for the same camera after resizing its images to (width, height)."""
        sx, sy = width / self.width, height / self.height
        K = self.K.copy()
        K[0, :] *= sx
        K[1, :] *= sy
        # Pixel-centre convention: (u + 0.5) * s - 0.5
        K[0, 2] = (self.K[0, 2] + 0.5) * sx - 0.5
        K[1, 2] = (self.K[1, 2] + 0.5) * sy - 0.5
        return replace(self, width=width, height=height, K=K)

    def undistorted(self) -> CameraIntrinsics:
        """Pinhole intrinsics of the undistorted image (same K, zero distortion)."""
        return replace(self, D=np.zeros(5), distortion_model="none")


@dataclass
class Calibration:
    trajectory: str
    cameras: dict[str, CameraIntrinsics] = field(default_factory=dict)
    #: camera name -> T_body_cam (pose of the camera optical frame in the reference body frame)
    extrinsics: dict[str, np.ndarray] = field(default_factory=dict)
    #: mocap rigid body the cameras are attached to (e.g. "SensorStack")
    reference_body: str | None = None
    #: sensor name -> offset added to that sensor's timestamps to land on the host clock
    time_offsets_ns: dict[str, int] = field(default_factory=dict)

    @classmethod
    def load(cls, calibration_dir: Path, trajectory: str) -> Calibration:
        calibration_dir = Path(calibration_dir)
        own_path = calibration_dir / f"{trajectory}.yaml"
        own = (yaml.safe_load(own_path.read_text()) or {}) if own_path.exists() else {}
        merged: dict[str, Any] = {}
        rig = calibration_dir / "rig.yaml"
        if rig.exists() and not _is_sim(own):  # the real rig's defaults
            _deep_merge(merged, yaml.safe_load(rig.read_text()) or {})
        _deep_merge(merged, own)
        return cls.from_dict(trajectory, merged)

    @classmethod
    def from_dict(cls, trajectory: str, data: dict) -> Calibration:
        cameras = {
            k: CameraIntrinsics.from_dict(k, v)
            for k, v in data.items()
            if k not in RESERVED_KEYS and isinstance(v, dict) and "K" in v
        }
        ext = data.get("extrinsics") or {}
        extrinsics = {
            name: parse_transform(v["T_body_cam"] if isinstance(v, dict) and "T_body_cam" in v else v)
            for name, v in (ext.get("cameras") or {}).items()
        }
        if not ext and _is_sim(data):
            ext = {"reference_body": SIM_REFERENCE_BODY}
            for name, v in data.items():
                if isinstance(v, dict) and "T_base_sensor" in v:
                    T = parse_transform(v["T_base_sensor"])
                    extrinsics[name] = T @ T_SENSOR_OPTICAL if name in cameras else T
        return cls(
            trajectory=trajectory,
            cameras=cameras,
            extrinsics=extrinsics,
            reference_body=ext.get("reference_body"),
            time_offsets_ns={k: int(v) for k, v in (data.get("time_offsets_ns") or {}).items()},
        )

    def intrinsics(self, camera: str) -> CameraIntrinsics:
        try:
            return self.cameras[camera]
        except KeyError:
            raise MissingCalibrationError(
                f"[{self.trajectory}] no intrinsics for camera {camera!r} "
                f"(available: {sorted(self.cameras)})"
            ) from None

    def T_body_cam(self, camera: str) -> np.ndarray:
        if self.reference_body is None or camera not in self.extrinsics:
            raise MissingCalibrationError(
                f"[{self.trajectory}] no extrinsics for camera {camera!r}: need "
                "`extrinsics.reference_body` and `extrinsics.cameras.<camera>.T_body_cam`"
            )
        return self.extrinsics[camera]

    def T_cam_cam(self, dst: str, src: str) -> np.ndarray:
        """Transform from camera ``src`` optical frame to camera ``dst`` optical frame."""
        return np.linalg.solve(self.T_body_cam(dst), self.T_body_cam(src))

    def time_offset_ns(self, sensor: str) -> int:
        return self.time_offsets_ns.get(sensor, 0)


def _is_sim(data: dict) -> bool:
    """Calibration written by the simulation pipeline (sensor mounts as ``T_base_sensor``)."""
    return any(isinstance(v, dict) and "T_base_sensor" in v for v in data.values())


def _deep_merge(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst
