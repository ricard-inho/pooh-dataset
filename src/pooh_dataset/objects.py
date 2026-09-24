"""Tracked object models (CAD mesh or box) and the mocap-marker -> model transform.

Read from ``models/objects.yaml`` in the dataset; see ``DATA_REQUIREMENTS.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .calibration import parse_transform

__all__ = ["ObjectModel", "load_object_models"]


@dataclass(eq=False)
class ObjectModel:
    name: str
    #: 1-based class / object id (BOP obj_id, COCO category_id; YOLO uses class_id - 1)
    class_id: int
    #: mocap rigid-body name tracking this object
    mocap_body: str
    #: vertices (N, 3) in metres, model frame
    vertices: np.ndarray
    #: triangle indices (M, 3)
    faces: np.ndarray
    #: pose of the model frame in the mocap rigid-body (marker) frame
    T_body_model: np.ndarray = field(default_factory=lambda: np.eye(4))
    mesh_path: Path | None = None
    #: BOP-style symmetries, passed through to exports untouched
    symmetries: dict | None = None

    @classmethod
    def from_box(cls, name: str, size_xyz, class_id: int = 1, mocap_body: str | None = None,
                 T_body_model=None) -> ObjectModel:
        """Axis-aligned box centred on the model origin; ``size_xyz`` in metres."""
        sx, sy, sz = (float(s) / 2 for s in size_xyz)
        v = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)])
        f = np.array([
            [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],  # -x, +x
            [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],  # -y, +y
            [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],  # -z, +z
        ])  # fmt: skip
        return cls(name, class_id, mocap_body or name, v, f,
                   np.eye(4) if T_body_model is None else np.asarray(T_body_model, float))

    @classmethod
    def from_mesh_file(cls, name: str, path: Path, scale: float = 1.0, **kw) -> ObjectModel:
        try:
            import trimesh
        except ImportError as e:  # pragma: no cover
            raise ImportError("Loading CAD meshes needs `pip install pooh-dataset[mesh]`") from e
        mesh = trimesh.load(path, force="mesh", process=False)
        return cls(name=name, vertices=np.asarray(mesh.vertices, float) * scale,
                   faces=np.asarray(mesh.faces, np.int64), mesh_path=Path(path), **kw)

    @property
    def corners(self) -> np.ndarray:
        """(8, 3) corners of the model's axis-aligned bounding box."""
        lo, hi = self.vertices.min(0), self.vertices.max(0)
        return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])

    @property
    def diameter(self) -> float:
        """Max distance between two vertices (approximated on the bbox for big meshes)."""
        pts = self.vertices if len(self.vertices) <= 2000 else self.corners
        d = pts[:, None, :] - pts[None, :, :]
        return float(np.sqrt((d**2).sum(-1)).max())


def load_object_models(models_dir: Path) -> dict[str, ObjectModel]:
    """Parse ``models/objects.yaml``. Returns {} when the file is not there."""
    models_dir = Path(models_dir)
    cfg_path = models_dir / "objects.yaml"
    if not cfg_path.exists():
        return {}
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    out: dict[str, ObjectModel] = {}
    for i, (name, d) in enumerate(sorted(cfg.items()), start=1):
        common = dict(
            class_id=int(d.get("class_id", i)),
            mocap_body=d.get("mocap_body", name),
            T_body_model=parse_transform(d["T_body_model"]) if "T_body_model" in d else np.eye(4),
        )
        if "mesh" in d:
            m = ObjectModel.from_mesh_file(name, models_dir.parent / d["mesh"],
                                           scale=float(d.get("mesh_scale", 1.0)), **common)
        elif "box" in d:
            m = ObjectModel.from_box(name, d["box"], **common)
        else:
            raise ValueError(f"models/objects.yaml: {name!r} needs either `mesh` or `box`")
        m.symmetries = d.get("symmetries")
        out[name] = m
    return out
