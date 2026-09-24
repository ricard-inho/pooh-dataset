"""BOP format (https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md).

One BOP scene per trajectory. Units follow BOP: millimetres for translations and models.
Images are undistorted by default because BOP assumes a pinhole camera.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from ..tasks import ObjectDetection
from ._common import image_ext, progress, raw_if_unchanged, write_image


def export_bop(task: ObjectDetection, out_dir: str | Path, split_name: str = "train") -> Path:
    """Export an :class:`~pooh_dataset.tasks.ObjectDetection` task (constructed with
    ``undistort=True``) to ``<out_dir>/<split_name>/<scene_id>/...`` plus ``models/``."""
    if any(task.view(task.dataset[t]).intrinsics.has_distortion for t in task.trajectories):
        raise ValueError("BOP assumes pinhole images: build the task with undistort=True")
    out = Path(out_dir)
    split_dir = out / split_name
    scenes: dict[str, dict] = {}
    for i in progress(range(len(task)), len(task), f"BOP {split_name}"):
        ref = task.refs[i]
        s = task[i]
        scene_id = task.trajectories.index(ref.trajectory)
        sc = scenes.setdefault(ref.trajectory, {"id": scene_id, "camera": {}, "gt": {}, "info": {}})
        scene_dir = split_dir / f"{scene_id:06d}"
        im_id = ref.index
        ext = image_ext(ref.camera)
        write_image(scene_dir / "rgb" / f"{im_id:06d}{ext}", s["image"], raw_if_unchanged(task, ref))
        sc["camera"][str(im_id)] = {"cam_K": s["K"].ravel().tolist(), "depth_scale": 1.0}
        gts, infos = [], []
        for k, (T, obj_id, box) in enumerate(zip(s["poses"], s["labels"], s["boxes"])):
            gts.append({
                "cam_R_m2c": T[:3, :3].ravel().tolist(),
                "cam_t_m2c": (T[:3, 3] * 1000.0).tolist(),
                "obj_id": int(obj_id),
            })
            x0, y0, x1, y1 = box.tolist()
            infos.append({
                "bbox_visib": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
                "px_count_visib": int(s["masks"][k].sum()) if "masks" in s else -1,
            })
            if "masks" in s:
                p = scene_dir / "mask_visib" / f"{im_id:06d}_{k:06d}.png"
                p.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(s["masks"][k].astype(np.uint8) * 255).save(p)
        sc["gt"][str(im_id)] = gts
        sc["info"][str(im_id)] = infos

    for traj, sc in scenes.items():
        scene_dir = split_dir / f"{sc['id']:06d}"
        (scene_dir / "scene_camera.json").write_text(json.dumps(sc["camera"]))
        (scene_dir / "scene_gt.json").write_text(json.dumps(sc["gt"]))
        (scene_dir / "scene_gt_info.json").write_text(json.dumps(sc["info"]))
        (scene_dir / "source_trajectory.txt").write_text(traj + "\n")

    _export_models(task, out / "models")
    return out


def _export_models(task: ObjectDetection, models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    info = {}
    for obj in task.objects:
        v = obj.vertices * 1000.0
        lo, hi = v.min(0), v.max(0)
        entry = {
            "diameter": obj.diameter * 1000.0,
            "min_x": lo[0], "min_y": lo[1], "min_z": lo[2],
            "size_x": hi[0] - lo[0], "size_y": hi[1] - lo[1], "size_z": hi[2] - lo[2],
        }  # fmt: skip
        if obj.symmetries:
            entry.update(obj.symmetries)
        info[str(obj.class_id)] = {k: (float(x) if isinstance(x, np.floating) else x) for k, x in entry.items()}
        _write_ply(models_dir / f"obj_{obj.class_id:06d}.ply", v, obj.faces)
        if obj.mesh_path is not None and obj.mesh_path.exists():
            shutil.copy(obj.mesh_path, models_dir / f"obj_{obj.class_id:06d}_source{obj.mesh_path.suffix}")
    (models_dir / "models_info.json").write_text(json.dumps(info, indent=1))


def _write_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    lines = [
        "ply", "format ascii 1.0", f"element vertex {len(vertices)}",
        "property float x", "property float y", "property float z",
        f"element face {len(faces)}", "property list uchar int vertex_indices", "end_header",
    ]  # fmt: skip
    lines += [f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices]
    lines += [f"3 {a} {b} {c}" for a, b, c in faces]
    path.write_text("\n".join(lines) + "\n")
