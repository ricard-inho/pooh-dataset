"""COCO instances format (boxes + uncompressed-RLE masks)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..tasks import ObjectDetection
from ._common import image_ext, progress, raw_if_unchanged, write_image


def mask_to_rle(mask: np.ndarray) -> dict:
    """Uncompressed COCO RLE (column-major run lengths, starting with background)."""
    flat = np.asarray(mask, bool).ravel(order="F")
    change = np.nonzero(np.diff(flat.astype(np.int8)))[0] + 1
    bounds = np.concatenate([[0], change, [len(flat)]])
    counts = np.diff(bounds).tolist()
    if flat[0]:
        counts = [0] + counts
    return {"counts": counts, "size": list(mask.shape)}


def export_coco(task: ObjectDetection, out_dir: str | Path, split_name: str = "train") -> Path:
    """Writes ``<out_dir>/images/<split>/...`` and ``<out_dir>/annotations/instances_<split>.json``."""
    out = Path(out_dir)
    images, annotations = [], []
    for i in progress(range(len(task)), len(task), f"COCO {split_name}"):
        ref = task.refs[i]
        s = task[i]
        rel = Path("images") / split_name / ref.trajectory / f"{ref.camera}_{ref.index:06d}{image_ext(ref.camera)}"
        write_image(out / rel, s["image"], raw_if_unchanged(task, ref))
        h, w = s["image"].shape[:2]
        images.append({
            "id": i, "file_name": str(rel.relative_to("images")), "width": w, "height": h,
            "trajectory": ref.trajectory, "camera": ref.camera, "frame_index": ref.index,
            "timestamp_ns": ref.t_ns, "K": s["K"].ravel().tolist(),
        })  # fmt: skip
        for k, (box, label) in enumerate(zip(s["boxes"], s["labels"])):
            x0, y0, x1, y1 = box.tolist()
            ann = {
                "id": len(annotations), "image_id": i, "category_id": int(label),
                "bbox": [x0, y0, x1 - x0, y1 - y0], "iscrowd": 0,
                "area": float(s["masks"][k].sum()) if "masks" in s else (x1 - x0) * (y1 - y0),
                "pose_T_cam_model": s["poses"][k].tolist(),
            }  # fmt: skip
            if "masks" in s:
                ann["segmentation"] = mask_to_rle(s["masks"][k])
            annotations.append(ann)
    categories = [{"id": o.class_id, "name": o.name} for o in task.objects]
    ann_dir = out / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    (ann_dir / f"instances_{split_name}.json").write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": categories})
    )
    return out
