"""Ultralytics YOLO detection format."""

from __future__ import annotations

from pathlib import Path

from ..tasks import ObjectDetection
from ._common import image_ext, progress, raw_if_unchanged, write_image


def export_yolo(task: ObjectDetection, out_dir: str | Path, split_name: str = "train") -> Path:
    """Writes ``images/<split>/``, ``labels/<split>/`` (``cls cx cy w h`` normalised,
    ``cls = class_id - 1``) and a ``data.yaml`` listing the classes."""
    out = Path(out_dir)
    for i in progress(range(len(task)), len(task), f"YOLO {split_name}"):
        ref = task.refs[i]
        s = task[i]
        stem = f"{ref.trajectory}_{ref.camera}_{ref.index:06d}"
        write_image(out / "images" / split_name / f"{stem}{image_ext(ref.camera)}", s["image"],
                    raw_if_unchanged(task, ref))
        h, w = s["image"].shape[:2]
        lines = []
        for (x0, y0, x1, y1), label in zip(s["boxes"].tolist(), s["labels"].tolist()):
            lines.append(f"{label - 1} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} "
                         f"{(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
        lbl = out / "labels" / split_name / f"{stem}.txt"
        lbl.parent.mkdir(parents=True, exist_ok=True)
        lbl.write_text("\n".join(lines) + ("\n" if lines else ""))

    names = {o.class_id - 1: o.name for o in sorted(task.objects, key=lambda o: o.class_id)}
    data_yaml = out / "data.yaml"
    splits = sorted({p.name for p in (out / "images").iterdir() if p.is_dir()})
    text = [f"path: {out.resolve()}"]
    text += [f"{sp}: images/{sp}" for sp in splits]
    text += ["names:"] + [f"  {k}: {v}" for k, v in names.items()]
    data_yaml.write_text("\n".join(text) + "\n")
    return out
