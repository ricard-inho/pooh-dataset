from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def write_image(path: Path, image: np.ndarray, raw_bytes: bytes | None = None, quality: int = 95) -> None:
    """Write a frame; when it was not resampled, the original encoded bytes are copied
    as-is (lossless and fast)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if raw_bytes is not None:
        path.write_bytes(raw_bytes)
        return
    img = Image.fromarray(image)
    if path.suffix.lower() in (".jpg", ".jpeg"):
        img.save(path, quality=quality)
    else:
        img.save(path)


def image_ext(camera: str) -> str:
    return ".png" if camera == "realsense_depth" else ".jpg"


def raw_if_unchanged(task, ref) -> bytes | None:
    traj = task.dataset[ref.trajectory]
    return traj.frame_bytes(ref.camera, ref.index) if task.view(traj).is_identity else None


def progress(it, total: int, desc: str):
    try:
        from tqdm import tqdm

        return tqdm(it, total=total, desc=desc)
    except ImportError:  # pragma: no cover
        return it
