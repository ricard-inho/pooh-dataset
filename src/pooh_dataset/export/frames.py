"""Unpack frames from Parquet into individual image files (no re-encoding)."""

from __future__ import annotations

import csv
from pathlib import Path

from ..trajectory import Trajectory
from ._common import image_ext, progress


def extract_frames(traj: Trajectory, modality: str, out_dir: str | Path, stride: int = 1) -> Path:
    """Writes ``<out_dir>/<trajectory>/<modality>/<index>_<timestamp_ns>.<ext>`` plus a
    ``timestamps.csv`` (index, timestamp_ns, file)."""
    out = Path(out_dir) / traj.name / modality
    out.mkdir(parents=True, exist_ok=True)
    ts = traj.timestamps(modality)
    ext = image_ext(modality)
    rows = []
    idx = range(0, len(ts), stride)
    for i in progress(idx, len(idx), f"{traj.name}/{modality}"):
        name = f"{i:06d}_{ts[i]}{ext}"
        (out / name).write_bytes(traj.frame_bytes(modality, i))
        rows.append((i, int(ts[i]), name))
    with open(out / "timestamps.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "timestamp_ns", "file"])
        w.writerows(rows)
    return out
