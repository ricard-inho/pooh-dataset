"""Event-camera containers and dense representations (histogram, voxel grid, time surface)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Events", "EVENT_SENSOR_SIZE", "event_histogram", "voxel_grid", "time_surface"]

#: Prophesee EVK4 / IMX636 resolution as (height, width)
EVENT_SENSOR_SIZE = (720, 1280)


@dataclass
class Events:
    """A slice of the event stream. ``t`` is int64 ns on the host clock, ``p`` is 0/1."""

    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    p: np.ndarray

    def __len__(self) -> int:
        return len(self.t)

    def __getitem__(self, sel) -> Events:
        return Events(self.t[sel], self.x[sel], self.y[sel], self.p[sel])

    @classmethod
    def empty(cls) -> Events:
        return cls(
            np.empty(0, np.int64), np.empty(0, np.uint16), np.empty(0, np.uint16), np.empty(0, np.uint8)
        )


def _in_bounds(ev: Events, height: int, width: int) -> Events:
    keep = (ev.x < width) & (ev.y < height)
    return ev if keep.all() else ev[keep]


def event_histogram(ev: Events, height: int = EVENT_SENSOR_SIZE[0], width: int = EVENT_SENSOR_SIZE[1]) -> np.ndarray:
    """(2, H, W) float32 per-pixel event counts; channel 0 = negative, 1 = positive."""
    ev = _in_bounds(ev, height, width)
    out = np.zeros((2, height * width), dtype=np.float32)
    idx = ev.y.astype(np.int64) * width + ev.x.astype(np.int64)
    pol = (ev.p > 0).astype(np.int64)
    np.add.at(out, (pol, idx), 1.0)
    return out.reshape(2, height, width)


def voxel_grid(
    ev: Events,
    num_bins: int = 5,
    height: int = EVENT_SENSOR_SIZE[0],
    width: int = EVENT_SENSOR_SIZE[1],
    t_start_ns: int | None = None,
    t_end_ns: int | None = None,
) -> np.ndarray:
    """(num_bins, H, W) float32 voxel grid with bilinear interpolation in time and polarity
    +1/-1 (Zhu et al., 2019). The time span defaults to the events' own span."""
    ev = _in_bounds(ev, height, width)
    grid = np.zeros(num_bins * height * width, dtype=np.float32)
    if len(ev) == 0:
        return grid.reshape(num_bins, height, width)
    t0 = ev.t[0] if t_start_ns is None else t_start_ns
    t1 = ev.t[-1] if t_end_ns is None else t_end_ns
    span = max(int(t1) - int(t0), 1)
    tn = (ev.t - t0).astype(np.float64) * (num_bins - 1) / span
    pol = np.where(ev.p > 0, 1.0, -1.0).astype(np.float32)
    pix = ev.y.astype(np.int64) * width + ev.x.astype(np.int64)
    left = np.floor(tn).astype(np.int64)
    for b, w in ((left, 1.0 - (tn - left)), (left + 1, tn - left)):
        ok = (b >= 0) & (b < num_bins)
        np.add.at(grid, b[ok] * height * width + pix[ok], (pol[ok] * w[ok]).astype(np.float32))
    return grid.reshape(num_bins, height, width)


def time_surface(
    ev: Events,
    t_ref_ns: int,
    tau_ns: float = 50e6,
    height: int = EVENT_SENSOR_SIZE[0],
    width: int = EVENT_SENSOR_SIZE[1],
) -> np.ndarray:
    """(2, H, W) exponentially-decayed time surface of the latest event per pixel/polarity."""
    ev = _in_bounds(ev, height, width)
    last = np.full((2, height * width), -np.inf)
    idx = ev.y.astype(np.int64) * width + ev.x.astype(np.int64)
    pol = (ev.p > 0).astype(np.int64)
    # events are time-sorted, so a plain assignment keeps the latest timestamp per pixel
    last[pol, idx] = (ev.t - t_ref_ns).astype(np.float64)
    return np.exp(last / tau_ns).astype(np.float32).reshape(2, height, width)
