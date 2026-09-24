"""Undistortion and resizing that keep images and intrinsics consistent (no OpenCV needed)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .calibration import CameraIntrinsics
from .geometry import distort_normalized

__all__ = ["CameraView"]


class CameraView:
    """Maps raw frames of a camera to the images a model is trained on.

    ``undistort=True`` removes lens distortion (output intrinsics have D = 0; required for
    BOP export and most pose-estimation methods). ``size=(W, H)`` resizes. Both are done in a
    single resampling pass, and :attr:`intrinsics` describes the output images.
    """

    def __init__(self, intr: CameraIntrinsics, undistort: bool = False,
                 size: tuple[int, int] | None = None):
        self.src = intr
        self.undistort = undistort and intr.has_distortion
        w, h = size or intr.size
        out = intr.resized(w, h) if (w, h) != intr.size else intr
        self.intrinsics = out.undistorted() if undistort else out

    @property
    def is_identity(self) -> bool:
        return not self.undistort and self.intrinsics.size == self.src.size

    def __call__(self, image: np.ndarray, nearest: bool = False) -> np.ndarray:
        """Apply to an (H, W[, C]) frame. Use ``nearest=True`` for depth / label images."""
        if self.is_identity:
            return image
        if not self.undistort:
            mode = Image.NEAREST if nearest else Image.BILINEAR
            return np.asarray(Image.fromarray(image).resize(self.intrinsics.size, mode))
        mx, my = _undistort_maps(self.src, self.intrinsics)
        return _remap(image, mx, my, nearest)


def _intr_key(i: CameraIntrinsics):
    return (i.width, i.height, tuple(i.K.ravel()), tuple(i.D.ravel()))


_MAP_CACHE: dict = {}


def _undistort_maps(src: CameraIntrinsics, dst: CameraIntrinsics):
    key = (_intr_key(src), _intr_key(dst))
    if key not in _MAP_CACHE:
        _MAP_CACHE.clear()  # one camera at a time is the common case; keep memory bounded
        u, v = np.meshgrid(np.arange(dst.width, dtype=np.float64), np.arange(dst.height, dtype=np.float64))
        Kd = dst.K
        y = (v - Kd[1, 2]) / Kd[1, 1]
        x = (u - Kd[0, 2] - Kd[0, 1] * y) / Kd[0, 0]
        xy = distort_normalized(np.stack([x.ravel(), y.ravel()], -1), src.D)
        Ks = src.K
        mx = Ks[0, 0] * xy[:, 0] + Ks[0, 1] * xy[:, 1] + Ks[0, 2]
        my = Ks[1, 1] * xy[:, 1] + Ks[1, 2]
        _MAP_CACHE[key] = (mx.reshape(dst.height, dst.width).astype(np.float32),
                           my.reshape(dst.height, dst.width).astype(np.float32))
    return _MAP_CACHE[key]


def _remap(img: np.ndarray, mx: np.ndarray, my: np.ndarray, nearest: bool) -> np.ndarray:
    h, w = img.shape[:2]
    if nearest:
        xi, yi = np.rint(mx).astype(np.int64), np.rint(my).astype(np.int64)
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        out = np.zeros(mx.shape + img.shape[2:], img.dtype)
        out[ok] = img[yi[ok], xi[ok]]
        return out
    x0 = np.floor(mx).astype(np.int64)
    y0 = np.floor(my).astype(np.int64)
    ok = (x0 >= 0) & (x0 < w - 1) & (y0 >= 0) & (y0 < h - 1)
    x0c, y0c = np.clip(x0, 0, w - 2), np.clip(y0, 0, h - 2)
    ax, ay = mx - x0c, my - y0c
    if img.ndim == 3:
        ax, ay = ax[..., None], ay[..., None]
    f = img.astype(np.float32)
    out = (
        f[y0c, x0c] * (1 - ax) * (1 - ay) + f[y0c, x0c + 1] * ax * (1 - ay)
        + f[y0c + 1, x0c] * (1 - ax) * ay + f[y0c + 1, x0c + 1] * ax * ay
    )
    out[~ok] = 0
    if np.issubdtype(img.dtype, np.integer):
        out = np.clip(np.rint(out), np.iinfo(img.dtype).min, np.iinfo(img.dtype).max)
    return out.astype(img.dtype)

