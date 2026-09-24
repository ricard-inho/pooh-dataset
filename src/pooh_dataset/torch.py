"""PyTorch adapters (``pip install pooh-dataset[torch]``).

>>> from pooh_dataset import PoohDataset
>>> from pooh_dataset.tasks import ObjectDetection
>>> from pooh_dataset.torch import TorchDataset, detection_collate
>>> ds = TorchDataset(ObjectDetection(PoohDataset(), "train", camera="realsense_color"))
>>> loader = torch.utils.data.DataLoader(ds, batch_size=8, collate_fn=detection_collate)
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as e:  # pragma: no cover
    raise ImportError("pooh_dataset.torch needs PyTorch: `pip install pooh-dataset[torch]`") from e

__all__ = ["TorchDataset", "to_tensor_sample", "detection_collate"]

_IMAGE_KEYS = {"image", "image0", "image1"}


def to_tensor_sample(sample: dict[str, Any], channels_first: bool = True) -> dict[str, Any]:
    """numpy -> torch. uint8 RGB images become float32 CHW in [0, 1]; flow becomes (2, H, W)."""
    out: dict[str, Any] = {}
    for k, v in sample.items():
        if not isinstance(v, np.ndarray):
            out[k] = v
            continue
        if k in _IMAGE_KEYS and v.ndim == 3 and v.dtype == np.uint8:
            t = torch.from_numpy(np.ascontiguousarray(v)).float().div_(255.0)
            out[k] = t.permute(2, 0, 1) if channels_first else t
        elif k in _IMAGE_KEYS and v.ndim == 2:  # depth
            out[k] = torch.from_numpy(v.astype(np.float32))[None]
        elif k == "flow" and channels_first:
            out[k] = torch.from_numpy(np.ascontiguousarray(v.transpose(2, 0, 1)))
        else:
            out[k] = torch.from_numpy(np.ascontiguousarray(v))
    return out


class TorchDataset(Dataset):
    """Wrap any :mod:`pooh_dataset.tasks` dataset. ``transform`` gets the tensor dict."""

    def __init__(self, task, transform=None, channels_first: bool = True):
        self.task = task
        self.transform = transform
        self.channels_first = channels_first

    def __len__(self) -> int:
        return len(self.task)

    def __getitem__(self, i: int):
        s = to_tensor_sample(self.task[i], self.channels_first)
        return self.transform(s) if self.transform else s


def detection_collate(batch: list[dict]) -> tuple[torch.Tensor, list[dict]]:
    """torchvision-detection style: stacked images + list of per-image target dicts."""
    images = torch.stack([b["image"] for b in batch])
    targets = [{k: v for k, v in b.items() if k != "image"} for b in batch]
    return images, targets
