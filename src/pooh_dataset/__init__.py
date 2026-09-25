"""pooh-dataset: download and process the POOH multimodal robotics dataset.

Quick start::

    from pooh_dataset import PoohDataset
    ds = PoohDataset.download(modalities=["mocap", "realsense_color"])
    traj = ds["trajectory_007"]
    frame = traj.frame("realsense_color", 0)
"""

__version__ = "0.2.0"

from .calibration import Calibration, CameraIntrinsics, MissingCalibrationError
from .dataset import PoohDataset
from .events import Events
from .hub import DEFAULT_REPO_ID, download
from .objects import ObjectModel
from .trajectory import Trajectory

__all__ = [
    "Calibration",
    "CameraIntrinsics",
    "DEFAULT_REPO_ID",
    "Events",
    "MissingCalibrationError",
    "ObjectModel",
    "PoohDataset",
    "Trajectory",
    "download",
]
