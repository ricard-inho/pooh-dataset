"""Lazy, memory-friendly access to one recorded trajectory on local disk."""

from __future__ import annotations

import functools
import io
import json
from collections import OrderedDict
from functools import cached_property
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import yaml
from PIL import Image

from .calibration import Calibration
from .events import Events
from .geometry import interpolate_poses

__all__ = ["Trajectory", "IMAGE_MODALITIES", "SIM_IMAGE_MODALITIES", "DEPTH_MODALITIES", "CAMERAS"]

IMAGE_MODALITIES = ("firefly", "realsense_color", "realsense_depth")
#: simulated camera (one pinhole camera, pixel-aligned with the simulated event camera)
SIM_IMAGE_MODALITIES = ("rgb", "depth", "semantic")
DEPTH_MODALITIES = ("realsense_depth", "depth")
#: every sensor that has intrinsics / can be a label target
CAMERAS = IMAGE_MODALITIES + SIM_IMAGE_MODALITIES + ("events",)
#: on-disk column holding the trajectory name in every table; dropped from the readers' output
TRAJECTORY_ID = "trajectory_id"


class Trajectory:
    """One recording session. Timestamps are only comparable within a trajectory.

    Data is read straight from the Parquet shards; nothing is loaded until asked for.
    Simulated trajectories live in ``<prefix><modality>/<name>/`` (prefix ``sim_3d_`` or
    ``sim_zerog_``); modalities are named without the prefix (``rgb``, ``depth``, ...).
    """

    def __init__(self, root: Path, name: str, info: dict | None = None, row_group_cache: int = 2,
                 prefix: str = ""):
        self.root = Path(root)
        self.name = name
        self.info = info or {}
        self.prefix = prefix
        self._rg_cache: OrderedDict = OrderedDict()
        self._rg_cache_size = row_group_cache
        self._pf_cache: dict[Path, pq.ParquetFile] = {}
        self._ts_cache: dict[str, np.ndarray] = {}
        self._row_index: dict[str, list[tuple[Path, int, int]]] = {}

    def __repr__(self) -> str:
        return f"Trajectory({self.name!r}, modalities={self.modalities})"

    # ------------------------------------------------------------------ files

    @property
    def domain(self) -> str:
        return self.info.get("domain", "real")

    @property
    def simulated(self) -> bool:
        """Recorded by the simulation pipeline (``sim_*`` folders on the Hub, or a local
        conversion with a ``scenario``). Not the same as ``domain == "synthetic"``, which
        trajectories in the real layout may also be marked with."""
        return bool(self.prefix) or "scenario" in self.info

    def folder(self, modality: str) -> Path:
        return self.root / f"{self.prefix}{modality}" / self.name

    def files(self, modality: str) -> list[Path]:
        return sorted(self.folder(modality).glob("*.parquet"))

    @cached_property
    def modalities(self) -> list[str]:
        return sorted(
            p.name[len(self.prefix):] for p in self.root.iterdir()
            if p.is_dir() and p.name.startswith(self.prefix)
            and (p / self.name).is_dir() and any((p / self.name).glob("*.parquet"))
        )

    @property
    def depth_modality(self) -> str | None:
        """The depth camera of this trajectory (``realsense_depth`` or simulated ``depth``)."""
        return next((m for m in DEPTH_MODALITIES if self.has(m)), None)

    def has(self, modality: str) -> bool:
        return modality in self.modalities

    def _require(self, modality: str) -> list[Path]:
        files = self.files(modality)
        if not files:
            raise FileNotFoundError(
                f"[{self.name}] no local data for {modality!r} under {self.folder(modality)}. "
                f"Download it with `pooh download --trajectory {self.name} --modality {modality}`."
            )
        return files

    def _parquet(self, path: Path) -> pq.ParquetFile:
        if path not in self._pf_cache:
            self._pf_cache[path] = pq.ParquetFile(path, memory_map=True)
        return self._pf_cache[path]

    def dataset(self, modality: str) -> pads.Dataset:
        return pads.dataset([str(p) for p in self._require(modality)], format="parquet")

    def table(self, modality: str, columns: list[str] | None = None, filter=None) -> pa.Table:
        """Read (a projection / filter of) a modality as one Arrow table."""
        return self.dataset(modality).to_table(columns=columns, filter=filter)

    def columns(self, modality: str) -> list[str]:
        """Column names of a modality, without the ``trajectory_id`` bookkeeping column."""
        return [c for c in self.dataset(modality).schema.names if c != TRAJECTORY_ID]

    @cached_property
    def calibration(self) -> Calibration:
        return Calibration.load(self.root / "calibration", self.name)

    # ------------------------------------------------------------ timestamps

    def timestamps(self, modality: str) -> np.ndarray:
        """int64 ns host-clock timestamps in row order, with the calibrated per-sensor time
        offset applied (not available for ``events``: use :meth:`time_range` / :meth:`events`)."""
        if modality == "events":
            raise ValueError("events has ~1e8 rows; use time_range('events') or events(t0, t1)")
        if modality not in self._ts_cache:
            col = self.table(modality, columns=["timestamp_ns"]).column("timestamp_ns")
            ts = col.to_numpy().astype(np.int64) + self.calibration.time_offset_ns(modality)
            self._ts_cache[modality] = ts
        return self._ts_cache[modality]

    def time_range(self, modality: str) -> tuple[int, int]:
        """(first, last) timestamp from Parquet statistics, without reading data."""
        col = "t" if modality == "events" else "timestamp_ns"
        lo, hi = None, None
        for path in self._require(modality):
            md = self._parquet(path).metadata
            idx = md.schema.to_arrow_schema().get_field_index(col)
            for rg in range(md.num_row_groups):
                st = md.row_group(rg).column(idx).statistics
                if st is None or not st.has_min_max:
                    ts = self.timestamps(modality)
                    return int(ts.min()), int(ts.max())
                lo = st.min if lo is None else min(lo, st.min)
                hi = st.max if hi is None else max(hi, st.max)
        off = self.calibration.time_offset_ns(modality)
        return int(lo) + off, int(hi) + off

    def num_rows(self, modality: str) -> int:
        return sum(self._parquet(p).metadata.num_rows for p in self._require(modality))

    # ---------------------------------------------------------------- images

    def _locate(self, modality: str, index: int) -> tuple[Path, int, int]:
        """row index -> (file, row_group, offset within the row group)."""
        if modality not in self._row_index:
            spans = []
            for path in self._require(modality):
                md = self._parquet(path).metadata
                for rg in range(md.num_row_groups):
                    spans.append((path, rg, md.row_group(rg).num_rows))
            self._row_index[modality] = spans
        if index < 0:
            index += self.num_rows(modality)
        for path, rg, n in self._row_index[modality]:
            if index < n:
                return path, rg, index
            index -= n
        raise IndexError(f"{modality} index out of range")

    def _row_group(self, path: Path, rg: int, column: str) -> pa.ChunkedArray:
        key = (path, rg, column)
        if key in self._rg_cache:
            self._rg_cache.move_to_end(key)
            return self._rg_cache[key]
        arr = self._parquet(path).read_row_group(rg, columns=[column]).column(column)
        self._rg_cache[key] = arr
        while len(self._rg_cache) > self._rg_cache_size:
            self._rg_cache.popitem(last=False)
        return arr

    def frame_bytes(self, modality: str, index: int) -> bytes:
        """Raw encoded bytes (JPEG for colour, 16-bit PNG for depth) of frame ``index``."""
        path, rg, off = self._locate(modality, index)
        cell = self._row_group(path, rg, "image")[off].as_py()
        return cell["bytes"] if isinstance(cell, dict) else cell

    def frame(self, modality: str, index: int) -> np.ndarray:
        """Decoded frame: (H, W, 3) uint8 RGB, or (H, W) uint16 raw depth."""
        return np.asarray(Image.open(io.BytesIO(self.frame_bytes(modality, index))))

    def depth(self, index: int, modality: str | None = None) -> np.ndarray:
        """Depth frame in metres (float32, 0 = no measurement). ``modality`` defaults to the
        trajectory's depth camera (:attr:`depth_modality`)."""
        modality = modality or self.depth_modality or DEPTH_MODALITIES[0]
        scale = self.calibration.cameras.get(modality)
        s = scale.depth_scale if scale is not None and scale.depth_scale else 0.001
        return self.frame(modality, index).astype(np.float32) * np.float32(s)

    def iter_frames(self, modality: str, start: int = 0, stop: int | None = None, step: int = 1):
        """Yield (index, timestamp_ns, frame) sequentially (row-group cache friendly)."""
        ts = self.timestamps(modality)
        for i in range(start, len(ts) if stop is None else min(stop, len(ts)), step):
            yield i, int(ts[i]), self.frame(modality, i)

    # ---------------------------------------------------------------- events

    def events(self, t_start_ns: int, t_end_ns: int) -> Events:
        """Events with ``t_start_ns <= t < t_end_ns``. Only the row groups overlapping the
        window are read (the stream is time-sorted with per-row-group statistics)."""
        off = self.calibration.time_offset_ns("events")
        flt = (pc.field("t") >= pa.scalar(int(t_start_ns) - off, pa.int64())) & (
            pc.field("t") < pa.scalar(int(t_end_ns) - off, pa.int64())
        )
        tab = self.table("events", columns=["t", "x", "y", "p"], filter=flt)
        if tab.num_rows == 0:
            return Events.empty()
        t, x, y, p = (tab.column(c).to_numpy() for c in ("t", "x", "y", "p"))
        return Events(t + off if off else t, x, y, p)

    # ------------------------------------------------------------ mocap/imu

    @cached_property
    def _mocap(self) -> dict[str, dict[str, np.ndarray]]:
        tab = self.table("mocap", columns=["body", "timestamp_ns", "x", "y", "z", "qx", "qy", "qz", "qw"])
        body = tab.column("body").to_numpy(zero_copy_only=False)
        cols = {c: tab.column(c).to_numpy() for c in ("timestamp_ns", "x", "y", "z", "qx", "qy", "qz", "qw")}
        cols["timestamp_ns"] = cols["timestamp_ns"] + self.calibration.time_offset_ns("mocap")
        out = {}
        for b in np.unique(body):
            sel = body == b
            order = np.argsort(cols["timestamp_ns"][sel], kind="stable")
            out[str(b)] = {k: v[sel][order] for k, v in cols.items()}
        return out

    @property
    def mocap_bodies(self) -> list[str]:
        return sorted(self._mocap)

    def mocap(self, body: str) -> dict[str, np.ndarray]:
        """Pose track of one rigid body: timestamp_ns, x, y, z, qx, qy, qz, qw (world frame)."""
        if body not in self._mocap:
            raise KeyError(f"[{self.name}] mocap body {body!r} not tracked (have {self.mocap_bodies})")
        return self._mocap[body]

    def body_pose(self, body: str, t_ns, max_gap_ns: int | None = 50_000_000):
        """T_world_body interpolated at ``t_ns`` (scalar or array). Returns (T, valid)."""
        m = self.mocap(body)
        pos = np.stack([m["x"], m["y"], m["z"]], -1)
        quat = np.stack([m["qx"], m["qy"], m["qz"], m["qw"]], -1)
        return interpolate_poses(m["timestamp_ns"], pos, quat, t_ns, max_gap_ns=max_gap_ns)

    def imu(self) -> dict[str, np.ndarray]:
        """IMU samples as column arrays. Linear acceleration is in g for real data (Livox
        driver unit) and in m/s^2 for simulated data."""
        tab = self.table("imu", columns=self.columns("imu"))
        return {c: tab.column(c).to_numpy() for c in tab.column_names}

    # ----------------------------------------------------------------- lidar

    def lidar_scan(self, index: int) -> dict[str, np.ndarray]:
        """One scan: arrays x, y, z, ... + scalars. Real (Mid-360): intensity, tag, line,
        timestamp; simulated: distance, channel, with points in the sensor frame."""
        path, rg, off = self._locate("lidar", index)
        cols = [c for c in self._parquet(path).schema_arrow.names if c != TRAJECTORY_ID]
        row = self._parquet(path).read_row_group(rg, columns=cols).slice(off, 1).to_pylist()[0]
        return {k: (np.asarray(v) if isinstance(v, list) else v) for k, v in row.items()}

    # ------------------------------------------------------------ simulation

    def steps(self) -> dict[str, np.ndarray]:
        """Simulated policy steps (agent rate) as column arrays: ``timestamp_ns``, ``step``,
        ``action`` (N, A), ``observation`` (N, O), ``reward``, ``reward_terms`` (N, R),
        ``goal_position_error``, ``goal_orientation_error``, ``terminated``, ``truncated``, and
        ``task_state`` (JSON strings). Term names are in :attr:`metadata`."""
        tab = self.table("steps", columns=self.columns("steps"))
        out = {}
        for c in tab.column_names:
            col = tab.column(c)
            if pa.types.is_list(col.type):
                out[c] = np.asarray(col.to_pylist(), dtype=np.float64)
            else:
                out[c] = col.to_numpy()
        return out

    def joint_states(self, articulation: str) -> dict[str, np.ndarray]:
        """Simulated joint states of one articulation (``robot``, ``arm_wall``, ...):
        ``timestamp_ns`` and (N, J) ``position``, ``velocity``, ``effort``. Joint names are in
        ``metadata["articulations"]``."""
        tab = self.table("joint_states", columns=["timestamp_ns", "position", "velocity", "effort"],
                         filter=pc.field("articulation") == articulation)
        if tab.num_rows == 0:
            raise KeyError(f"[{self.name}] no joint states for articulation {articulation!r}")
        tab = tab.sort_by("timestamp_ns")
        out = {"timestamp_ns": tab.column("timestamp_ns").to_numpy()}
        for c in ("position", "velocity", "effort"):
            out[c] = np.asarray(tab.column(c).to_pylist(), dtype=np.float64)
        return out

    @cached_property
    def metadata(self) -> dict:
        """``metadata/<trajectory>.yaml`` of a simulated trajectory: scenario, rates, frames and
        units, semantic classes, observation / reward term layout ({} for real data)."""
        path = self.root / "metadata" / f"{self.name}.yaml"
        if not path.exists():
            return {}
        return yaml.safe_load(path.read_text()) or {}

    @property
    def target(self) -> dict | None:
        """The simulated target's CAD (``models/sim/targets.json``): ``model``, ``kind``
        (satellite_mockup, cubesat, iss, gateway, venus_express) and ``scale``; None for real
        data or when the models are not downloaded."""
        return _sim_targets(self.root / "models" / "sim" / "targets.json").get(self.name)

    @property
    def target_model_path(self) -> Path | None:
        """``models/sim/<model>/model.obj``: metres, in the frame of the mocap body ``target``,
        **unscaled** (multiply by ``target["scale"]``). The simulator's USD is next to it."""
        t = self.target
        return None if t is None else self.root / "models" / "sim" / t["model"] / "model.obj"

    def target_mesh(self, scaled: bool = True):
        """The target's CAD as a ``trimesh.Trimesh`` (needs ``pooh-dataset[mesh]``), in the
        ``target`` body frame, scaled to this trajectory unless ``scaled=False``."""
        import trimesh

        path = self.target_model_path
        if path is None or not path.exists():
            raise FileNotFoundError(f"{self.name}: no target CAD; download it with pooh download (sim data)")
        mesh = trimesh.load(path, force="mesh", process=False)
        if scaled:
            mesh.apply_scale(float(self.target["scale"]))
        return mesh

    @property
    def semantic_classes(self) -> dict[str, int]:
        """Class name -> id of the simulated ``semantic`` images (e.g. ``target``: 2)."""
        return dict(self.metadata.get("semantic_classes", {}))

    # -------------------------------------------------------------- messages

    @cached_property
    def topics(self) -> list[str]:
        return sorted(pc.unique(self.table("robot_messages", columns=["topic"]).column("topic")).to_pylist())

    def messages(self, topic: str) -> tuple[np.ndarray, list[dict]]:
        """(timestamp_ns array, decoded JSON payloads) of one robot topic, time-sorted."""
        tab = self.table("robot_messages", columns=["timestamp_ns", "data"],
                         filter=pc.field("topic") == topic)
        ts = tab.column("timestamp_ns").to_numpy()
        order = np.argsort(ts, kind="stable")
        data = tab.column("data").to_pylist()
        return ts[order], [json.loads(data[i]) for i in order]

    def message_array(self, topic: str, field: str = "data") -> tuple[np.ndarray, np.ndarray]:
        """For array topics (Float64MultiArray, ...): (timestamp_ns, (N, D) array of ``field``)."""
        ts, msgs = self.messages(topic)
        return ts, np.asarray([m[field] for m in msgs], dtype=np.float64)


@functools.lru_cache(maxsize=4)
def _sim_targets_cached(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text())


def _sim_targets(path: Path) -> dict:
    """``models/sim/targets.json`` (cached per file version), {} without it."""
    return _sim_targets_cached(str(path), path.stat().st_mtime) if path.exists() else {}
