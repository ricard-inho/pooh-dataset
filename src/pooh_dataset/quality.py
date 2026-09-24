"""Per-trajectory health report: sensor rates and gaps, mocap dropouts, goal reaching.

Only metadata and small tables are read (mocap, messages, timestamps), so this runs in
seconds even when the heavy modalities are not downloaded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .dataset import PoohDataset
from .geometry import quat_to_rotmat
from .labels import nearest_indices
from .trajectory import IMAGE_MODALITIES, Trajectory

__all__ = ["StreamStats", "TrajectoryReport", "trajectory_report", "dataset_report", "yaw_from_quat"]

GOAL_TOPIC = "/observation_formater_input"
ALL_MODALITIES = ["events", "firefly", "realsense_color", "realsense_depth", "lidar", "imu", "mocap",
                  "robot_messages"]
ROBOT_BODY = "floatingplatform"


def yaw_from_quat(q_xyzw: np.ndarray) -> np.ndarray:
    """Heading (rad) about world z of (..., 4) xyzw quaternions."""
    R = quat_to_rotmat(q_xyzw)
    return np.arctan2(R[..., 1, 0], R[..., 0, 0])


def _wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


@dataclass
class StreamStats:
    name: str
    count: int
    rate_hz: float
    median_dt_ms: float
    max_gap_ms: float
    #: gaps longer than ``gap_factor`` x the median period
    n_gaps: int

    @classmethod
    def from_timestamps(cls, name: str, ts: np.ndarray, gap_factor: float = 3.0) -> StreamStats:
        ts = np.sort(np.asarray(ts, np.int64))
        if len(ts) < 2:
            return cls(name, len(ts), 0.0, float("nan"), float("nan"), 0)
        dt = np.diff(ts) / 1e6
        med = float(np.median(dt))
        span = (ts[-1] - ts[0]) / 1e9
        return cls(name, len(ts), (len(ts) - 1) / span if span > 0 else 0.0, med,
                   float(dt.max()), int((dt > gap_factor * med).sum()))


@dataclass
class TrajectoryReport:
    trajectory: str
    domain: str
    duration_s: float = float("nan")
    local_modalities: list[str] = field(default_factory=list)
    #: sensors that recorded nothing for this trajectory (per the manifest)
    missing_modalities: list[str] = field(default_factory=list)
    #: recorded, but not downloaded locally (not a problem, just not analysed)
    not_downloaded: list[str] = field(default_factory=list)
    streams: dict[str, StreamStats] = field(default_factory=dict)
    mocap_bodies: list[str] = field(default_factory=list)
    #: fraction of camera frames with a valid (gap-free) interpolated robot pose
    label_coverage: dict[str, float] = field(default_factory=dict)
    goals: list[list[float]] = field(default_factory=list)  # [x, y, yaw] per goal message
    final_pos_error_m: float = float("nan")
    final_yaw_error_deg: float = float("nan")
    min_goal_dist_m: float = float("nan")
    path_length_m: float = float("nan")
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.warnings

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ok"] = self.ok
        return d


def trajectory_report(
    traj: Trajectory,
    expected_modalities: list[str] | None = None,
    robot_body: str = ROBOT_BODY,
    max_gap_ns: int = 50_000_000,
    goal_tolerance_m: float = 0.10,
    goal_tolerance_deg: float = 10.0,
    min_camera_hz: dict[str, float] | None = None,
) -> TrajectoryReport:
    rep = TrajectoryReport(traj.name, traj.info.get("domain", "real"), local_modalities=traj.modalities)
    counts = traj.info.get("counts", {})
    expected = expected_modalities or ALL_MODALITIES
    recorded = [m for m in expected if counts.get(m, 0) > 0 or traj.has(m)]
    rep.missing_modalities = [m for m in expected if m not in recorded] if counts else []
    rep.not_downloaded = [m for m in recorded if not traj.has(m)]
    min_camera_hz = {"firefly": 20.0, "realsense_color": 10.0, "realsense_depth": 10.0,
                     **(min_camera_hz or {})}
    spans = []

    for m in IMAGE_MODALITIES + ("imu", "lidar"):
        if traj.has(m):
            ts = traj.timestamps(m)
            rep.streams[m] = st = StreamStats.from_timestamps(m, ts)
            spans.append((ts.min(), ts.max()))
            if m in min_camera_hz and st.rate_hz < min_camera_hz[m]:
                rep.warnings.append(f"{m} at {st.rate_hz:.1f} Hz (< {min_camera_hz[m]:g})")
            if st.n_gaps:
                rep.warnings.append(f"{m}: {st.n_gaps} gaps (max {st.max_gap_ms:.0f} ms)")
    if traj.has("events"):
        lo, hi = traj.time_range("events")
        n = traj.num_rows("events")
        rep.streams["events"] = StreamStats("events", n, n / max((hi - lo) / 1e9, 1e-9),
                                            float("nan"), float("nan"), 0)
        spans.append((lo, hi))

    if traj.has("mocap"):
        rep.mocap_bodies = traj.mocap_bodies
        for b in rep.mocap_bodies:
            ts = traj.mocap(b)["timestamp_ns"]
            st = StreamStats.from_timestamps(f"mocap/{b}", ts)
            st.n_gaps = int((np.diff(ts) > max_gap_ns).sum())
            rep.streams[f"mocap/{b}"] = st
            spans.append((ts.min(), ts.max()))
            if st.n_gaps:
                rep.warnings.append(f"mocap/{b}: {st.n_gaps} dropouts > {max_gap_ns / 1e6:.0f} ms "
                                    f"(max {st.max_gap_ms:.0f} ms)")
        if robot_body not in rep.mocap_bodies:
            rep.warnings.append(f"robot body {robot_body!r} not tracked by mocap")
        else:
            _robot_metrics(traj, rep, robot_body, max_gap_ns, goal_tolerance_m, goal_tolerance_deg)
    if spans:
        rep.duration_s = (max(s[1] for s in spans) - min(s[0] for s in spans)) / 1e9
    if rep.missing_modalities:
        rep.warnings.append("not recorded: " + ", ".join(rep.missing_modalities))
    if traj.has("realsense_color") and traj.has("realsense_depth"):
        _, ok = nearest_indices(traj.timestamps("realsense_depth"),
                                traj.timestamps("realsense_color"), 20_000_000)
        rep.label_coverage["color_depth_sync_20ms"] = float(ok.mean())
    return rep


def _robot_metrics(traj, rep, body, max_gap_ns, tol_m, tol_deg):
    m = traj.mocap(body)
    xy = np.stack([m["x"], m["y"]], -1)
    yaw = yaw_from_quat(np.stack([m["qx"], m["qy"], m["qz"], m["qw"]], -1))
    rep.path_length_m = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    for cam in IMAGE_MODALITIES:
        if traj.has(cam):
            _, ok = traj.body_pose(body, traj.timestamps(cam), max_gap_ns)
            rep.label_coverage[cam] = float(np.mean(ok))
    if traj.has("robot_messages") and GOAL_TOPIC in traj.topics:
        _, msgs = traj.messages(GOAL_TOPIC)
        for g in msgs:
            p, o = g["pose"]["position"], g["pose"]["orientation"]
            gyaw = float(yaw_from_quat(np.array([o["x"], o["y"], o["z"], o["w"]])))
            rep.goals.append([p["x"], p["y"], gyaw])
    if rep.goals:
        gx, gy, gyaw = rep.goals[-1]
        rep.final_pos_error_m = float(np.hypot(xy[-1, 0] - gx, xy[-1, 1] - gy))
        rep.final_yaw_error_deg = float(np.degrees(abs(_wrap(yaw[-1] - gyaw))))
        rep.min_goal_dist_m = float(np.hypot(xy[:, 0] - gx, xy[:, 1] - gy).min())
        if rep.final_pos_error_m > tol_m:
            rep.warnings.append(f"goal missed: final error {rep.final_pos_error_m * 100:.1f} cm")
        if rep.final_yaw_error_deg > tol_deg:
            rep.warnings.append(f"heading missed: final error {rep.final_yaw_error_deg:.1f} deg")
    else:
        rep.warnings.append(f"no goal message on {GOAL_TOPIC}")


def dataset_report(ds: PoohDataset, trajectories: list[str] | None = None, **kw) -> list[TrajectoryReport]:
    names = trajectories or ds.local_trajectory_names
    return [trajectory_report(ds[t], **kw) for t in names]


def format_reports(reports: list[TrajectoryReport]) -> str:
    """Compact text table + warnings, for the terminal."""
    def f(x, fmt):
        width = int(fmt.split(".")[0] or 0)
        return "-".rjust(width) if x is None or np.isnan(x) else format(x, fmt)

    head = f"{'trajectory':18s} {'dur s':>6s} {'final err cm':>12s} {'yaw err°':>8s} {'path m':>7s} " \
           f"{'mocap drops':>11s} {'label cov':>9s}  status"
    lines = [head, "-" * len(head)]
    for r in reports:
        drops = sum(s.n_gaps for k, s in r.streams.items() if k.startswith("mocap/"))
        covs = [r.label_coverage[c] for c in IMAGE_MODALITIES if c in r.label_coverage]
        cov = min(covs) * 100 if covs else float("nan")
        lines.append(
            f"{r.trajectory:18s} {f(r.duration_s, '6.1f')} {f(r.final_pos_error_m * 100, '12.1f')} "
            f"{f(r.final_yaw_error_deg, '8.1f')} {f(r.path_length_m, '7.2f')} {drops:11d} "
            f"{f(cov, '8.0f') + '%' if covs else '        -'}  {'OK' if r.ok else 'CHECK'}"
        )
    lines.append("")
    lines.append("label cov = share of camera frames with a gap-free robot mocap pose (worst camera)")
    for r in reports:
        for w in r.warnings:
            lines.append(f"  {r.trajectory}: {w}")
        if r.not_downloaded:
            lines.append(f"  {r.trajectory}: (not downloaded, not checked: {', '.join(r.not_downloaded)})")
    return "\n".join(lines)
