"""Synchronized playback of trajectories in the Rerun viewer (``pip install pooh-dataset[viz]``).

Every trajectory becomes one Rerun *recording* on a shared ``time`` timeline (seconds from
the trajectory start), with:

* a 3D mocap world: every tracked body's pose + full path, and the goal pose(s)
* x, y, yaw of every body vs time (goal as a dashed reference), distance to goal
* the camera views: Firefly, RealSense colour + depth, and event frames
* the Livox point cloud, IMU, thruster and reaction-wheel commands

Scrub the timeline or press play; switch trajectories in the viewer's recording panel.
"""

from __future__ import annotations

import numpy as np

try:
    import rerun as rr
    import rerun.blueprint as rrb
except ImportError as e:  # pragma: no cover
    raise ImportError("visualization needs Rerun: `pip install pooh-dataset[viz]`") from e

from .events import EVENT_SENSOR_SIZE, event_histogram
from .quality import GOAL_TOPIC, ROBOT_BODY, yaw_from_quat
from .trajectory import Trajectory

__all__ = ["log_trajectory", "blueprint", "render_events", "visualize"]

TIMELINE = "time"
APP_ID = "pooh_dataset"
BODY_COLORS = {"floatingplatform": (255, 140, 0), "SensorStack": (60, 160, 255), "cubesat": (80, 220, 120)}
THRUSTER_TOPIC = "/minimal_thruster_command_to_px4"
RW_TOPIC = "/rw_effort_controller/commands"
TASK_TOPIC = "/task_available_interface"


def _color(body: str, i: int = 0):
    fallback = [(230, 80, 200), (240, 220, 60), (160, 160, 160)]
    return BODY_COLORS.get(body, fallback[i % len(fallback)])


def _turbo(v: np.ndarray) -> np.ndarray:
    """Turbo colormap (polynomial fit, Mikhailov 2019) for values in [0, 1] -> uint8 RGB."""
    v = np.clip(v, 0, 1)[:, None]
    c = np.array([
        [0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943],
        [0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604],
        [0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973],
    ])  # fmt: skip
    powers = v ** np.arange(6)
    return (np.clip(powers @ c.T, 0, 1) * 255).astype(np.uint8)


def render_events(ev, height: int, width: int, scale: float = 1.0) -> np.ndarray:
    """(h, w, 3) uint8 frame: dark background, blue = positive, red = negative polarity."""
    h, w = max(1, int(height * scale)), max(1, int(width * scale))
    if scale != 1.0 and len(ev):
        ev = type(ev)(ev.t, (ev.x * scale).astype(np.uint16), (ev.y * scale).astype(np.uint16), ev.p)
    neg, pos = event_histogram(ev, h, w)
    img = np.full((h, w, 3), 25, np.uint8)
    img[pos > neg] = (70, 170, 255)
    img[neg > pos] = (255, 80, 80)
    return img


def blueprint(traj: Trajectory | None = None, modalities: list[str] | None = None) -> rrb.Blueprint:
    def has(m: str) -> bool:
        return (traj is None or traj.has(m)) and (modalities is None or m in modalities)

    cams = [rrb.Spatial2DView(origin=f"cams/{c}", name=c)
            for c in ("firefly", "realsense_color", "realsense_depth", "events") if has(c)]
    right = [rrb.Grid(*cams)] if cams else []
    if has("lidar"):
        right.append(rrb.Spatial3DView(origin="lidar", name="Livox Mid-360"))
    plots_right = []
    if has("robot_messages"):
        plots_right.append(rrb.TimeSeriesView(origin="plots/actions", name="thrusters / reaction wheel"))
    if has("imu"):
        plots_right.append(rrb.TimeSeriesView(origin="plots/imu", name="IMU"))
    if plots_right:
        right.append(rrb.Horizontal(*plots_right))
    left = rrb.Vertical(
        rrb.Spatial3DView(origin="world", name="mocap world"),
        rrb.Grid(
            rrb.TimeSeriesView(origin="plots/x", name="x [m]"),
            rrb.TimeSeriesView(origin="plots/y", name="y [m]"),
            rrb.TimeSeriesView(origin="plots/yaw", name="yaw [deg]"),
            rrb.TimeSeriesView(origin="plots/goal_distance", name="distance to goal [m]"),
        ),
        row_shares=[3, 2],
    )
    layout = rrb.Horizontal(left, rrb.Vertical(*right), column_shares=[2, 3]) if right else left
    return rrb.Blueprint(layout, collapse_panels=True)


def _t0(traj: Trajectory) -> int:
    starts = [traj.time_range(m)[0] for m in traj.modalities if m != "robot_messages"]
    return min(starts)


def log_trajectory(
    traj: Trajectory,
    rec: rr.RecordingStream,
    *,
    modalities: list[str] | None = None,
    robot_body: str = ROBOT_BODY,
    start_s: float = 0.0,
    end_s: float | None = None,
    firefly_stride: int = 2,
    events_fps: float = 20.0,
    events_scale: float = 0.5,
    lidar_stride: int = 1,
    progress: bool = True,
    objects: list | None = None,
    labels: bool = True,
    max_mask_pixels: int = 1_000_000,
) -> None:
    """Log one trajectory into ``rec``. Only locally available modalities are logged.

    ``objects`` (e.g. ``list(dataset.objects.values())``) are drawn in the mocap world at
    their mocap pose, through their ``T_body_model``: a quick visual check of that transform.
    With ``labels``, every camera that has extrinsics also gets the ground truth drawn on its
    images (box, cube wireframe, and the mask for cameras up to ``max_mask_pixels``), and its
    frustum in the 3D view.
    """
    mods = [m for m in (modalities or traj.modalities) if traj.has(m)]
    t0 = _t0(traj)
    lo = t0 + int(start_s * 1e9)
    hi = None if end_s is None else t0 + int(end_s * 1e9)

    def window(ts):
        ts = np.asarray(ts, np.int64)
        return (ts >= lo) & (ts <= (hi if hi is not None else ts.max(initial=lo)))

    def secs(ts):
        return (np.asarray(ts, np.int64) - t0) / 1e9

    def tcol(ts):
        return [rr.TimeColumn(TIMELINE, duration=secs(ts))]

    rec.send_recording_name(traj.name)
    rec.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rec.log("lidar", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rec.log("info", rr.TextDocument(_info_text(traj), media_type="text/markdown"), static=True)

    goals = []
    if "robot_messages" in mods and GOAL_TOPIC in traj.topics:
        _, msgs = traj.messages(GOAL_TOPIC)
        for g in msgs:
            p, o = g["pose"]["position"], g["pose"]["orientation"]
            goals.append((p["x"], p["y"], p["z"], float(yaw_from_quat(np.array([o["x"], o["y"], o["z"], o["w"]])))))

    step = _Progress(progress, traj.name)

    if "mocap" in mods:
        step("mocap")
        span = None
        for i, body in enumerate(traj.mocap_bodies):
            m = traj.mocap(body)
            sel = window(m["timestamp_ns"])
            if not sel.any():
                continue
            ts = m["timestamp_ns"][sel]
            pos = np.stack([m["x"], m["y"], m["z"]], -1)[sel]
            quat = np.stack([m["qx"], m["qy"], m["qz"], m["qw"]], -1)[sel]
            yaw = np.degrees(yaw_from_quat(quat))
            col = _color(body, i)
            rec.log(f"world/bodies/{body}", rr.TransformAxes3D(0.15), static=True)
            if not any(o.mocap_body == body and o.present_in(traj.name) for o in objects or ()):
                rec.log(f"world/bodies/{body}/box", rr.Boxes3D(half_sizes=[0.08, 0.08, 0.04], colors=[col],
                                                              labels=[body]), static=True)
            rr.send_columns(f"world/bodies/{body}", indexes=tcol(ts),
                            columns=rr.Transform3D.columns(translation=pos, quaternion=quat), recording=rec)
            for obj in objects or ():
                if obj.mocap_body == body and obj.present_in(traj.name):
                    T = obj.T_body_model
                    rec.log(f"world/bodies/{body}/model", rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]),
                            static=True)
                    rec.log(f"world/bodies/{body}/model/mesh", rr.Mesh3D(
                        vertex_positions=obj.vertices, triangle_indices=obj.faces,
                        albedo_factor=[*col, 160]), static=True)
                    rec.log(f"world/bodies/{body}/model", rr.TransformAxes3D(0.08), static=True)
            rec.log(f"world/paths/{body}", rr.LineStrips3D([pos[::5]], colors=[col], radii=[0.004]), static=True)
            for axis, vals in (("x", pos[:, 0]), ("y", pos[:, 1]), ("yaw", yaw)):
                rec.log(f"plots/{axis}/{body}", rr.SeriesLines(colors=[col], names=[body]), static=True)
                rr.send_columns(f"plots/{axis}/{body}", indexes=tcol(ts),
                                columns=rr.Scalars.columns(scalars=vals), recording=rec)
            span = (ts[0], ts[-1]) if span is None else (min(span[0], ts[0]), max(span[1], ts[-1]))
            if body == robot_body and goals:
                gx, gy = goals[-1][:2]
                rec.log("plots/goal_distance/robot", rr.SeriesLines(colors=[col], names=[f"{body} to goal"]),
                        static=True)
                rr.send_columns("plots/goal_distance/robot", indexes=tcol(ts),
                                columns=rr.Scalars.columns(scalars=np.hypot(pos[:, 0] - gx, pos[:, 1] - gy)),
                                recording=rec)
        if goals and span is not None:
            g = np.array(goals)
            rec.log("world/goals", rr.Points3D(g[:, :3], colors=[(255, 255, 255)], radii=[0.03],
                                               labels=[f"goal {k}" for k in range(len(g))]), static=True)
            rec.log("world/goals/heading", rr.Arrows3D(
                origins=g[:, :3], vectors=np.stack([0.25 * np.cos(g[:, 3]), 0.25 * np.sin(g[:, 3]),
                                                    np.zeros(len(g))], -1), colors=[(255, 255, 255)]),
                static=True)
            ends = np.array([span[0], span[1]])
            for axis, val in (("x", g[-1, 0]), ("y", g[-1, 1]), ("yaw", np.degrees(g[-1, 3]))):
                rec.log(f"plots/{axis}/goal", rr.SeriesLines(colors=[(255, 255, 255)], names=["goal"],
                                                             widths=[1.0]), static=True)
                rr.send_columns(f"plots/{axis}/goal", indexes=tcol(ends),
                                columns=rr.Scalars.columns(scalars=[val, val]), recording=rec)

    if "robot_messages" in mods:
        step("robot messages")
        for topic, entity, names in ((THRUSTER_TOPIC, "plots/actions/thrusters", None),
                                     (RW_TOPIC, "plots/actions/reaction_wheel", ["reaction wheel"])):
            if topic not in traj.topics:
                continue
            ts, arr = traj.message_array(topic)
            sel = window(ts)
            if sel.any():
                arr = arr[sel]
                rec.log(entity, rr.SeriesLines(names=names or [f"thruster {k}" for k in range(arr.shape[1])]),
                        static=True)
                rr.send_columns(entity, indexes=tcol(ts[sel]), columns=rr.Scalars.columns(scalars=arr),
                                recording=rec)
        if TASK_TOPIC in traj.topics:
            ts, msgs = traj.messages(TASK_TOPIC)
            sel = window(ts)
            if sel.any():
                rec.log("plots/actions/task_available", rr.SeriesLines(names=["task available"]), static=True)
                rr.send_columns("plots/actions/task_available", indexes=tcol(ts[sel]),
                                columns=rr.Scalars.columns(scalars=[float(m["data"]) for m, s in zip(msgs, sel) if s]),
                                recording=rec)

    if "imu" in mods:
        step("imu")
        imu = traj.imu()
        sel = window(imu["timestamp_ns"])
        if sel.any():
            ts = imu["timestamp_ns"][sel]
            for name, key in (("gyro_rad_s", "angular_velocity"), ("accel_g", "linear_acceleration")):
                vals = np.stack([imu[f"{key}_{a}"][sel] for a in "xyz"], -1)
                rec.log(f"plots/imu/{name}", rr.SeriesLines(names=[f"{name} {a}" for a in "xyz"]), static=True)
                rr.send_columns(f"plots/imu/{name}", indexes=tcol(ts), columns=rr.Scalars.columns(scalars=vals),
                                recording=rec)

    objs_here = [o for o in objects or () if o.present_in(traj.name)
                 and traj.has("mocap") and o.mocap_body in traj.mocap_bodies]
    if labels and objs_here:
        rec.log("cams", rr.AnnotationContext(
            [rr.AnnotationInfo(id=o.class_id, label=o.name, color=_color(o.mocap_body, k))
             for k, o in enumerate(objs_here)]), static=True)
    for cam, stride in (("firefly", firefly_stride), ("realsense_color", 1)):
        if cam in mods:
            labeler = _labeler(traj, cam, objs_here, max_mask_pixels) if labels else None
            if labeler is not None and "mocap" in mods:
                _log_frustum(rec, traj, cam)
            ts = traj.timestamps(cam)
            idx = np.nonzero(window(ts))[0][::stride]
            for i in step.iter(idx, cam + (" + labels" if labeler else "")):
                rec.set_time(TIMELINE, duration=secs(ts[i]))
                rec.log(f"cams/{cam}", rr.EncodedImage(contents=traj.frame_bytes(cam, i), media_type="image/jpeg"))
                if labeler is not None:
                    labeler(rec, f"cams/{cam}", int(ts[i]))

    if "realsense_depth" in mods:
        ts = traj.timestamps("realsense_depth")
        intr = traj.calibration.cameras.get("realsense_depth")
        meter = 1.0 / (intr.depth_scale if intr is not None and intr.depth_scale else 0.001)
        for i in step.iter(np.nonzero(window(ts))[0], "realsense_depth"):
            rec.set_time(TIMELINE, duration=secs(ts[i]))
            rec.log("cams/realsense_depth", rr.EncodedDepthImage(traj.frame_bytes("realsense_depth", i),
                                                                 media_type="image/png", meter=meter,
                                                                 colormap="turbo"))

    if "events" in mods:
        ev_lo, ev_hi = traj.time_range("events")
        a, b = max(lo, ev_lo), ev_hi if hi is None else min(hi, ev_hi)
        dt = int(1e9 / events_fps)
        intr = traj.calibration.cameras.get("events")
        H, W = (intr.height, intr.width) if intr is not None else EVENT_SENSOR_SIZE
        chunk = 1_000_000_000  # read 1 s at a time, then slice into frames
        for c0 in step.iter(range(a, b, chunk), "events (s)"):
            ev = traj.events(c0, min(c0 + chunk, b))
            edges = np.searchsorted(ev.t, np.arange(c0, min(c0 + chunk, b) + 1, dt))
            for k in range(len(edges) - 1):
                rec.set_time(TIMELINE, duration=secs(c0 + (k + 1) * dt))
                img = render_events(ev[edges[k]:edges[k + 1]], H, W, events_scale)
                rec.log("cams/events", rr.Image(img).compress(jpeg_quality=85))

    if "lidar" in mods:
        ts = traj.timestamps("lidar")
        for i in step.iter(np.nonzero(window(ts))[0][::lidar_stride], "lidar"):
            s = traj.lidar_scan(i)
            pts = np.stack([s["x"], s["y"], s["z"]], -1)
            keep = np.isfinite(pts).all(1) & (np.abs(pts).sum(1) > 0)
            inten = np.log1p(np.asarray(s["intensity"], float)[keep]) / np.log1p(255.0)
            rec.set_time(TIMELINE, duration=secs(ts[i]))
            rec.log("lidar/points", rr.Points3D(pts[keep], colors=_turbo(inten), radii=[0.01]))
    step.done()


#: edges of a box given as 8 corners in x-major product order (index = 4ix + 2iy + iz)
_BOX_EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]


def _labeler(traj: Trajectory, cam: str, objs: list, max_mask_pixels: int):
    """Callable logging the ground truth of one frame, or None if the camera isn't calibrated."""
    from .calibration import MissingCalibrationError
    from .labels import mask_to_bbox, object_pose_in_camera, render_instances

    if not objs:
        return None
    try:
        intr = traj.calibration.intrinsics(cam)
        traj.calibration.T_body_cam(cam)
    except MissingCalibrationError:
        return None
    with_masks = intr.width * intr.height <= max_mask_pixels

    def log(rec, path: str, t_ns: int) -> None:
        items = []
        for o in objs:
            T, ok = object_pose_in_camera(traj, cam, o, t_ns)
            if ok:
                items.append((o, T))
        inst, vis = render_instances(items, intr)
        boxes, cls, names, strips = [], [], [], []
        for (o, T), m in zip(items, vis):
            box = mask_to_bbox(m)
            if box is None:
                continue
            boxes.append(box)
            cls.append(o.class_id)
            names.append(o.name)
            kp = o.click_points
            if len(kp) == 8:
                uv, okp = intr.project((T[:3, :3] @ kp.T).T + T[:3, 3])
                strips += [uv[[a, b]] for a, b in _BOX_EDGES if okp[a] and okp[b]]
        if not boxes:  # nothing in view: clear, or the last labels would linger on screen
            rec.log(f"{path}/labels", rr.Clear(recursive=True))
            return
        rec.log(f"{path}/labels/boxes", rr.Boxes2D(array=np.array(boxes), array_format=rr.Box2DFormat.XYXY,
                                                   class_ids=cls, labels=names))
        rec.log(f"{path}/labels/wireframe", rr.LineStrips2D(strips, colors=[(255, 255, 255)], radii=[0.5]))
        if with_masks:
            seg = np.zeros(inst.shape, np.uint8)
            for k, (o, _) in enumerate(items):
                seg[inst == k + 1] = o.class_id
            rec.log(f"{path}/labels/mask", rr.SegmentationImage(seg, opacity=0.45))

    return log


def _log_frustum(rec, traj: Trajectory, cam: str) -> None:
    """Static camera frustum under its rig body, drawn in the 3D mocap view."""
    cal = traj.calibration
    intr, T = cal.intrinsics(cam), cal.T_body_cam(cam)
    path = f"world/bodies/{cal.reference_body}/{cam}"
    rec.log(path, rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]), static=True)
    rec.log(path, rr.Pinhole(image_from_camera=intr.K, resolution=[intr.width, intr.height],
                             camera_xyz=rr.ViewCoordinates.RDF, image_plane_distance=0.15), static=True)


def _info_text(traj: Trajectory) -> str:
    try:
        from .quality import trajectory_report

        r = trajectory_report(traj)
        lines = [f"# {traj.name}", f"domain: {r.domain}, duration {r.duration_s:.1f} s", ""]
        if r.goals:
            lines.append(f"final goal error: {r.final_pos_error_m * 100:.1f} cm, "
                         f"{r.final_yaw_error_deg:.1f} deg")
        lines += ["", "**warnings**" if r.warnings else "no warnings"] + [f"- {w}" for w in r.warnings]
        return "\n".join(lines)
    except Exception as e:  # never block visualization on the report
        return f"# {traj.name}\n(report failed: {e})"


class _Progress:
    def __init__(self, enabled: bool, name: str):
        self.enabled, self.name = enabled, name

    def __call__(self, what: str) -> None:
        if self.enabled:
            print(f"[{self.name}] logging {what}", flush=True)

    def iter(self, it, what: str):
        if not self.enabled:
            return it
        from tqdm import tqdm

        return tqdm(it, desc=f"[{self.name}] {what}", leave=False)

    def done(self) -> None:
        if self.enabled:
            print(f"[{self.name}] done", flush=True)


def visualize(
    trajs: list[Trajectory],
    mode: str = "spawn",
    save_dir: str | None = None,
    web_port: int | None = None,
    grpc_port: int | None = None,
    **log_kw,
) -> list[rr.RecordingStream]:
    """Log trajectories and show them.

    ``mode``: ``"spawn"`` (native viewer window), ``"web"`` (viewer in the browser, served
    from this process; blocks until Ctrl-C) or ``"save"`` (one ``<trajectory>.rrd`` per
    trajectory in ``save_dir``; open later with ``rerun *.rrd``).
    """
    from pathlib import Path

    recs, url = [], None
    for traj in trajs:
        rec = rr.RecordingStream(APP_ID, recording_id=traj.name)
        bp = blueprint(traj, log_kw.get("modalities"))
        if mode == "spawn":
            rec.spawn(default_blueprint=bp, memory_limit="75%")
        elif mode == "save":
            out = Path(save_dir or ".")
            out.mkdir(parents=True, exist_ok=True)
            rec.save(out / f"{traj.name}.rrd", default_blueprint=bp)
        elif mode == "web":
            if url is None:
                import os

                url = rec.serve_grpc(grpc_port=grpc_port, default_blueprint=bp, server_memory_limit="8GiB")
                wp = web_port or 9090
                has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
                rr.serve_web_viewer(web_port=wp, connect_to=url, open_browser=has_display)
                from urllib.parse import quote

                gp = int(url.rsplit(":", 1)[1].split("/")[0])
                # the viewer only knows where the data is through ?url=...; without it, it shows
                # Rerun's landing page
                page = f"http://localhost:{wp}/?url={quote(url.replace('127.0.0.1', 'localhost'), safe='')}"
                print(f"\nopen this URL (the ?url=... part is required):\n  {page}")
                print(f"  remote machine? first, on your laptop:  ssh -L {wp}:localhost:{wp} -L {gp}:localhost:{gp} <this host>\n")
            else:
                rec.connect_grpc(url, default_blueprint=bp)
        else:
            raise ValueError(f"unknown mode {mode!r}")
        log_trajectory(traj, rec, **log_kw)
        rec.flush()
        recs.append(rec)
    if mode == "web":
        print("viewer running in your browser; Ctrl-C to stop")
        try:
            import time

            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    return recs
