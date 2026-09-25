"""``pooh`` command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .calibration import MissingCalibrationError
from .hub import DEFAULT_REPO_ID, default_root


def _dataset(args):
    from .dataset import PoohDataset

    return PoohDataset(args.root, args.repo)


def cmd_download(args) -> None:
    from .hub import download

    root = download(args.repo, args.root, args.trajectory, args.modality, args.revision)
    print(f"dataset at {root}")


def cmd_info(args) -> None:
    ds = _dataset(args)
    print(f"{ds.repo_id} @ {ds.root}")
    print("splits: " + ", ".join(f"{k}={v}" for k, v in ds.splits.items()))
    for t in ds.trajectory_names:
        info = ds.trajectory_info(t)
        local = ds[t].modalities
        counts = info["counts"]
        mods = ", ".join(f"{m}{'' if m in local else ' (remote)'}={n:,}" for m, n in counts.items())
        print(f"  {t} [{info['domain']}] {mods or 'local: ' + ', '.join(local)}")
    print(f"objects: {sorted(ds.objects) or 'none'}")


def cmd_check(args) -> None:
    """Report which calibration / model inputs are present (see DATA_REQUIREMENTS.md)."""
    from .trajectory import CAMERAS

    ds = _dataset(args)
    ok = lambda b: "ok " if b else "-- "  # noqa: E731
    print(f"splits.json        {ok((ds.root / 'splits.json').exists())}")
    print(f"models/objects.yaml {ok(bool(ds.objects))} {sorted(ds.objects)}")
    for name, obj in ds.objects.items():
        ident = bool(np.allclose(obj.T_body_model, np.eye(4)))
        size = obj.vertices.max(0) - obj.vertices.min(0)
        print(f"  {name}: size={' x '.join(f'{v:.3f}' for v in size)} m  "
              f"mocap_body={obj.mocap_body} verts={len(obj.vertices)} "
              f"T_body_model={'IDENTITY (placeholder?)' if ident else 'set'} "
              f"trajectories={obj.trajectories or 'all'}")
    for t in ds.trajectory_names:
        traj = ds[t]
        cal = traj.calibration
        print(f"{t}  (local modalities: {', '.join(traj.modalities) or 'none'})")
        body_ok = cal.reference_body is not None
        if body_ok and traj.has("mocap"):
            body_ok = cal.reference_body in traj.mocap_bodies
        print(f"  reference_body     {ok(body_ok)} {cal.reference_body}")
        for cam in CAMERAS:
            intr = cal.cameras.get(cam)
            src = "" if intr is None else f" source={intr.source}"
            print(f"  {cam:17s}  intrinsics {ok(intr is not None)} extrinsics "
                  f"{ok(cam in cal.extrinsics)}{src}")


def _task(args):
    from .tasks import ObjectDetection

    ds = _dataset(args)
    return ObjectDetection(
        ds, args.split, camera=args.camera, objects=args.objects, stride=args.stride,
        undistort=args.undistort or args.format == "bop",
        image_size=tuple(args.size) if args.size else None,
        masks=args.format != "yolo",
    )


def cmd_export(args) -> None:
    from .export import export_bop, export_coco, export_yolo

    fn = {"bop": export_bop, "coco": export_coco, "yolo": export_yolo}[args.format]
    out = fn(_task(args), args.out, split_name=args.split)
    print(f"wrote {args.format} to {out}")


def cmd_extract(args) -> None:
    from .export import extract_frames

    ds = _dataset(args)
    for t in args.trajectory or ds.local_trajectory_names:
        for m in args.modality:
            print(extract_frames(ds[t], m, args.out, args.stride))


def cmd_quality(args) -> None:
    import json

    from .quality import dataset_report, format_reports

    ds = _dataset(args)
    reports = dataset_report(ds, args.trajectory, robot_body=args.body,
                             goal_tolerance_m=args.goal_tol_cm / 100)
    print(format_reports(reports))
    if args.json:
        args.json.write_text(json.dumps([r.to_dict() for r in reports], indent=1, default=float))
        print(f"wrote {args.json}")


def cmd_viz(args) -> None:
    if args.download:  # before opening the dataset: the local folder may not exist yet
        from .hub import download

        mods = args.modality or [m for m in VIZ_MODALITIES if m != "firefly"]
        args.root = download(args.repo, args.root, args.trajectory, mods, args.revision)
    ds = _dataset(args)
    names = args.trajectory or ds.local_trajectory_names
    from .viz import visualize

    trajs = [ds[t] for t in names]
    if not any(t.modalities for t in trajs):
        raise FileNotFoundError(f"nothing downloaded for {names}; add --download")
    visualize(
        trajs, mode="save" if args.save else ("web" if args.web else "spawn"), save_dir=args.save,
        web_port=args.web_port, grpc_port=args.grpc_port, modalities=args.modality, robot_body=args.body,
        start_s=args.start, end_s=args.end, firefly_stride=args.firefly_stride,
        events_fps=args.events_fps, events_scale=args.events_scale, lidar_stride=args.lidar_stride,
        objects=list(ds.objects.values()), labels=not args.no_labels,
    )


def cmd_calibrate(args) -> None:
    import shutil

    from . import calibrate as cal

    ds = _dataset(args)
    if args.object not in ds.objects:
        raise FileNotFoundError(f"object {args.object!r} not in models/objects.yaml (have {sorted(ds.objects)})")
    obj = ds.objects[args.object]
    if args.box_keypoints:
        lo, hi = np.array(args.box_keypoints[:3]), np.array(args.box_keypoints[3:])
        obj.keypoints = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    elif obj.keypoints is None:
        print(f"note: {obj.name} has no `keypoints` in objects.yaml; using the corners of the whole "
              "mesh's bounding box. Pass --box-keypoints if that includes more than the clickable body.")
    if args.fixed_model and args.refine_model:
        raise SystemExit("error: --fixed-model and --refine-model are mutually exclusive")
    # A T_body_model measured with `pooh align-markers` is more reliable than one solved from
    # clicks (which trades off against the camera position): keep it unless asked otherwise.
    args.fixed_model = args.fixed_model or (obj.motive_file is not None and not args.refine_model)
    if args.fixed_model and not args.refine_model and obj.motive_file:
        print(f"{obj.name}: T_body_model comes from {obj.motive_file}; solving the camera only "
              "(pass --refine-model to also adjust the object)")
    rig_body = args.body
    out = args.out
    clicks_path = args.clicks or out / "calibration" / f"clicks_{args.camera}.json"
    if clicks_path.exists():
        clicks = cal.load_clicks(clicks_path)
        print(f"loaded {len(clicks['frames'])} clicked frames from {clicks_path}")
    else:
        clicks = {"version": 1, "camera": args.camera, "object": obj.name, "rig_body": rig_body,
                  "keypoints": obj.click_points.tolist(), "frames": []}

    if args.add or not clicks["frames"]:

        done = {(f["trajectory"], f["index"]) for f in clicks["frames"]}
        # fresh run: --frames in total; with --add: --frames more
        need = args.frames if args.add else max(args.frames - len(done), 1)
        cands = cal.select_frames(ds, args.camera, obj, rig_body, need, args.trajectory, exclude=done)
        if not cands:
            if done:
                raise SystemExit(f"error: every usable {args.camera} frame is already clicked "
                                 f"({len(done)} frames); nothing left to add")
            raise FileNotFoundError(
                f"no frames with {args.camera} + mocap ({rig_body}, {obj.mocap_body}) for {obj.name}; "
                f"download them: pooh download -m {args.camera} -m mocap")

        def frames():
            for traj, idx, t in cands:
                yield (traj, idx, t), f"{traj}  {args.camera} frame {idx}", ds[traj].frame(args.camera, idx)

        cache = {}

        def hint(key):
            n = len(clicks["frames"])
            if n < 3:
                return None
            if cache.get("n") != n:
                try:
                    obs = cal.observations_from_clicks(ds, clicks, obj)
                    cache.update(n=n, res=cal.solve(obs, obj, not args.fixed_model, n_rotations=1500, top_k=3))
                except Exception:  # a hint is optional
                    return None
            traj, idx, t = key
            T_r, _ = ds[traj].body_pose(rig_body, t)
            T_o, _ = ds[traj].body_pose(obj.mocap_body, t)
            o = cal.Observation(traj, idx, t, np.zeros((0, 2)), T_r, T_o,
                                ds[traj].calibration.intrinsics(args.camera))
            uv, ok = cal.predict(cache["res"].T_body_cam, cache["res"].T_body_model, o, obj.click_points)
            return uv[ok]

        def on_accept(key, pts):
            traj, idx, t = key
            clicks["frames"].append({"trajectory": traj, "index": idx, "t_ns": t, "points": pts.tolist()})
            cal.save_clicks(clicks_path, clicks)  # saved after every frame: safe to quit and resume

        from .calibrate_web import WebClickSession

        WebClickSession(frames(), need, hint=hint, on_accept=on_accept, port=args.port,
                        open_browser=not args.no_browser).run()
        if clicks["frames"]:
            print(f"{len(clicks['frames'])} clicked frames saved to {clicks_path}")

    obs = cal.observations_from_clicks(ds, clicks, obj)
    if len(obs) < 2:
        raise SystemExit(f"error: need clicked corners in at least 2 frames with valid mocap "
                         f"(have {len(obs)}); run `pooh calibrate ... --add` to click more")
    print(f"solving from {sum(len(o.points) for o in obs)} clicks in {len(obs)} frames ...")
    res = cal.solve(obs, obj, refine_model=not args.fixed_model, refine_tilt=args.refine_tilt)
    print(f"RMS reprojection error: {res.rms_px:.2f} px")
    for o, e in zip(obs, res.per_frame_rms_px):
        print(f"  {o.trajectory} frame {o.index:5d}: {e:6.2f} px  ({len(o.points)} clicks)")
    T = res.T_body_cam
    print(f"T_body_cam ({rig_body} -> {args.camera}): t = {np.round(T[:3, 3], 4).tolist()} m")
    if not args.fixed_model:
        print(f"{obj.name} model correction: yaw {res.yaw_correction_deg:+.1f} deg, tilt {res.tilt_correction_deg:.1f} deg, "
              f"translation {np.round(res.translation_correction_m * 1000, 1).tolist()} mm")
    for n in res.notes:
        print(f"  ! {n}")
    suspicious = []
    if res.rms_px > args.max_rms:
        suspicious.append(f"RMS {res.rms_px:.1f} px > {args.max_rms:g} px")
    if np.linalg.norm(T[:3, 3]) > 0.5:
        suspicious.append(f"camera {np.linalg.norm(T[:3, 3]):.2f} m from the {rig_body} pivot (> 0.5 m)")
    if not args.fixed_model and np.linalg.norm(res.translation_correction_m) * 100 > args.max_model_shift_cm:
        suspicious.append(f"{obj.name} shifted {np.linalg.norm(res.translation_correction_m) * 100:.0f} cm "
                          f"(> {args.max_model_shift_cm:g} cm): is T_body_model's pivot assumption right?")
    if suspicious:
        print("\n  !! result looks wrong: " + "; ".join(suspicious))
        print("  !! check the overlays, re-click bad frames (delete them from the clicks file) or add "
              "more with --add. Not applied; pass --force to apply anyway.")

    rejected = bool(suspicious) and not args.force
    written = [] if rejected else cal.write_outputs(ds, res, args.camera, obj, rig_body, out, args.also,
                                                    not args.fixed_model)
    ov_dir = out / f"overlays_{args.camera}"
    ov_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    for o in obs:
        img = cal.render_overlay(ds[o.trajectory].frame(args.camera, o.index), o, res, obj)
        Image.fromarray(img).save(ov_dir / f"{o.trajectory}_{o.index:06d}.png")
    if written:
        print(f"wrote {', '.join(str(p) for p in written)}")
    print(f"overlays (green = rendered mask outline, red = your clicks): {ov_dir}/")
    if rejected:
        print("  (no calibration files written for a rejected result)")
        return
    if not args.no_apply:
        for p in written:
            dst = ds.root / p.relative_to(out)
            if dst.exists() and not dst.with_suffix(dst.suffix + ".orig").exists():
                shutil.copy(dst, dst.with_suffix(dst.suffix + ".orig"))
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(p, dst)
        print(f"applied to the local dataset at {ds.root} (originals kept as *.orig)")
    print("\nshare it with everyone (the local copy is overwritten by `pooh download` until uploaded):")
    print(f"  hf upload {ds.repo_id} {out} . --repo-type dataset --exclude 'overlays_*/*'")


def _parse_axes(spec: str) -> np.ndarray:
    """"x,-z,y" -> 3x3 matrix M with out = M @ in (each output axis = a signed input axis)."""
    M = np.zeros((3, 3))
    for i, tok in enumerate(spec.replace(" ", "").split(",")):
        sign = -1.0 if tok.startswith("-") else 1.0
        M[i, "xyz".index(tok.lstrip("+-"))] = sign
    if len(spec.split(",")) != 3 or abs(abs(np.linalg.det(M)) - 1) > 1e-9:
        raise SystemExit(f"error: bad --cad-axes {spec!r}; expected e.g. 'x,-z,y'")
    return M


def cmd_align_markers(args) -> None:
    import shutil

    from . import calibrate as cal

    ds = _dataset(args)
    if args.object not in ds.objects:
        raise FileNotFoundError(f"object {args.object!r} not in models/objects.yaml (have {sorted(ds.objects)})")
    motive_path = args.motive if args.motive.exists() else ds.root / args.motive
    motive = cal.read_motive_markers(motive_path)
    scale = {"mm": 1e-3, "cm": 1e-2, "m": 1.0}[args.cad_units]
    cad = np.array(args.cad_marker, float) * scale @ _parse_axes(args.cad_axes).T
    T, r, idx = cal.align_markers(motive, cad, _parse_axes(args.motive_to_body))
    print(f"{len(motive)} Motive markers, {len(cad)} CAD markers -> matched Motive markers {idx}")
    print(f"fit residuals: {np.round(r * 1000, 2).tolist()} mm (max {r.max() * 1000:.2f} mm)")
    pivot = -T[:3, :3].T @ T[:3, 3]
    print(f"mocap pivot in CAD frame: {np.round(pivot * 1000, 1).tolist()} mm")
    if r.max() > args.max_residual_mm / 1000:
        raise SystemExit(f"error: residual above {args.max_residual_mm} mm: wrong units, axes or marker list?")
    note = (f"from {len(cad)} CAD marker positions matched to {motive_path.name} "
            f"(max residual {r.max() * 1000:.1f} mm)")
    path = cal.write_object_transform(ds, args.object, T, args.out, note,
                                      {"cad_markers": np.round(cad, 6).tolist(),
                                       "motive_file": f"models/{motive_path.name}"})
    print(f"wrote {path}")
    dst = ds.root / "models" / "objects.yaml"
    if not args.no_apply:
        if dst.exists() and not dst.with_suffix(".yaml.orig").exists():
            shutil.copy(dst, dst.with_suffix(".yaml.orig"))
        shutil.copy(path, dst)
        print(f"applied to the local dataset at {ds.root} (original kept as objects.yaml.orig)")
    print(f"share it:  hf upload {ds.repo_id} {path} models/objects.yaml --repo-type dataset")


VIZ_MODALITIES = ["mocap", "robot_messages", "imu", "lidar", "realsense_color", "realsense_depth",
                  "events", "firefly"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="pooh", description="Download and process the POOH multimodal robotics dataset.")
    p.add_argument("--repo", default=DEFAULT_REPO_ID, help="Hub dataset repo id (default: %(default)s)")
    p.add_argument("--root", type=Path, default=None,
                   help=f"local dataset folder (default: {default_root()}, or $POOH_DATA_DIR)")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="download (part of) the dataset; resumes / updates")
    d.add_argument("-t", "--trajectory", action="append", help="repeatable; default all")
    d.add_argument("-m", "--modality", action="append", help="repeatable; default all")
    d.add_argument("--revision", help="commit sha / tag to pin")
    d.set_defaults(fn=cmd_download)

    sub.add_parser("info", help="list trajectories, modalities, splits").set_defaults(fn=cmd_info)
    sub.add_parser("check", help="which calibration / CAD inputs are missing").set_defaults(fn=cmd_check)

    e = sub.add_parser("export", help="export detection / segmentation / pose labels")
    e.add_argument("format", choices=["bop", "coco", "yolo"])
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--split", default="train")
    e.add_argument("--camera", default="firefly")
    e.add_argument("--objects", nargs="*")
    e.add_argument("--stride", type=int, default=1, help="use every N-th frame")
    e.add_argument("--undistort", action="store_true", help="always on for bop")
    e.add_argument("--size", type=int, nargs=2, metavar=("W", "H"))
    e.set_defaults(fn=cmd_export)

    x = sub.add_parser("extract-frames", help="unpack image frames to files")
    x.add_argument("--out", type=Path, required=True)
    x.add_argument("-t", "--trajectory", action="append")
    x.add_argument("-m", "--modality", action="append", required=True)
    x.add_argument("--stride", type=int, default=1)
    x.set_defaults(fn=cmd_extract)

    q = sub.add_parser("quality", help="per-trajectory health report (good / bad trajectories)")
    q.add_argument("-t", "--trajectory", action="append", help="repeatable; default all local")
    q.add_argument("--body", default="floatingplatform", help="mocap body of the robot")
    q.add_argument("--goal-tol-cm", type=float, default=10.0)
    q.add_argument("--json", type=Path, help="also write the full report as JSON")
    q.set_defaults(fn=cmd_quality)

    v = sub.add_parser("viz", help="synchronized playback in the Rerun viewer (needs [viz])")
    v.add_argument("-t", "--trajectory", action="append", help="repeatable; default all local")
    v.add_argument("-m", "--modality", action="append", choices=VIZ_MODALITIES,
                   help="repeatable; default everything downloaded")
    v.add_argument("--download", action="store_true",
                   help="fetch the modalities first (default: all but firefly)")
    v.add_argument("--revision")
    v.add_argument("--body", default="floatingplatform")
    v.add_argument("--start", type=float, default=0.0, help="seconds from trajectory start")
    v.add_argument("--end", type=float, default=None)
    v.add_argument("--firefly-stride", type=int, default=2)
    v.add_argument("--events-fps", type=float, default=20.0)
    v.add_argument("--events-scale", type=float, default=0.5)
    v.add_argument("--lidar-stride", type=int, default=1)
    v.add_argument("--no-labels", action="store_true",
                   help="don't draw ground-truth masks / boxes on calibrated cameras (faster)")
    out = v.add_mutually_exclusive_group()
    out.add_argument("--web", action="store_true", help="open the viewer in a browser")
    out.add_argument("--save", metavar="DIR", help="write <trajectory>.rrd files instead")
    v.add_argument("--web-port", type=int, help="web viewer port (default 9090)")
    v.add_argument("--grpc-port", type=int, help="data stream port for --web (default 9876)")
    v.set_defaults(fn=cmd_viz)

    c = sub.add_parser("calibrate", help="camera extrinsics + object alignment from clicked corners (needs [calib])")
    c.add_argument("--camera", default="realsense_color", choices=["realsense_color", "firefly"])
    c.add_argument("--object", default="cubesat")
    c.add_argument("--body", default="SensorStack", help="mocap body the cameras are mounted on")
    c.add_argument("-t", "--trajectory", action="append", help="repeatable; default all local")
    c.add_argument("--frames", type=int, default=10,
                   help="frames to click (default %(default)s); with --add: how many more")
    c.add_argument("--out", type=Path, default=Path("pooh_calibration"),
                   help="output folder, laid out like the Hub repo (default %(default)s)")
    c.add_argument("--clicks", type=Path, help="clicks file (default <out>/calibration/clicks_<camera>.json)")
    c.add_argument("--add", action="store_true", help="click more frames on top of an existing clicks file")
    c.add_argument("--fixed-model", action="store_true",
                   help="only solve the camera; keep T_body_model (default if it came from align-markers)")
    c.add_argument("--refine-model", action="store_true",
                   help="also adjust T_body_model, even if it came from align-markers")
    c.add_argument("--box-keypoints", type=float, nargs=6, metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1"),
                   help="model-frame box whose 8 corners are the clickable keypoints (metres)")
    c.add_argument("--also", action="append", default=[],
                   help="also give this camera the same extrinsics (e.g. realsense_depth if aligned to color)")
    c.add_argument("--no-apply", action="store_true", help="don't copy results into the local dataset")
    c.add_argument("--max-rms", type=float, default=10.0, help="refuse to apply above this RMS (px)")
    c.add_argument("--refine-tilt", action="store_true",
                   help="also correct the object's tilt (not just its rotation about the vertical)")
    c.add_argument("--max-model-shift-cm", type=float, default=5.0,
                   help="refuse to apply if the object model moves more than this (default %(default)s)")
    c.add_argument("--force", action="store_true", help="apply even if the result looks implausible")
    c.add_argument("--port", type=int, default=8765, help="port of the web page (default %(default)s)")
    c.add_argument("--no-browser", action="store_true", help="don't try to open a browser automatically")
    c.set_defaults(fn=cmd_calibrate)

    a = sub.add_parser("align-markers", help="object T_body_model from Motive markers + their CAD positions")
    a.add_argument("--object", default="cubesat")
    a.add_argument("--motive", type=Path, required=True,
                   help="Motive rigid body export (.motive); path, or relative to the dataset root")
    a.add_argument("--cad-marker", type=float, nargs=3, action="append", required=True, metavar=("X", "Y", "Z"),
                   help="a marker centre in CAD coordinates; repeat for each (>= 3)")
    a.add_argument("--cad-units", choices=["mm", "cm", "m"], default="mm")
    a.add_argument("--cad-axes", default="x,y,z",
                   help="mesh-frame axes in terms of the given coordinates, e.g. 'x,-z,y' for an "
                        "Onshape Z-up part exported Y-up (default %(default)s)")
    a.add_argument("--motive-to-body", default="x,-z,y",
                   help="ROS body axes in terms of Motive's Y-up body axes (default %(default)s)")
    a.add_argument("--max-residual-mm", type=float, default=3.0)
    a.add_argument("--out", type=Path, default=Path("pooh_calibration"))
    a.add_argument("--no-apply", action="store_true")
    a.set_defaults(fn=cmd_align_markers)

    args = p.parse_args(argv)
    try:
        args.fn(args)
    except (MissingCalibrationError, FileNotFoundError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
