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
        print(f"  {name}: mocap_body={obj.mocap_body} verts={len(obj.vertices)} "
              f"T_body_model={'IDENTITY (placeholder?)' if ident else 'set'}")
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
        web_port=args.web_port, modalities=args.modality, robot_body=args.body,
        start_s=args.start, end_s=args.end, firefly_stride=args.firefly_stride,
        events_fps=args.events_fps, events_scale=args.events_scale, lidar_stride=args.lidar_stride,
    )


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
    out = v.add_mutually_exclusive_group()
    out.add_argument("--web", action="store_true", help="open the viewer in a browser")
    out.add_argument("--save", metavar="DIR", help="write <trajectory>.rrd files instead")
    v.add_argument("--web-port", type=int)
    v.set_defaults(fn=cmd_viz)

    args = p.parse_args(argv)
    try:
        args.fn(args)
    except (MissingCalibrationError, FileNotFoundError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
