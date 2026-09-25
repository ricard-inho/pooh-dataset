# Calibration and model files: status and what is still needed

`pooh-dataset` can download, read and synchronize every modality. Ground-truth **labels**
(6D pose, masks, boxes, optical flow) are computed from motion capture, and they need the
calibration and model files below. A label task for a camera or object that is not
calibrated yet raises `MissingCalibrationError`, pointing here.

**Today:** labels work for the `cubesat` in the **RealSense colour** camera in trajectories
007–009. The other cameras still need extrinsics (see the checklist).

Run `pooh check` at any time to see what is present:

```
$ pooh check
models/objects.yaml ok  ['cubesat']
  cubesat: size=0.100 x 0.134 x 0.100 m  mocap_body=cubesat verts=27354 T_body_model=set ...
trajectory_007
  reference_body     ok  SensorStack
  firefly            intrinsics ok  extrinsics --  source=supplement:...
  realsense_color    intrinsics ok  extrinsics ok  source=ros_camera_info
  realsense_depth    intrinsics ok  extrinsics --
  events             intrinsics --  extrinsics --
```

All files go in the **Hub dataset repo** (`r3m3c3/off_test`), not in this Python package,
so everyone gets them with `pooh download` and they can be versioned with the data.

Conventions used everywhere:

- `T_a_b` is a 4x4 homogeneous transform that maps points in frame `b` to frame `a`
  (it is the pose of `b` expressed in `a`). Metres, right-handed.
- Quaternions are `[qx, qy, qz, qw]` (the mocap column order).
- Camera optical frames follow ROS: +z forward, +x right, +y down.
- A transform can be written as 16 row-major numbers **or** as
  `{translation: [x, y, z], rotation_xyzw: [qx, qy, qz, qw]}`.

---

## Checklist

| # | Item | File | Needed for | Status |
|---|------|------|------------|--------|
| 1 | Camera extrinsics: `SensorStack` body → each camera optical frame | `calibration/rig.yaml` | labels in that camera | **realsense_color done** (`pooh calibrate`, 8.8 px RMS); firefly, realsense_depth, events missing |
| 2 | Event camera intrinsics (1280x720) | `calibration/rig.yaml` or `<traj>.yaml` | event labels / flow | missing |
| 3 | Firefly intrinsics from a real calibration (current ones are `supplement:`) | `calibration/rig.yaml` | Firefly labels | verify |
| 4 | Cubesat CAD mesh | `models/CubeSat_U1.obj` | masks, boxes, BOP | **done** |
| 5 | Cubesat marker → CAD transform `T_body_model` | `models/objects.yaml` | all cubesat labels | **done** (`pooh align-markers`, 1.2 mm) |
| 6 | Floating platform mesh + `T_body_model` (optional, for occlusion + a 2nd class) | `models/objects.yaml` | better masks | optional |
| 7 | Per-sensor time offsets, if any sensor is not on the host clock | `calibration/rig.yaml` | exact sync | optional |
| 8 | RealSense depth scale + which frame depth is aligned to | `calibration/<traj>.yaml` | flow | verify |
| 9 | Official train / val / test split | `splits.json` | comparable results | missing |
| 10 | Per-trajectory metadata (real / synthetic, task type) | `manifest.json` | filtering | optional |

---

## 1. Rig extrinsics — `calibration/rig.yaml`

The cameras are rigidly mounted on the rig tracked by the `SensorStack` mocap body. For each
camera we need `T_body_cam`: the pose of the camera **optical** frame in the SensorStack
mocap rigid-body frame (the frame Motive/OptiTrack reports for that body, i.e. what the
`x, y, z, qx..qw` columns in `mocap` describe).

```yaml
# calibration/rig.yaml  — shared by all trajectories unless a <trajectory>.yaml overrides it
extrinsics:
  reference_body: SensorStack          # must match a `body` value in the mocap table
  cameras:
    firefly:
      T_body_cam: {translation: [0.05, 0.00, 0.02], rotation_xyzw: [-0.5, 0.5, -0.5, 0.5]}
    realsense_color:
      T_body_cam: [1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1]
    realsense_depth:
      T_body_cam: ...
    events:
      T_body_cam: ...
```

How to get it: **`pooh calibrate --camera <camera>`** (see the README) solves it from cube
corners clicked in existing frames, keeping the cube's measured `T_body_model` fixed. That is
how `realsense_color` was done: 40 frames, 8.8 px RMS overall and about 4 px on far frames.
Close-up frames are the least accurate, mostly because the foil is over-exposed there.
Alternatives:

- **Hand-eye calibration**: record a checkerboard / AprilTag board that also carries mocap
  markers (or is static in the mocap world), then solve `AX = XB` (e.g. OpenCV
  `calibrateHandEye`) between the SensorStack mocap poses and the board poses per camera.
- **Camera-to-camera** (Kalibr, OpenCV stereo) for all cameras + a single hand-eye for one of
  them; the package composes them.
- **From CAD**: `pooh align-markers` with `models/SensorStack.motive` and the SensorStack
  CAD's marker positions gives the rig body → CAD transform; the camera pose in that CAD then
  gives `T_body_cam`.

Note: the RealSense is mounted upside down relative to the cube (the marker holder, which is
under the cube, appears on top in its images). `T_body_cam` accounts for that; nothing to do.

If the rig is re-assembled, put a new `extrinsics:` block in that trajectory's
`calibration/<trajectory>.yaml`: it overrides `rig.yaml` key by key.

## 2. Event camera intrinsics

The calibration files have no `events` entry. Add one in the same schema as the others
(the event camera needs a regular calibration, e.g. with a blinking-pattern screen and
Prophesee's calibration tool, or E2VID reconstructions + a checkerboard):

```yaml
events:
  source: prophesee_calibration          # anything not starting with "supplement"
  width: 1280
  height: 720
  distortion_model: plumb_bob
  D: [k1, k2, p1, p2, k3]
  K: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
```

## 3. Firefly intrinsics

The Firefly entry is marked `source: supplement:/data/calibration.json (driver did not
publish real intrinsics)`. If that file is a real calibration at 3072x2048, just change the
`source` to say where it came from. The distortion is significant (k1 = -0.23), so labels
depend on it being right. The package handles it: projection is exact, and BOP export
undistorts images.

## 4–6. Object models — `models/objects.yaml`

The current entry (generated by `pooh align-markers`):

```yaml
cubesat:
  class_id: 1                   # 1-based; BOP obj_id / COCO category_id (YOLO uses class_id-1)
  mocap_body: cubesat           # body name in the mocap table
  mesh: models/CubeSat_U1.obj   # Onshape export, metres, Y-up; origin = centre of the face
                                #   opposite the marker holder
  mesh_scale: 1.0               # CAD units -> metres
  trajectories: [trajectory_007, trajectory_008, trajectory_009]   # where the cube was present
  T_body_model: {translation: [...], rotation_xyzw: [...]}   # p_body = R @ p_model + t
  cad_markers: [...]            # the CAD marker centres it was fitted from (mesh frame, m)
  motive_file: models/CubeSat_U1.motive
  keypoints: [...]              # the cube body's 8 corners, clicked by `pooh calibrate`
```

**How `T_body_model` was measured.** The Motive export (`models/CubeSat_U1.motive`) lists the
8 LED positions in the rigid body's own frame (origin = LED centroid, which is the pivot).
The 4 corner LEDs' positions in the CAD (from Onshape) were matched to them with a rigid fit
(Kabsch, 0.6–1.2 mm residuals):

```bash
pooh align-markers --motive models/CubeSat_U1.motive --cad-axes x,-z,y \
  --cad-marker 46.077 -46.03447 -117.99807 --cad-marker -47.35423 -47.40068 -121.5 \
  --cad-marker -47.35423 47.39831 -121.5 --cad-marker 47.44476 47.39831 -121.5
```

- `--cad-axes x,-z,y`: Onshape is Z-up, the exported mesh is Y-up.
- Motive stores bodies Y-up and the ROS mocap stream is Z-up. The standard conversion
  `(x, y, z)_ros = (x, -z, y)_motive` is the default (`--motive-to-body`), and it is the one
  consistent with the lab. The marker holder is on the cube's **bottom** face: the cube sits
  on it, tilted about 23° from vertical. The cube centre is 6.8 cm above the LED centroid.

For a new object: export its rigid body from Motive, measure its marker centres in the CAD,
and run the same command. Without a CAD, `box: [sx, sy, sz]` (metres, centred on the model
origin) instead of `mesh` gives correct boxes and masks for a cuboid.

Meshes should be simplified to about 10k faces or fewer. Masks are rasterised per triangle,
so huge CAD meshes slow down label generation. Masks are hole-filled, because hollow CAD
shells can be see-through where the real object is solid.

## 7. Time offsets (optional)

Every stream is already on the recording host's clock. If a sensor turns out to have a
constant latency (e.g. the event anchoring is a few ms off, or a camera's exposure midpoint
differs from its stamp), add:

```yaml
time_offsets_ns:
  events: 0
  firefly: 0
  realsense_color: 0
  realsense_depth: 0
  mocap: 0
```

The offset is **added** to that sensor's timestamps everywhere in the package.

## 8. Depth

- `depth_scale` (metres per raw unit). The package assumes 0.001 (RealSense default) if
  absent: `realsense_depth: {depth_scale: 0.001, ...}`.
- The README says depth is "aligned to color", but `realsense_depth` has its own `K` and
  `frame_id: camera_depth_optical_frame`. Which is it? If depth is aligned to color, the
  depth entry should use the colour intrinsics and `T_body_cam` should equal the colour one.

## 9. Splits — `splits.json`

```json
{
  "version": 1,
  "train": ["trajectory_007", "trajectory_008"],
  "val":   [],
  "test":  ["trajectory_009"]
}
```

Splits are by whole trajectories: consecutive frames are near-duplicates, so frame-level
splits leak between train and test. Without this file the package falls back to a stable
hash of the trajectory name and prints a warning. Add new trajectories to the file as
they are uploaded, and never move old ones between splits. Bump `version` whenever the
file changes, so results can cite it.

## 10. Trajectory metadata — `manifest.json`

Add an optional `trajectory_info` block next to the existing `trajectories` counts:

```json
"trajectory_info": {
  "trajectory_007": {"domain": "real", "task": "go_to_pose", "notes": "..."},
  "sim_000001":     {"domain": "synthetic", "task": "waypoints", "generator": "isaac-sim 4.5"}
}
```

`domain` lets people filter (`ds.split("train", domain="synthetic")`). Synthetic trajectories
should use the **same layout** (`<modality>/<trajectory>/*.parquet`, same columns, a
calibration YAML, mocap-equivalent GT poses in a `mocap` table); then every tool in the
package works on them unchanged.

---

## Data-quality notes found while building the package

Run `pooh quality` for the current numbers. Found so far:

- **trajectory_008 has no `floatingplatform` mocap body** (only SensorStack and cubesat),
  so its robot trajectory and goal error cannot be evaluated.
- **IMU at ~20 Hz with gaps up to 0.6 s** in all three trajectories. The Mid-360 IMU normally
  publishes at 200 Hz, so most samples were dropped at recording time.
- **Mocap dropouts**: all three bodies show ~75 gaps longer than 50 ms (max 130 ms) at the
  same instants, which points to recorder / network drops rather than occlusion. With the
  default 50 ms interpolation limit about 11% of frames get no label; tune with
  `max_gap_ns=`, or fix at the recorder (e.g. record mocap on a dedicated QoS / higher queue).
- **RealSense rates**: colour is at 7.3 Hz (135 ms period) and depth at 14.5 Hz, so only 60% of
  colour frames have a depth frame within 20 ms. That suggests frames were dropped at 15 fps
  (or a 7.5 fps colour profile).
- The first Firefly frame is almost black (max pixel value 48). Exposure may still have
  been settling; only that one frame was checked.
- **Mocap timestamps arrive in bursts** (median 0.4 ms between samples at ~215 Hz on
  average), so they are receipt times rather than capture times. Labels are not very
  sensitive to this (a constant camera time offset of -150 to +100 ms changes little), but
  it is one of the remaining error sources in the camera calibration.
- **Cube orientation noise** is up to ~2° (p95) between consecutive samples: its LEDs are
  close together. That is worth a few pixels of label error in close-up frames.

## Checking labels

After calibrating a camera, look at the labels before uploading:

```bash
pooh viz -t trajectory_009 -m realsense_color -m mocap        # add --web over SSH
```

The camera view shows the mask, box and cube wireframe on every frame. Scrub the timeline:
the overlay should stay on the cube. `pooh calibrate` also writes
`pooh_calibration/overlays_<camera>/*.png` for the clicked frames.
