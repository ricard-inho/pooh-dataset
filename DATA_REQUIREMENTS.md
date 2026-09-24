# What the dataset still needs for labels

`pooh-dataset` can already download, read and synchronize every modality. Ground-truth
**labels** (6D pose, masks, boxes, optical flow) are computed from motion capture, and that
only works once the dataset repo contains the calibration and model files below. Until then
the label tasks raise `MissingCalibrationError`, pointing here.

Run `pooh check` at any time to see what is present:

```
$ pooh check
splits.json         --
models/objects.yaml --  []
trajectory_007
  reference_body     --  None
  firefly            intrinsics ok  extrinsics --  source=supplement:...
  realsense_color    intrinsics ok  extrinsics --
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
| 1 | Camera extrinsics: `SensorStack` marker body → each camera optical frame | `calibration/rig.yaml` | everything with labels | missing |
| 2 | Event camera intrinsics (1280x720) | `calibration/rig.yaml` or `<traj>.yaml` | event labels / flow | missing |
| 3 | Firefly intrinsics from a real calibration (current ones are `supplement:`) | `calibration/rig.yaml` | Firefly labels | verify |
| 4 | Cubesat CAD mesh | `models/cubesat.ply` (or .obj/.stl) | masks, boxes, BOP | missing |
| 5 | Cubesat marker → CAD transform `T_body_model` | `models/objects.yaml` | all cubesat labels | missing |
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

How to get it (any one of):

- **Hand-eye calibration**: record a checkerboard / AprilTag board that also carries mocap
  markers (or is static in the mocap world), then solve `AX = XB` (e.g. OpenCV
  `calibrateHandEye`) between the SensorStack mocap poses and the board poses per camera.
- **Camera-to-camera** (Kalibr, OpenCV stereo) for all cameras + a single hand-eye for one of
  them; the package composes them.
- CAD of the sensor mount + marker positions: acceptable to start, but check it by projecting
  the cubesat (see "Sanity check" below).

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

```yaml
cubesat:
  class_id: 1                   # 1-based; BOP obj_id / COCO category_id (YOLO uses class_id-1)
  mocap_body: cubesat           # body name in the mocap table
  mesh: models/cubesat.ply      # path relative to the dataset root (.ply/.obj/.stl/.glb)
  mesh_scale: 0.001             # CAD units -> metres (0.001 if the CAD is in mm)
  # Pose of the CAD model frame in the mocap rigid-body frame of the marker on the face:
  T_body_model: {translation: [0.0, 0.0, -0.05], rotation_xyzw: [0, 0, 0, 1]}
  # optional, passed to BOP models_info.json:
  # symmetries: {symmetries_discrete: [[...16 numbers...]]}

floatingplatform:
  class_id: 2
  mocap_body: floatingplatform
  mesh: models/floatingplatform.ply
  mesh_scale: 0.001
  T_body_model: ...
```

Until the CAD is uploaded, you can use the cube's dimensions instead of `mesh` and get
correct boxes and masks for a cuboid:

```yaml
cubesat:
  mocap_body: cubesat
  box: [0.10, 0.10, 0.10]       # size in metres, centred on the model origin
  T_body_model: ...
```

**`T_body_model` for the marker on a face.** The OptiTrack rigid body's origin and axes are
whatever was chosen when the body was created in Motive: by default the marker centroid, with
axes aligned to the mocap world at creation time. They are *not* the CAD origin. Options:

1. In Motive, set the rigid body pivot to a known point (e.g. the cube centre) and align its
   orientation to the cube faces, then `T_body_model` = offset of the CAD origin from there.
2. Measure the marker positions on the CAD, export the rigid body definition from Motive
   (marker positions in the body frame), and solve the rigid transform between the two point
   sets (Kabsch). This is the most accurate. A helper for it can be added to the package
   once the files exist.

Meshes should be simplified to about 10k faces or fewer. Masks are rasterised per triangle,
so huge CAD meshes slow down label generation.

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

## Sanity check once 1 + 4 + 5 are in

```python
from pooh_dataset import PoohDataset
from pooh_dataset.tasks import ObjectDetection
import numpy as np, PIL.Image as I

ds = PoohDataset()
det = ObjectDetection(ds, "all", camera="realsense_color", stride=50)
s = det[0]
over = s["image"].copy(); over[s["masks"].any(0)] = [255, 0, 0]
I.fromarray(over).save("overlay.png")   # the red mask should sit exactly on the cube
```
