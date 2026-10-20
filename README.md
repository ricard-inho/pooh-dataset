# pooh-dataset

Download, synchronize and turn the POOH multimodal robotics dataset
([`r3m3c3/off_test`](https://huggingface.co/datasets/r3m3c3/off_test) on the Hugging Face Hub)
into training data for **6D pose estimation, object detection, segmentation and optical
flow**.

The dataset contains trajectories of a PPO-controlled floating-platform robot recorded with:
a Prophesee EVK4 event camera, a FLIR Firefly RGB camera, an Intel RealSense D455 (colour +
depth), a Livox Mid-360 lidar + IMU, OptiTrack motion capture (`SensorStack`, `cubesat`,
`floatingplatform`), and the robot's control / observation topics. All streams in a trajectory
share one clock.

The same repo also holds **simulated** trajectories recorded in
[Space Robotics Bench](https://github.com/snt-spacer/space_robotics_bench): orbital inspection
(`sim_3d`: ISS, Gateway, CubeSat, ... ) and a twin of the Zero-G lab (`sim_zerog`), each with
RGB, depth, semantic labels, emulated events, lidar, IMU, ground-truth poses and the policy's
steps. See [Simulated data](#simulated-data).

## Install

```bash
pip install pooh-dataset            # core: download, read, sync, labels, exports
pip install "pooh-dataset[torch]"   # + PyTorch Dataset wrappers
pip install "pooh-dataset[mesh]"    # + loading CAD meshes (.ply/.obj/.stl)
pip install "pooh-dataset[viz]"     # + the Rerun playback dashboard (pooh viz)
pip install "pooh-dataset[calib]"   # + calibration tools (pooh align-markers, pooh calibrate)
pip install "pooh-dataset[all]"     # everything above
pip install "pooh-dataset[torch,viz]"   # or any combination

# with uv
uv add "pooh-dataset[all]"          # in a uv project
uv pip install "pooh-dataset[all]"  # into the active environment
uv tool install "pooh-dataset[viz]" # just the `pooh` CLI, isolated, on your PATH
```

Python ≥ 3.10. No OpenCV or ROS needed.

## Download

Data is large (≈ 11 GB per trajectory, mostly Firefly frames), so download only what you
need. Downloads resume, and a rerun only fetches files that are new on the Hub.

```bash
pooh download                                  # all real data
pooh download -m mocap -m realsense_color -m realsense_depth
pooh download -t trajectory_007 -m events
pooh download --domain sim_3d -m rgb -m mocap  # simulated: synthetic, sim_3d, sim_zerog; all = both
pooh download --domain sim_zerog --task docking --split test   # one simulated task, eval runs only
pooh download --benchmark v1 --domain sim_zerog --task docking --split test  # same, frozen paper benchmark
pooh download --revision <commit-sha>          # pin a dataset version for a paper
pooh info                                      # what is local / remote, splits
pooh splits                                    # train / val / test counts per domain and task
pooh check                                     # which calibration / CAD inputs exist
```

Without `-t`, `--domain` picks what is downloaded (default `real`), and `--split` / `--task`
narrow it down. With `-t`, the named trajectories are downloaded whatever their domain.

The default location is `~/.cache/pooh_dataset/<repo>`. Override it with `--root` or
`$POOH_DATA_DIR`, and point at another repo with `--repo` / `$POOH_REPO_ID`.

## Inspect trajectories: is it good or bad?

**Health report**: runs in seconds on mocap + messages only.

```bash
pooh download -m mocap -m robot_messages -m imu
pooh quality                     # table + warnings; --json report.json for the full numbers
```

```
trajectory          dur s final err cm yaw err°  path m mocap drops label cov  status
trajectory_007       40.9          6.0      0.4    1.52         229       89%  CHECK
trajectory_009       40.9          2.6      0.5    2.32         231        -   CHECK
  trajectory_007: realsense_color at 7.3 Hz (< 10)
  trajectory_007: mocap/floatingplatform: 72 dropouts > 50 ms (max 130 ms)
  ...
```

It reports the final position and heading error against the goal on
`/observation_formater_input`, path length, per-sensor rate and gaps, mocap dropouts, the
share of camera frames that can be labelled, and colour/depth sync. Thresholds are
arguments (`--goal-tol-cm`, `--body`).

**Playback dashboard**: synchronized, scrubbable playback in [Rerun](https://rerun.io):

```bash
pooh viz -t trajectory_007 --download        # fetches everything but Firefly, then opens the viewer
pooh viz -t trajectory_007 -t trajectory_009 # several trajectories: switch in the recordings panel
pooh viz -m mocap -m robot_messages          # all local trajectories, trajectory + actions only
pooh viz -t trajectory_008 --start 10 --end 20 -m events -m realsense_color
pooh viz --web                               # viewer in the browser (e.g. over SSH port-forward)
pooh viz --save out/                         # write out/<trajectory>.rrd; open with `rerun out/*.rrd`
```

Layout: 3D mocap world (bodies, full paths, goal and its heading) · x / y / yaw of every
body with the goal as a reference line · distance to goal · Firefly, RealSense colour +
depth and event frames (blue = ON, red = OFF) · Livox point cloud · thruster, reaction-wheel
and IMU plots · a text panel with the quality warnings. All panels share one timeline (seconds
from trajectory start).

Firefly frames are large (≈ 300 kB each), so they're logged every 2nd frame by default
(`--firefly-stride`). Events are rendered at `--events-fps 20` and `--events-scale 0.5`.
One trajectory without Firefly is ≈ 270 MB in the viewer and takes ≈ 20 s to log.

## Calibrate: object alignment + camera extrinsics

Masks and image-space poses need two transforms:

1. **Object → its OptiTrack body** (`T_body_model` in `models/objects.yaml`), from the
   Motive rigid-body export and the marker positions in the CAD.
2. **Camera → the `SensorStack` body** (`calibration/rig.yaml`), from cube corners you
   click in about 10 frames. The cube is static while the rig moves, so no calibration board
   or extra recording is needed.

The current dataset already has both for `cubesat` and `realsense_color`. The steps below
are for a new object, a re-built rigid body, or another camera.

```bash
pip install "pooh-dataset[calib]"
pooh download -m realsense_color -m mocap

# 1) object: Motive export + the marker centres in the CAD (here Onshape mm, Z-up part
#    exported Y-up, hence --cad-axes x,-z,y)
pooh align-markers --motive models/CubeSat_U1.motive --cad-axes x,-z,y \
  --cad-marker 46.077 -46.03447 -117.99807 --cad-marker -47.35423 -47.40068 -121.5 \
  --cad-marker -47.35423 47.39831 -121.5 --cad-marker 47.44476 47.39831 -121.5

# 2) camera: click the cube corners
pooh calibrate --camera realsense_color --box-keypoints -0.05 0 -0.05 0.05 0.1 0.05
```

`align-markers` matches your CAD markers to Motive's (subset and order are found
automatically) and refuses a fit worse than 3 mm. Motive stores bodies Y-up and the ROS
stream is Z-up; the standard conversion is the default (`--motive-to-body x,-z,y`).

`--box-keypoints` gives the cube body's 8 corners in CAD coordinates, excluding the marker
holder. When `T_body_model` comes from `align-markers`, `pooh calibrate` only solves the
camera (`--refine-model` overrides this). Solving both from clicks is unreliable: the
camera position and the object position trade off.

`pooh calibrate` prints a local URL (default http://localhost:8765) and opens it in your
browser. On a remote machine, run `ssh -L 8765:localhost:8765 <host>` on your laptop and open
the URL there.

**What to click:** the visible corners of the cube body, and nothing else. Skip the marker
holder, hidden corners and the stand; see [docs/calibration_example.png](docs/calibration_example.png).

On the page: **click** a corner · **right-click / u** undo · **wheel** zoom · **drag** pan ·
**f** fit · **n / Enter** next frame · **s** skip a frame where the cube is cut off or
over-exposed · **q** finish. Clicks are saved after every frame; `--add --frames N` clicks
N more later. After the first frames, yellow crosses show where the current solution
predicts the corners. A result with RMS > 10 px, or with implausible offsets, is flagged and
not written unless you pass `--force`.

Outputs, in `pooh_calibration/` (git-ignored), laid out like the Hub repo:

- `calibration/rig.yaml`: `T_body_cam` of the camera (`--also realsense_depth` gives the
  aligned depth camera the same transform)
- `models/objects.yaml`: `T_body_model` from `align-markers`, plus the keypoints
- `calibration/clicks_<camera>.json`: your clicks. Re-running re-solves from them.
- `overlays_<camera>/*.png`: rendered mask outline (green) and clicks (red), for checking

The results are also copied into your local dataset right away. Check them with
`pooh viz -m <camera> -m mocap`, which draws the masks on the images. Then share them:

```bash
hf upload r3m3c3/off_test pooh_calibration . --repo-type dataset --exclude 'overlays_*/*'
```

## Read and synchronize

```python
from pooh_dataset import PoohDataset
from pooh_dataset.labels import nearest_indices

ds = PoohDataset.download(modalities=["mocap", "realsense_color", "realsense_depth", "events"])
traj = ds["trajectory_007"]

rgb   = traj.frame("realsense_color", 100)        # (480, 640, 3) uint8
depth = traj.depth(200)                           # (480, 640) float32 metres
ts    = traj.timestamps("realsense_color")        # int64 ns, host clock

ev = traj.events(ts[100] - 25_000_000, ts[100] + 25_000_000)   # reads only overlapping row groups
T_world_cube, ok = traj.body_pose("cubesat", ts)               # interpolated (slerp) mocap poses
idx, ok = nearest_indices(traj.timestamps("realsense_depth"), ts, tolerance_ns=20_000_000)

scan = traj.lidar_scan(10)                        # dict of x, y, z, intensity, ... arrays
t, obs = traj.message_array("/pingu_low_level")   # (N, D) policy observations
t, act = traj.message_array("/minimal_thruster_command_to_px4")
```

Event representations: `pooh_dataset.events.event_histogram`, `voxel_grid`, `time_surface`.

## Training data

Every task splits by **trajectory** (`splits.json` on the Hub) and returns dicts of numpy
arrays. Use `pooh_dataset.torch` for PyTorch.

```python
from pooh_dataset.tasks import PoseEstimation, ObjectDetection, Segmentation, OpticalFlow

pose = PoseEstimation(ds, "train", camera="firefly", undistort=True, image_size=(1536, 1024))
s = pose[0]            # image, K, poses (N, 4, 4) T_cam_model, object_ids, valid

det = ObjectDetection(ds, "train", camera="realsense_color", stride=2)
s = det[0]             # image, boxes (xyxy of the visible part), labels, masks, poses

seg = Segmentation(ds, "val", camera="realsense_color")    # semantic + instance maps

flow = OpticalFlow(ds, "train", camera="realsense_color")  # image0, image1, flow, valid
eflow = OpticalFlow(ds, "train", camera="events", window_ns=50_000_000)  # voxel grid, flow
```

```python
import torch
from pooh_dataset.torch import TorchDataset, detection_collate

loader = torch.utils.data.DataLoader(TorchDataset(det), batch_size=8,
                                     collate_fn=detection_collate, num_workers=4)
```

How labels are made: object pose = mocap pose of the object (plus marker-to-CAD transform)
seen from the camera (SensorStack mocap pose plus camera extrinsics). Masks and boxes are
rendered from the CAD mesh at that pose, with occlusion between tracked objects. Optical
flow comes from RealSense depth moved by the camera and object motion measured by mocap.

> **Labels need calibration.** They work today for the `cubesat` in the RealSense colour
> camera (trajectories 007–009). Other cameras raise `MissingCalibrationError` until they are
> calibrated (see "Calibrate" above). Run `pooh check` for the current status and see
> [DATA_REQUIREMENTS.md](https://github.com/ricard-inho/pooh-dataset/blob/main/DATA_REQUIREMENTS.md).

## Simulated data

Simulated trajectories are named `trajectory_<tag>_<n>` and stored on the Hub as
`sim_3d_<modality>/<trajectory>/` and `sim_zerog_<modality>/<trajectory>/`. The library hides
the prefix: they are read like the real ones, with their own modality names.

| modality | content |
|---|---|
| `rgb`, `depth`, `semantic` | one pinhole camera (320 x 240, 20 Hz); depth in metres via `traj.depth(i)`; semantic class ids, names in `traj.semantic_classes` (`target` = 2) |
| `events` | emulated IMX636 events, pixel-aligned with the camera |
| `lidar` | scans in the sensor frame (`x`, `y`, `z`, `distance`, `channel`) |
| `mocap` | ground-truth poses: `robot`, `target`, `goal`, `camera` (+ `arm_end_effector` in the lab) |
| `imu` | robot IMU (linear acceleration in m/s^2, not g) |
| `joint_states` | lab: floating platform and arm joints (`traj.joint_states("arm_wall")`) |
| `steps` | policy steps at 10 Hz: action, observation, rewards, goal errors (`traj.steps()`) |

```python
from pooh_dataset import PoohDataset
from pooh_dataset.labels import camera_pose

ds = PoohDataset.download(domain="sim_3d", modalities=["rgb", "depth", "semantic", "mocap"])
ds.scenarios()                         # {"orbital/iss/earth": [...], ...}
ds.split("train", domain="sim_3d")     # train / test by recording run (splits.json)
ds.split("test", domain="sim_zerog", task="docking")
ds.tasks()                             # {"sim_3d/approach": [...], "sim_3d/orbit": [...],
                                       #  "sim_zerog/docking": [...], "sim_zerog/inspection": [...]}

traj = ds[ds.scenarios()["orbital/iss/earth"][0]]
traj.info                              # group, scenario, task, run, duration_s, goal errors, counts
traj.metadata                          # scenario parameters, rates, frames and units, term names
rgb, sem = traj.frame("rgb", 0), traj.frame("semantic", 0)
target_mask = sem == traj.semantic_classes["target"]
T_world_cam, ok = camera_pose(traj, "rgb", traj.timestamps("rgb")[0])
```

Calibration comes from the simulation (`calibration/<trajectory>.yaml`): intrinsics, and each
sensor's mount on the robot base, read as extrinsics on the `robot` mocap body (cameras in
their optical frame). So `pooh_dataset.labels.camera_pose`, `traj.body_pose("target", t)` and
`OpticalFlow(ds, ..., camera="rgb" | "events")` work as for the real data. The object tasks
(`PoseEstimation`, `ObjectDetection`, `Segmentation`) need a CAD model in
`models/objects.yaml` and are real-data only; in simulation, the `semantic` images give the
masks and the `target` mocap body the pose.

Domains in `ds.split(..., domain=...)`: `real`, `synthetic` (all simulated), `sim_3d`,
`sim_zerog`. Tasks in `ds.split(..., task=...)`: `approach` and `orbit` (sim_3d), `docking` and
`inspection` (sim_zerog).

### Splits

`splits.json` on the Hub lists the train and test (eval) trajectories; there is no simulated
`val`. It is updated by the sim publisher every time new runs are uploaded (or by hand with
`pooh splits update --push`), append-only: a trajectory never changes split. All episodes of
one recording run (same scenario, same seed) are in the same split, and about 20% of the runs
of each scenario config (target, background or arm, task) are in `test`, so every task has
eval runs from every day of recording. Real trajectories are not in the file yet: they keep
the hash-based split (`trajectory_007`, `trajectory_008` train, `trajectory_009` test). Pin
`--revision` to cite one version of the file.

### Benchmarks (for papers)

The split above grows every day. Paper numbers use a **frozen benchmark** instead: the data of
one Hub revision with a fixed train / test split. It ships with the package
(`pooh_dataset/benchmarks/v1.json`), and the Hub's `splits.json` carries a copy.

| benchmark | Hub revision | trajectories | train / test |
|---|---|---|---|
| `v1` (`benchmark_v1`, frozen 2026-10-20) | `8205a11` | 2195 sim + 3 real | sim_3d/approach 378 / 93, sim_3d/orbit 800 / 201, sim_zerog/docking 300 / 72, sim_zerog/inspection 282 / 69, real 2 / 1 |

```bash
pooh download --benchmark v1 --domain sim_zerog --task docking --split test -m rgb -m depth -m mocap
pooh splits --benchmark v1
```

```python
ds = PoohDataset.download(benchmark="v1", domain="sim_3d", split="train", modalities=["rgb", "mocap"])
ds.split("test", domain="sim_zerog", task="docking", benchmark="v1")
```

`--benchmark` pins the download to the benchmark's revision, so later uploads and repairs never
change its data. Its simulated trajectories keep their split in the growing `splits.json` too.
Cite the benchmark name and revision in the paper. `ds.hub_splits` gives the splits shown by the Hub viewer (simulation scenarios and
groups of real trajectories); they organise the data and are not train / test splits.

## Export to standard formats

```bash
pooh export bop  --camera firefly         --split train --out out/bop     # undistorted, mm
pooh export coco --camera realsense_color --split val   --out out/coco --stride 2
pooh export yolo --camera firefly --size 1536 1024 --split train --out out/yolo
pooh extract-frames -m firefly --stride 10 --out out/frames               # raw JPEGs + timestamps.csv
```

The same exporters are available from Python in `pooh_dataset.export`. Frames that are not
resized or undistorted are copied byte-for-byte, with no re-encoding.

## Development

```bash
uv sync
uv run pytest        # tests run on a synthetic dataset with analytically known labels
uv run ruff check
```

Release: bump `__version__` in `src/pooh_dataset/__init__.py`, tag `vX.Y.Z` and push the tag.
The `publish` GitHub workflow builds the package and uploads it to PyPI via trusted publishing.

Code: MIT. Data: CC-BY-4.0.
