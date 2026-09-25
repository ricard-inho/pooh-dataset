import numpy as np

from pooh_dataset.calibrate import Observation, predict, solve
from pooh_dataset.calibration import CameraIntrinsics
from pooh_dataset.geometry import invert_pose, rotvec_to_rotmat, transform_points
from pooh_dataset.objects import ObjectModel

CUBE_CORNERS = np.array([[x, y, z] for x in (-0.05, 0.05) for y in (0.0, 0.1) for z in (-0.05, 0.05)])


def _pose(rv, t):
    T = np.eye(4)
    T[:3, :3] = rotvec_to_rotmat(rv)
    T[:3, 3] = t
    return T


def _scene(seed=0, n_frames=10, noise_px=1.0):
    rng = np.random.default_rng(seed)
    intr = CameraIntrinsics("cam", 640, 480, np.array([[384, 0, 320], [0, 384, 240], [0, 0, 1.0]]),
                            np.array([-0.05, 0.06, 0, 0, -0.02]))
    # camera looks along the rig's +x (z_cam = x_body), mounted 8 cm ahead, 5 cm up
    R_bc = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float) @ rotvec_to_rotmat([0.05, -0.1, 0.03])
    T_body_cam = np.eye(4)
    T_body_cam[:3, :3], T_body_cam[:3, 3] = R_bc, [0.08, 0.0, 0.05]
    # the cube's mocap body frame is tilted; the true model sits 30 deg off the initial guess
    T_objbody_model_true = _pose([0.3, 0.1, -0.2], [0.01, -0.02, -0.07]) @ _pose([0, np.radians(30), 0], [0.004, 0.0, -0.003])
    T_objbody_model_guess = T_objbody_model_true @ invert_pose(_pose([0, np.radians(30), 0], [0.004, 0.0, -0.003]))
    # the model stands upright (model +Y = world up), like the cube on its stand
    T_world_model = _pose([np.pi / 2, 0, 0], [2.6, -2.15, 0.7]) @ _pose([0, 0.4, 0], [0, 0, 0])
    T_world_obj = T_world_model @ invert_pose(T_objbody_model_true)
    obs = []
    for _ in range(n_frames):
        # rig 0.8-1.4 m from the cube, roughly facing it
        ang = rng.uniform(-0.6, 0.6)
        dist = rng.uniform(0.8, 1.4)
        pos = T_world_model[:3, 3] + np.array([-dist * np.cos(ang), -dist * np.sin(ang), rng.uniform(-0.1, 0.1)])
        T_world_body = _pose([0, 0, ang + rng.uniform(-0.15, 0.15)], pos - np.array([0.08, 0, 0.05]))
        T_cam_model = invert_pose(T_body_cam) @ invert_pose(T_world_body) @ T_world_model
        pts_cam = transform_points(T_cam_model, CUBE_CORNERS)
        uv, ok = predict(T_body_cam, T_objbody_model_true, Observation("t", 0, 0, np.zeros((0, 2)), T_world_body, T_world_obj, intr), CUBE_CORNERS)
        vis = np.argsort(pts_cam[:, 2])[:5]  # the 5 nearest corners ~ the visible ones
        clicks = uv[vis] + rng.normal(0, noise_px, (len(vis), 2))
        rng.shuffle(clicks)
        obs.append(Observation("traj", len(obs), 0, clicks, T_world_body, T_world_obj, intr))
    obj = ObjectModel.from_box("cube", [0.1, 0.1, 0.1], T_body_model=T_objbody_model_guess)
    obj.keypoints = CUBE_CORNERS
    return obs, obj, T_body_cam, T_objbody_model_true


def test_solve_recovers_extrinsics_and_yaw():
    obs, obj, T_bc_true, T_bm_true = _scene()
    res = solve(obs, obj, n_rotations=3000)
    assert res.rms_px < 2.0, res
    dR = res.T_body_cam[:3, :3].T @ T_bc_true[:3, :3]
    assert np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))) < 1.0
    assert np.linalg.norm(res.T_body_cam[:3, 3] - T_bc_true[:3, 3]) < 0.02
    # cube corners must land where the true ones are (up to the cube's 90 deg symmetry)
    true_c = transform_points(T_bm_true, CUBE_CORNERS)
    est_c = transform_points(res.T_body_model, CUBE_CORNERS)
    d = np.linalg.norm(true_c[:, None] - est_c[None], axis=-1).min(axis=1)
    assert d.max() < 0.01


def test_solve_camera_only_when_model_fixed():
    obs, obj, T_bc_true, T_bm_true = _scene(seed=1)
    obj.T_body_model = T_bm_true
    res = solve(obs, obj, refine_model=False, n_rotations=3000)
    assert res.rms_px < 2.0 and np.linalg.norm(res.T_body_cam[:3, 3] - T_bc_true[:3, 3]) < 0.02
    assert np.allclose(res.T_body_model, T_bm_true)


def test_calibrate_cli_from_clicks_file(fake_root, tmp_path):
    """Solve from a saved clicks file (no GUI): the fake rig's camera is the identity."""
    import json
    import shutil

    import yaml

    from pooh_dataset import PoohDataset
    from pooh_dataset.cli import main
    from pooh_dataset.labels import object_pose_in_camera

    root = tmp_path / "ds"
    shutil.copytree(fake_root, root)
    ds = PoohDataset(root)
    traj, cube = ds["trajectory_a"], ds.objects["cubesat"]
    intr = traj.calibration.intrinsics("firefly")
    frames = []
    for i, t in enumerate(traj.timestamps("firefly")):
        T, _ = object_pose_in_camera(traj, "firefly", cube, int(t))
        pts = transform_points(T, cube.corners)
        uv, _ = intr.project(pts)
        front = np.argsort(pts[:, 2])[:4]  # the face towards the camera
        frames.append({"trajectory": "trajectory_a", "index": i, "t_ns": int(t),
                       "points": (uv[front] + 0.2).tolist()})
    clicks = tmp_path / "clicks.json"
    clicks.write_text(json.dumps({"version": 1, "camera": "firefly", "object": "cubesat",
                                  "rig_body": "SensorStack", "frames": frames}))
    (root / "calibration" / "rig.yaml").unlink()  # start without extrinsics
    out = tmp_path / "out"
    assert main(["--root", str(root), "calibrate", "--camera", "firefly", "--clicks", str(clicks),
                 "--out", str(out), "--fixed-model"]) == 0
    rig = yaml.safe_load((out / "calibration/rig.yaml").read_text())
    assert rig["extrinsics"]["reference_body"] == "SensorStack"
    T = rig["extrinsics"]["cameras"]["firefly"]["T_body_cam"]
    assert np.allclose(T["translation"], 0, atol=0.02)
    assert abs(abs(T["rotation_xyzw"][3]) - 1) < 1e-3
    assert len(list((out / "overlays_firefly").glob("*.png"))) == len(frames)
    # applied to the local dataset: labels now work there
    assert (root / "calibration" / "rig.yaml").exists()
    ds2 = PoohDataset(root)
    assert "firefly" in ds2["trajectory_a"].calibration.extrinsics

    # an object whose T_body_model was measured from Motive markers is kept fixed by default
    objs = yaml.safe_load((root / "models" / "objects.yaml").read_text())
    objs["cubesat"]["motive_file"] = "models/fake.motive"
    (root / "models" / "objects.yaml").write_text(yaml.safe_dump(objs))
    out2 = tmp_path / "out2"
    assert main(["--root", str(root), "calibrate", "--camera", "firefly", "--clicks", str(clicks),
                 "--out", str(out2), "--no-apply"]) == 0
    assert (out2 / "calibration/rig.yaml").exists() and not (out2 / "models/objects.yaml").exists()


def test_web_click_session_over_http():
    import json
    import threading
    import urllib.request

    from pooh_dataset.calibrate_web import WebClickSession

    frames = iter([((f"t{i}", i, i), f"frame {i}", np.full((20, 30, 3), i * 40, np.uint8)) for i in range(4)])
    got = []
    sess = WebClickSession(frames, 2, hint=lambda key: np.array([[1.0, 2.0]]),
                           on_accept=lambda k, p: got.append((k, p)), port=18765, open_browser=False)
    th = threading.Thread(target=sess.run, daemon=True)
    th.start()
    for _ in range(100):  # wait for the server
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{sess.port}/state", timeout=1)
            break
        except OSError:
            import time

            time.sleep(0.05)
    base = f"http://127.0.0.1:{sess.port}"

    def post(path, body):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(), method="POST")
        return json.loads(urllib.request.urlopen(req, timeout=5).read())

    assert b"pooh calibrate" in urllib.request.urlopen(base + "/", timeout=5).read()
    st = json.loads(urllib.request.urlopen(base + "/state", timeout=5).read())
    assert st["title"] == "frame 0" and st["hint"] == []
    assert urllib.request.urlopen(base + f"/image/{st['frame_id']}", timeout=5).read()[:2] == b"\xff\xd8"
    st = post("/accept", {"frame_id": st["frame_id"], "points": [[1, 1], [2, 2]]})  # too few: refused
    assert st["done"] == 0 and not got
    st = post("/accept", {"frame_id": st["frame_id"], "points": [[1, 1], [2, 2], [3, 3]]})
    assert st["done"] == 1 and st["hint"] == [[1.0, 2.0]] and got[0][0] == ("t0", 0, 0)
    st = post("/skip", {"frame_id": st["frame_id"]})
    assert st["title"] == "frame 2"
    st = post("/accept", {"frame_id": st["frame_id"], "points": [[5, 5], [6, 6], [7, 7]]})
    assert st["finished"] and len(got) == 2
    th.join(timeout=5)
    assert not th.is_alive()


def test_select_frames_skips_clicked(ds):
    from pooh_dataset.calibrate import select_frames

    cube = ds.objects["cubesat"]
    first = select_frames(ds, "firefly", cube, "SensorStack", 2, step=1)
    done = {(t, i) for t, i, _ in first[:3]}
    again = select_frames(ds, "firefly", cube, "SensorStack", 2, step=1, exclude=done)
    assert again and not {(t, i) for t, i, _ in again} & done
    assert len(set(again)) == len(again)


def test_align_markers_recovers_transform():
    from pooh_dataset.calibrate import MOTIVE_TO_ROS_BODY, align_markers

    rng = np.random.default_rng(3)
    cad = np.array([[0.046, 0.118, -0.046], [-0.047, 0.1215, -0.047], [-0.047, 0.1215, 0.047], [0.047, 0.1215, 0.047]])
    T_true = _pose([0.2, -0.4, 0.1], [0.01, -0.07, 0.02])  # model -> ROS body
    # Motive stores the body Y-up; it also has 4 extra markers the CAD list doesn't include
    body = transform_points(T_true, cad)
    motive = body @ MOTIVE_TO_ROS_BODY  # inverse of the (orthonormal) axis change
    extra = rng.normal(0, 0.03, (4, 3))
    all_markers = np.vstack([extra[:2], motive, extra[2:]]) + rng.normal(0, 3e-4, (8, 3))
    T, r, idx = align_markers(all_markers, cad)
    assert r.max() < 0.002 and idx == [2, 3, 4, 5]
    assert np.allclose(T[:3, :3], T_true[:3, :3], atol=1e-2) and np.allclose(T[:3, 3], T_true[:3, 3], atol=2e-3)
