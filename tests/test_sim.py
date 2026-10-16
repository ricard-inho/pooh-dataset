"""Simulated trajectories in the Hub layout (``sim_<group>_<modality>/<trajectory>``)."""

import numpy as np
import pytest
from conftest import FRAME_DT, N_FRAMES, SIM_F, SIM_H, SIM_TARGET_X, SIM_TRAJ, SIM_W, SPEED

from pooh_dataset import hub
from pooh_dataset.labels import camera_pose
from pooh_dataset.tasks import OpticalFlow


def test_listing_and_domains(ds):
    traj = ds[SIM_TRAJ]
    assert traj.simulated and traj.domain == "synthetic" and traj.prefix == "sim_3d_"
    assert traj.modalities == ["depth", "events", "imu", "lidar", "mocap", "rgb", "semantic", "steps"]
    info = ds.trajectory_info(SIM_TRAJ)
    assert info["group"] == "sim_3d" and info["scenario"] == "orbital/cubesat/black"
    assert info["hub_split"] == "cubesat_black" and "source_dir" not in info
    assert ds.split("all", "sim_3d") == [SIM_TRAJ] and ds.split("all", "sim_zerog") == []
    assert SIM_TRAJ not in ds.split("all", "real")
    assert ds.hub_splits == {"cubesat_black": [SIM_TRAJ]}
    assert ds.scenarios() == {"orbital/cubesat/black": [SIM_TRAJ]}
    assert not ds["trajectory_a"].simulated


def test_reading(ds):
    traj = ds[SIM_TRAJ]
    assert traj.frame("rgb", 0).shape == (SIM_H, SIM_W, 3)
    assert traj.depth_modality == "depth" and np.allclose(traj.depth(0), SIM_TARGET_X)
    assert traj.semantic_classes["target"] == 2 and (traj.frame("semantic", 0) == 2).any()
    assert traj.mocap_bodies == ["robot", "target"]
    assert "trajectory_id" not in traj.imu() and "trajectory_id" not in traj.lidar_scan(0)
    ev = traj.events(0, 500_000_000)
    assert len(ev) == 500 and ev.x.max() < SIM_W
    st = traj.steps()
    assert st["action"].shape == (4, 2) and st["observation"].shape == (4, 3) and st["truncated"][-1]
    assert traj.metadata["source"] == "simulation"
    # the real trajectories' tables also carry trajectory_id: never in the readers' output
    assert "trajectory_id" not in ds["trajectory_a"].columns("mocap")


def test_calibration_and_camera_pose(ds):
    traj = ds[SIM_TRAJ]
    cal = traj.calibration
    assert cal.reference_body == "robot"  # rig.yaml (SensorStack) does not apply to simulation
    assert cal.cameras["depth"].depth_scale == 0.001
    assert np.allclose(cal.T_body_cam("lidar"), np.eye(4))  # lidar stays in its sensor frame
    t = 2 * FRAME_DT
    T_wc, ok = camera_pose(traj, "rgb", t)
    assert ok
    # optical frame: z along the robot's x (forward), x along -y (right), y along -z (down)
    assert np.allclose(T_wc[:3, :3], [[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    assert np.allclose(T_wc[:3, 3], [0, SPEED * t / 1e9, 0])
    # the static target projects onto its semantic mask
    target = traj.body_pose("target", t)[0][:3, 3]
    p = np.linalg.solve(T_wc, np.append(target, 1))[:3]
    uv, good = cal.intrinsics("rgb").project(p[None])
    u, v = np.rint(uv[0]).astype(int)
    assert good[0] and traj.frame("semantic", 2)[v, u] == 2


def test_optical_flow_sim(ds):
    task = OpticalFlow(ds, [SIM_TRAJ], camera="rgb")
    assert len(task) == N_FRAMES - 1
    s = task[0]
    assert s["valid"].all()
    # the camera moves 0.1 m to its left in front of a 5 m wall: the image moves right
    expected = SIM_F * SPEED * FRAME_DT / 1e9 / SIM_TARGET_X
    assert np.allclose(s["flow"][..., 0], expected, atol=1e-3) and np.allclose(s["flow"][..., 1], 0, atol=1e-3)
    ev = OpticalFlow(ds, [SIM_TRAJ], camera="events", window_ns=100_000_000)
    assert len(ev) > 0 and ev[0]["events"].shape == (5, SIM_H, SIM_W)



def test_quality_sim(ds):
    from pooh_dataset.quality import trajectory_report

    rep = trajectory_report(ds[SIM_TRAJ])
    assert rep.final_pos_error_m == pytest.approx(0.02) and rep.ok, rep.warnings
    assert "rgb" in rep.streams and rep.label_coverage["rgb"] == 1.0


def test_download_patterns(monkeypatch, tmp_path):
    calls = []
    import huggingface_hub

    def fake_snapshot(repo_id, **kw):
        calls.append(kw["allow_patterns"])

    def fake_file(repo_id, filename, **kw):
        p = tmp_path / filename
        p.write_text('{"trajectories": {"trajectory_007": {}}}')
        return str(p)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_file)

    hub.download(root=tmp_path)
    pats = calls.pop()
    assert "realsense_color/*/*.parquet" in pats and "calibration/trajectory_007.yaml" in pats
    assert not any(p.startswith(("sim_3d", "sim_zerog", "sim_*", "*/", "calibration/*", "metadata/*")) for p in pats)

    hub.download(root=tmp_path, domain="sim_zerog", modalities=["rgb", "mocap"])
    pats = calls.pop()
    assert {"sim_zerog_rgb/*/*.parquet", "sim_zerog_mocap/*/*.parquet", "calibration/*", "metadata/*"} <= set(pats)
    assert not any(p.startswith(("sim_3d", "rgb/", "mocap/")) for p in pats)

    hub.download(root=tmp_path, trajectories=["trajectory_hera_000001"], modalities=["rgb"])
    pats = calls.pop()
    assert {"rgb/trajectory_hera_000001/*.parquet", "sim_*_rgb/trajectory_hera_000001/*.parquet",
            "calibration/trajectory_hera_000001.yaml", "metadata/trajectory_hera_000001.yaml"} <= set(pats)
    assert "calibration/*" not in pats

    with pytest.raises(ValueError, match="unknown domain"):
        hub.download(root=tmp_path, domain="sim")
