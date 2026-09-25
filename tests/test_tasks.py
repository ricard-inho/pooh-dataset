import json

import numpy as np
from conftest import FRAME_DT, N_FRAMES, SPEED, H, W

from pooh_dataset.cli import main
from pooh_dataset.export import export_bop, export_coco, export_yolo, extract_frames
from pooh_dataset.tasks import ObjectDetection, OpticalFlow, PoseEstimation, Segmentation


def test_pose_task(ds):
    task = PoseEstimation(ds, "train", camera="firefly")
    assert len(task) == N_FRAMES
    s = task[1]
    assert s["image"].shape == (H, W, 3) and s["poses"].shape == (1, 4, 4) and s["valid"].all()
    assert np.allclose(s["poses"][0, :3, 3], [-0.1, 0, 2])


def test_detection_and_segmentation(ds):
    det = ObjectDetection(ds, "test", camera="firefly", stride=2)
    assert len(det) == 3
    s = det[0]
    assert s["boxes"].shape == (1, 4) and s["labels"].tolist() == [1]
    assert s["masks"].shape == (1, H, W) and s["masks"].sum() > 20
    seg = Segmentation(ds, "test", camera="firefly")[0]
    assert set(np.unique(seg["semantic"])) == {0, 1}


def test_resized_task_scales_labels(ds):
    s = ObjectDetection(ds, "train", camera="firefly", image_size=(160, 120))[0]
    assert s["image"].shape == (120, 160, 3) and s["masks"].shape == (1, 120, 160)
    cx = (s["boxes"][0, 0] + s["boxes"][0, 2]) / 2
    assert abs(cx - 80) <= 1.5


def test_flow_frames(ds):
    task = OpticalFlow(ds, "train", camera="firefly")
    assert len(task) == N_FRAMES - 1
    s = task[0]
    expected_du = -60 * SPEED * FRAME_DT / 1e9 / 2.0  # fx * dx / Z
    assert s["valid"].mean() > 0.9
    assert np.allclose(np.median(s["flow"][s["valid"]], axis=0), [expected_du, 0], atol=1e-6)


def test_flow_events(ds):
    task = OpticalFlow(ds, "train", camera="events", window_ns=FRAME_DT)
    s = task[0]
    assert s["events"].shape == (5, H, W) and s["flow"].shape == (H, W, 2)
    assert np.allclose(np.median(s["flow"][s["valid"]], axis=0), [-3, 0], atol=1e-6)


def test_exports(ds, tmp_path):
    det = ObjectDetection(ds, "train", camera="firefly")
    coco = json.loads((export_coco(det, tmp_path / "coco") / "annotations/instances_train.json").read_text())
    assert len(coco["images"]) == N_FRAMES and len(coco["annotations"]) == N_FRAMES
    rle = coco["annotations"][0]["segmentation"]
    assert sum(rle["counts"]) == H * W

    export_yolo(det, tmp_path / "yolo")
    lines = sorted((tmp_path / "yolo/labels/train").glob("*.txt"))[0].read_text().split()
    assert lines[0] == "0" and abs(float(lines[1]) - 0.5) < 0.02

    out = export_bop(ObjectDetection(ds, "train", camera="firefly", undistort=True), tmp_path / "bop")
    gt = json.loads((out / "train/000000/scene_gt.json").read_text())
    assert np.allclose(gt["0"][0]["cam_t_m2c"], [0, 0, 2000])
    assert (out / "models/models_info.json").exists()

    frames = extract_frames(ds["trajectory_a"], "firefly", tmp_path / "frames", stride=2)
    assert len(list(frames.glob("*.jpg"))) == 3


def test_cli(fake_root, capsys):
    assert main(["--root", str(fake_root), "info"]) == 0
    assert main(["--root", str(fake_root), "check"]) == 0
    out = capsys.readouterr().out
    assert "trajectory_a" in out and "cubesat" in out


def test_objects_limited_to_their_trajectories(ds):
    ds.objects["cubesat"].trajectories = ["trajectory_b"]
    assert PoseEstimation(ds, "train", camera="firefly").refs == []  # train = trajectory_a
    task = PoseEstimation(ds, "all", camera="firefly")
    assert {r.trajectory for r in task.refs} == {"trajectory_b"}
    # flow still works in trajectory_a, with only background motion
    assert len(OpticalFlow(ds, "train", camera="firefly")) == N_FRAMES - 1
