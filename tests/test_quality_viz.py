import json

import numpy as np
import pytest
from conftest import H, W

from pooh_dataset.cli import main
from pooh_dataset.quality import StreamStats, format_reports, trajectory_report


def test_stream_stats_gaps():
    ts = np.array([0, 10, 20, 30, 100, 110]) * 1_000_000
    st = StreamStats.from_timestamps("x", ts)
    assert st.median_dt_ms == 10 and st.max_gap_ms == 70 and st.n_gaps == 1


def test_report_goal_and_flags(ds):
    r = trajectory_report(ds["trajectory_a"], robot_body="SensorStack")
    assert abs(r.final_pos_error_m - 0.03) < 1e-9 and r.final_yaw_error_deg < 1e-6
    assert r.min_goal_dist_m < 0.011 and r.goals[0][0] > 0
    assert r.label_coverage["firefly"] == 1.0
    r2 = trajectory_report(ds["trajectory_a"])  # default robot body is not in the fake mocap
    assert any("not tracked" in w for w in r2.warnings)
    assert "trajectory_a" in format_reports([r, r2])


def test_quality_cli_json(fake_root, tmp_path, capsys):
    out = tmp_path / "q.json"
    assert main(["--root", str(fake_root), "quality", "--body", "SensorStack", "--json", str(out)]) == 0
    data = json.loads(out.read_text())
    assert {d["trajectory"] for d in data} == {"trajectory_a", "trajectory_b"}


def test_viz_download_into_missing_root(tmp_path, monkeypatch, fake_root):
    """--download must work when the local dataset folder does not exist yet."""
    pytest.importorskip("rerun")
    import shutil

    import pooh_dataset.hub as hub

    calls = []

    def fake_download(repo_id, root, trajectories, modalities, revision):
        calls.append((trajectories, modalities))
        shutil.copytree(fake_root, root)
        return root

    monkeypatch.setattr(hub, "download", fake_download)
    root = tmp_path / "fresh"
    assert main(["--root", str(root), "viz", "-t", "trajectory_a", "-m", "mocap", "--download",
                 "--body", "SensorStack", "--save", str(tmp_path / "rrd")]) == 0
    assert calls == [(["trajectory_a"], ["mocap"])]
    assert (tmp_path / "rrd" / "trajectory_a.rrd").exists()


def test_viz_save(ds, tmp_path):
    pytest.importorskip("rerun")
    from pooh_dataset.viz import render_events, visualize

    ev = ds["trajectory_a"].events(*ds["trajectory_a"].time_range("events"))
    img = render_events(ev, H, W, 0.5)
    assert img.shape == (H // 2, W // 2, 3)
    visualize([ds["trajectory_a"]], mode="save", save_dir=str(tmp_path), robot_body="SensorStack",
              progress=False, events_fps=10)
    rrd = tmp_path / "trajectory_a.rrd"
    assert rrd.exists() and rrd.stat().st_size > 10_000
