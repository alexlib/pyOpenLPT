# pyright: reportAttributeAccessIssue=false
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


TEST_STB_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TEST_STB_ROOT.parents[2]
MODULE_PATH = TEST_STB_ROOT / "regenerate_test_stb.py"
METRICS_MODULE_PATH = TEST_STB_ROOT / "evaluate_tracking_metrics.py"
PR_RUNNER_MODULE_PATH = TEST_STB_ROOT / "run_pr_tracking_regression.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("regenerate_test_stb", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("regenerate_test_stb", module)
    spec.loader.exec_module(module)
    return module


def _load_metrics_module():
    spec = importlib.util.spec_from_file_location("evaluate_tracking_metrics", METRICS_MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("evaluate_tracking_metrics", module)
    spec.loader.exec_module(module)
    return module


def _load_pr_runner_module():
    spec = importlib.util.spec_from_file_location("run_pr_tracking_regression", PR_RUNNER_MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("run_pr_tracking_regression", module)
    spec.loader.exec_module(module)
    return module


def test_resolve_frame_range_defaults_and_subset():
    mod = _load_module()

    assert mod.resolve_frame_range(None, 250) == (0, 249)
    assert mod.resolve_frame_range((0, 4), 250) == (0, 4)
    assert mod.resolve_frame_range((249, 249), 250) == (249, 249)


def test_resolve_frame_range_rejects_invalid_bounds():
    mod = _load_module()

    try:
        mod.resolve_frame_range((4, 0), 250)
    except ValueError as exc:
        assert "START <= END" in str(exc)
    else:
        raise AssertionError("expected ValueError for reversed frame range")

    try:
        mod.resolve_frame_range((0, 250), 250)
    except ValueError as exc:
        assert "within 0..249" in str(exc)
    else:
        raise AssertionError("expected ValueError for out-of-range frame")


def test_build_runtime_config_omits_optional_load_paths():
    mod = _load_module()

    config_text = mod.build_runtime_config_text(frame_start=0, frame_end=4, n_threads=0)

    assert "0,4" in config_text
    assert "../camFile/cam1.txt,255" in config_text
    assert "imgFile/cam1ImageNames.txt" in config_text
    assert "../../../results/test_STB_generated/" in config_text
    assert "tracerConfig.txt" in config_text
    assert "0,-1" in config_text
    assert "Path to active long track files" not in config_text
    assert "Path to active short track files" not in config_text


def test_build_runtime_config_supports_in_place_paths():
    mod = _load_module()

    config_text = mod.build_runtime_config_text(
        frame_start=0,
        frame_end=49,
        n_threads=0,
        camera_prefix="camFile",
        output_path="../../results/test_STB/",
    )

    assert "camFile/cam1.txt,255" in config_text
    assert "../camFile/cam1.txt,255" not in config_text
    assert "imgFile/cam1ImageNames.txt" in config_text
    assert "../../results/test_STB/" in config_text
    assert "0,-1" in config_text


def test_compute_projection_flags_distinguishes_visible_and_rendered():
    mod = _load_module()

    projection = np.array([
        [5.0, 6.0],
        [10.0, -0.1],
        [np.nan, 3.0],
        [-2.2, 8.0],
    ])

    visible, rendered = mod.compute_projection_flags(projection, n_row=8, n_col=10, tr_radius=2)

    assert visible.tolist() == [True, False, False, False]
    assert rendered.tolist() == [True, True, False, False]


def test_parse_camera_file_reads_legacy_layout():
    mod = _load_module()

    camera = mod.parse_camera_file(TEST_STB_ROOT / "camFile" / "cam1.txt")

    assert camera.name == "cam1"
    assert camera.n_row == 1024
    assert camera.n_col == 1024
    assert camera.cam_matrix.shape == (3, 3)
    assert camera.dist_coeff.shape == (1, 5)
    assert camera.rot_vec.shape == (3, 1)
    assert camera.trans_vec.shape == (3, 1)


def test_metrics_all_camera_mask_combines_camera_flags():
    mod = _load_metrics_module()

    import pandas as pd

    manifest = pd.DataFrame(
        {
            "cam1_visible": [1, 1, 0],
            "cam2_visible": [1, 0, 1],
            "cam3_visible": [1, 1, 1],
            "cam4_visible": [1, 1, 1],
        }
    )

    assert mod.all_camera_mask(manifest, "visible").tolist() == [True, False, False]


def test_metrics_default_result_dir_points_to_test_results():
    mod = _load_metrics_module()

    args = mod.parse_args([])

    assert args.result_dir == REPO_ROOT / "test" / "results" / "test_STB" / "ConvergeTrack"


def test_pr_runner_accepts_metrics_that_meet_thresholds():
    mod = _load_pr_runner_module()

    metrics = {
        "metrics": [
            {
                "label": "all_4_camera_rendered_roi",
                "coverage_C_track": 0.9975,
                "position_error_mean_mm": 0.0010,
                "fragmentation_F_mean_detected_tracks_per_covered_gt": 1.04,
                "correct_connection_Cr_mean_per_detected_track": 0.8895,
            }
        ]
    }

    failures = mod.check_thresholds(metrics, mod.DEFAULT_THRESHOLDS)

    assert failures == []


def test_pr_runner_rejects_metrics_below_thresholds():
    mod = _load_pr_runner_module()
    metrics = {
        "metrics": [
            {
                "label": "all_4_camera_rendered_roi",
                "coverage_C_track": 0.9,
                "position_error_mean_mm": 0.02,
                "fragmentation_F_mean_detected_tracks_per_covered_gt": 1.5,
                "correct_connection_Cr_mean_per_detected_track": 0.7,
            }
        ]
    }

    failures = mod.check_thresholds(metrics, mod.DEFAULT_THRESHOLDS)

    assert len(failures) == 4
    assert any("coverage_C_track" in failure for failure in failures)
