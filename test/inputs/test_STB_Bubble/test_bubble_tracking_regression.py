from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


BUBBLE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BUBBLE_ROOT.parents[2]
MANIFEST_MODULE_PATH = BUBBLE_ROOT / "build_bubble_manifest.py"
METRICS_MODULE_PATH = BUBBLE_ROOT / "evaluate_tracking_metrics.py"
PR_RUNNER_MODULE_PATH = BUBBLE_ROOT / "run_pr_tracking_regression.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


def test_bubble_manifest_maps_one_based_frames_and_radius_by_track_id():
    mod = _load_module(MANIFEST_MODULE_PATH, "build_bubble_manifest")

    tracks = np.array(
        [
            [1.0, 2.0, 3.0, 1.0, 10.0],
            [4.0, 5.0, 6.0, 2.0, 20.0],
        ]
    )
    r_mm = np.array([[20.0, 0.45], [10.0, 0.25]])

    manifest = mod.build_manifest_dataframe(tracks, r_mm)

    assert manifest["runtime_frame_0based"].tolist() == [0, 1]
    assert manifest["source_frame_1based"].tolist() == [1, 2]
    assert manifest["track_id"].tolist() == [10, 20]
    assert manifest["radius_mm"].tolist() == [0.25, 0.45]


def test_bubble_metrics_default_result_dir_points_to_bubble_results():
    mod = _load_module(METRICS_MODULE_PATH, "evaluate_bubble_tracking_metrics")

    args = mod.parse_args([])

    assert args.result_dir == REPO_ROOT / "test" / "results" / "test_STB_Bubble" / "ConvergeTrack"


def test_bubble_metrics_defaults_evaluate_last_50_frames_of_100_frame_fixture():
    mod = _load_module(METRICS_MODULE_PATH, "evaluate_bubble_tracking_metrics_default_frames")

    args = mod.parse_args([])

    assert args.frame_end == 99
    assert args.eval_start == 50
    assert args.eval_end == 99


def test_bubble_runner_defaults_match_100_frame_fixture():
    mod = _load_module(PR_RUNNER_MODULE_PATH, "run_bubble_pr_tracking_regression_default_frames")

    args = mod.parse_args([])

    assert args.frame_end == 99
    assert args.eval_start == 50
    assert args.eval_end == 99


def test_bubble_metrics_compute_radius_error_on_correct_matches():
    mod = _load_module(METRICS_MODULE_PATH, "evaluate_bubble_tracking_metrics_radius")

    gt = pd.DataFrame(
        {
            "runtime_frame_0based": [0, 0],
            "track_id": [1, 2],
            "x": [0.0, 10.0],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
            "radius_mm": [0.20, 0.50],
        }
    )
    detections = pd.DataFrame(
        {
            "TrackID": [100, 101],
            "FrameID": [0, 0],
            "WorldX": [0.01, 10.01],
            "WorldY": [0.0, 0.0],
            "WorldZ": [0.0, 0.0],
            "R3D": [0.23, 0.44],
            "source_file": ["LongTrackActive_99.csv", "LongTrackActive_99.csv"],
        }
    )

    result = mod.evaluate_subset(gt, detections, "all_bubble_manifest_points", threshold_mm=0.1)

    assert result["coverage_C_track"] == 1.0
    assert result["radius_error_mean_mm"] == np.mean([0.03, 0.06])
    assert result["radius_error_median_mm"] == np.median([0.03, 0.06])
    assert result["radius_error_p95_mm"] == np.percentile([0.03, 0.06], 95)


def test_bubble_pr_runner_rejects_radius_error_threshold_failures():
    mod = _load_module(PR_RUNNER_MODULE_PATH, "run_bubble_pr_tracking_regression")

    metrics = {
        "metrics": [
            {
                "label": "all_bubble_manifest_points",
                "coverage_C_track": 1.0,
                "position_error_mean_mm": 0.01,
                "fragmentation_F_mean_detected_tracks_per_covered_gt": 1.0,
                "correct_connection_Cr_mean_per_detected_track": 0.95,
                "radius_error_mean_mm": 0.2,
                "radius_error_p95_mm": 0.4,
            }
        ]
    }

    failures = mod.check_thresholds(
        metrics,
        {
            "all_bubble_manifest_points": {
                "radius_error_mean_mm": {"max": 0.05},
                "radius_error_p95_mm": {"max": 0.15},
            }
        },
    )

    assert len(failures) == 2
    assert any("radius_error_mean_mm" in failure for failure in failures)
    assert any("radius_error_p95_mm" in failure for failure in failures)
