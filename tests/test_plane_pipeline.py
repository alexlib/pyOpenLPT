# pyright: reportMissingImports=false, reportAttributeAccessIssue=false
import json
import sys
from types import SimpleNamespace
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import cv2
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import scripts.case_026_plane_debug_loop as case_026_plane_debug_loop
from modules.camera_calibration.wand_calibration.refraction_calibration_BA import (
    RefractiveBAConfig,
    RefractiveBAOptimizer,
)
from modules.camera_calibration.wand_calibration.refraction_wand_calibrator import PlaneInitializer
from scripts.case_026_plane_debug_loop import TRACE_STAGE_ORDER, compare_planes, run_trace


CASE_026_DIR = Path("J:/Refraction_test/case_026")
J_DRIVE_SKIP_REASON = "J: drive unavailable: J:/Refraction_test/case_026 not found"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _make_gt_window(*, plane_pt_far, plane_n, thick_mm):
    plane_pt_far = np.asarray(plane_pt_far, dtype=np.float64)
    plane_n = np.asarray(plane_n, dtype=np.float64)
    plane_n = plane_n / np.linalg.norm(plane_n)
    plane_pt_closest = plane_pt_far - float(thick_mm) * plane_n
    return {
        "plane_pt": plane_pt_far.tolist(),
        "plane_pt_far": plane_pt_far.tolist(),
        "plane_pt_closest": plane_pt_closest.tolist(),
        "plane_n": plane_n.tolist(),
        "thick_mm": float(thick_mm),
    }


@pytest.fixture(scope="module")
def live_trace_results(tmp_path_factory):
    if not CASE_026_DIR.is_dir():
        pytest.skip(J_DRIVE_SKIP_REASON)

    results_dir = tmp_path_factory.mktemp("case_026_trace")
    manifest = run_trace(
        case_dir=CASE_026_DIR,
        results_dir=results_dir,
        frame_budget=25,
        frame_selection="sequential",
    )
    return {"results_dir": Path(results_dir), "manifest": manifest}


def test_comparison_metric_reports_per_window_errors():
    gt_artifact = {
        "interface_convention": "farthest",
        "windows": {
            "0": _make_gt_window(plane_pt_far=[10.0, 0.0, 0.0], plane_n=[1.0, 0.0, 0.0], thick_mm=2.0),
            "1": _make_gt_window(plane_pt_far=[0.0, 0.0, 0.0], plane_n=[1.0, 0.0, 0.0], thick_mm=3.0),
        },
    }
    stage_artifact = {
        "stage": "UNIT",
        "interface_convention": "farthest",
        "windows": {
            "0": {"plane_pt": [11.0, 0.0, 0.0], "plane_n": [1.0, 0.0, 0.0]},
            "1": {"plane_pt": [0.0, 0.0, 0.0], "plane_n": [0.0, 1.0, 0.0]},
            "2": {"plane_pt": [5.0, 0.0, 0.0], "plane_n": [1.0, 0.0, 0.0]},
        },
    }

    result = compare_planes(stage_artifact, gt_artifact)

    assert result["stage"] == "UNIT"
    assert result["stage_interface_convention"] == "farthest"

    window0 = result["windows"]["0"]
    assert window0["stage_present"] is True
    assert window0["gt_present"] is True
    assert window0["gt_plane_point_used"] == "plane_pt_far"
    assert window0["angular_error_deg"] == pytest.approx(0.0)
    assert window0["point_distance_mm"] == pytest.approx(1.0)

    window1 = result["windows"]["1"]
    assert window1["angular_error_deg"] == pytest.approx(90.0)
    assert window1["point_distance_mm"] == pytest.approx(0.0)

    window2 = result["windows"]["2"]
    assert window2["stage_present"] is True
    assert window2["gt_present"] is False
    assert window2["angular_error_deg"] is None
    assert window2["point_distance_mm"] is None


def test_closest_farthest_shift_is_reversible():
    gt_artifact = {
        "interface_convention": "farthest",
        "windows": {
            "0": _make_gt_window(plane_pt_far=[10.0, 4.0, -2.0], plane_n=[0.0, 0.0, 1.0], thick_mm=6.5),
        },
    }

    gt_window = gt_artifact["windows"]["0"]
    plane_n = np.asarray(gt_window["plane_n"], dtype=np.float64)
    plane_pt_far = np.asarray(gt_window["plane_pt_far"], dtype=np.float64)
    plane_pt_closest = np.asarray(gt_window["plane_pt_closest"], dtype=np.float64)
    thick_mm = gt_window["thick_mm"]

    np.testing.assert_allclose(plane_pt_closest + thick_mm * plane_n, plane_pt_far, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(plane_pt_far - thick_mm * plane_n, plane_pt_closest, atol=1e-12, rtol=0.0)

    closest_result = compare_planes(
        {
            "stage": "INIT",
            "interface_convention": "closest",
            "windows": {"0": {"plane_pt": plane_pt_closest.tolist(), "plane_n": plane_n.tolist()}},
        },
        gt_artifact,
    )
    farthest_result = compare_planes(
        {
            "stage": "EXPORT",
            "interface_convention": "farthest",
            "windows": {"0": {"plane_pt": plane_pt_far.tolist(), "plane_n": plane_n.tolist()}},
        },
        gt_artifact,
    )

    assert closest_result["windows"]["0"]["gt_plane_point_used"] == "plane_pt_closest"
    assert farthest_result["windows"]["0"]["gt_plane_point_used"] == "plane_pt_far"
    assert closest_result["windows"]["0"]["angular_error_deg"] == pytest.approx(0.0)
    assert farthest_result["windows"]["0"]["angular_error_deg"] == pytest.approx(0.0)
    assert closest_result["windows"]["0"]["point_distance_mm"] == pytest.approx(0.0)
    assert farthest_result["windows"]["0"]["point_distance_mm"] == pytest.approx(0.0)


def test_trace_artifact_live_case_schema(live_trace_results):
    manifest = live_trace_results["manifest"]
    results_dir = live_trace_results["results_dir"]

    assert manifest["mode"] == "trace"
    assert set(manifest["artifacts"].keys()) == set(TRACE_STAGE_ORDER)
    assert manifest["ba_frame_budget"] == 25
    assert manifest["ba_frame_selection"] == "sequential"

    gt_planes = _read_json(results_dir / "gt_planes.json")
    bootstrap_stage = _read_json(Path(manifest["artifacts"]["BOOTSTRAP"]))
    export_stage = _read_json(Path(manifest["artifacts"]["EXPORT"]))

    assert gt_planes["artifact"] == "gt_planes"
    assert gt_planes["interface_convention"] == "farthest"
    assert gt_planes["windows"]

    assert bootstrap_stage["stage"] == "BOOTSTRAP"
    assert bootstrap_stage["plane_state"] == "not_initialized"
    assert bootstrap_stage["interface_convention"] == "closest"
    assert bootstrap_stage["metadata"]["frame_budget_scope"] == "ba_only"
    assert bootstrap_stage["metadata"]["trace_scaffold"] is True

    for window in bootstrap_stage["windows"].values():
        assert set(window) >= {"plane_pt", "plane_n", "thick_mm", "plane_state"}
        assert window["plane_pt"] is None
        assert window["plane_n"] is None
        assert window["plane_state"] == "not_initialized"

    assert export_stage["stage"] == "EXPORT"
    assert export_stage["interface_convention"] == "farthest"
    assert export_stage["windows"]
    for window in export_stage["windows"].values():
        assert set(window) >= {"plane_pt", "plane_n", "thick_mm", "cam_file"}
        assert window["plane_pt"] is not None
        assert window["plane_n"] is not None
        assert window["thick_mm"] is not None


def test_interface_convention_live_case_matches_stage_contract(live_trace_results):
    manifest = live_trace_results["manifest"]
    init_stage = _read_json(Path(manifest["artifacts"]["INIT"]))
    export_stage = _read_json(Path(manifest["artifacts"]["EXPORT"]))
    render_stage = _read_json(Path(manifest["artifacts"]["RENDER"]))
    final_refine_post_align = _read_json(Path(manifest["artifacts"]["FINAL_REFINE_POST_ALIGN"]))

    assert init_stage["interface_convention"] == "closest"
    assert final_refine_post_align["interface_convention"] == "closest"
    assert export_stage["interface_convention"] == "farthest"
    assert render_stage["interface_convention"] == "farthest"
    assert init_stage["source"] == "live_init_window_planes_from_cameras"
    assert final_refine_post_align["source"] == "bundle_cache_proxy"
    assert export_stage["source"] == "calibrated_camera_files"
    assert render_stage["source"] == "calibrated_camera_files_proxy"

    common_window_ids = sorted(set(init_stage["windows"]).intersection(export_stage["windows"]), key=int)
    assert common_window_ids

    for window_id in common_window_ids:
        init_window = init_stage["windows"][window_id]
        export_window = export_stage["windows"][window_id]
        assert init_window["plane_pt"] is not None
        assert init_window["plane_n"] is not None
        assert export_window["plane_pt"] is not None
        assert export_window["plane_n"] is not None
        assert export_window["thick_mm"] is not None
        assert np.isfinite(np.asarray(init_window["plane_pt"], dtype=np.float64)).all()
        assert np.isfinite(np.asarray(init_window["plane_n"], dtype=np.float64)).all()
        assert np.isfinite(np.asarray(export_window["plane_pt"], dtype=np.float64)).all()
        assert np.isfinite(np.asarray(export_window["plane_n"], dtype=np.float64)).all()


def _make_cam_params(*, center, rvec=None):
    center = np.asarray(center, dtype=np.float64)
    rvec = np.zeros(3, dtype=np.float64) if rvec is None else np.asarray(rvec, dtype=np.float64)
    rotation, _ = cv2.Rodrigues(rvec)
    tvec = (-rotation @ center.reshape(3, 1)).reshape(3,)
    return np.array([
        rvec[0],
        rvec[1],
        rvec[2],
        tvec[0],
        tvec[1],
        tvec[2],
        1000.0,
        640.0,
        480.0,
        0.0,
        0.0,
    ], dtype=np.float64)


def _angle_deg(lhs, rhs):
    lhs = np.asarray(lhs, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    lhs = lhs / np.linalg.norm(lhs)
    rhs = rhs / np.linalg.norm(rhs)
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(lhs, rhs)), -1.0, 1.0))))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_init_window_planes_seed_from_camera_cluster_to_object_centroid_direction():
    cam_params = {
        0: _make_cam_params(center=[-120.0, -20.0, 0.0]),
        1: _make_cam_params(center=[-120.0, 20.0, 0.0]),
    }
    cam_to_window = {0: 0, 1: 0}
    window_media = {0: {"thickness": 10.0}}
    err_px = {0: 0.1, 1: 0.1}
    x_a_list = {
        0: np.array([-2.0, -1.0, 8.0]),
        1: np.array([1.0, 1.0, 9.0]),
        2: np.array([0.0, -2.0, 10.0]),
    }
    x_b_list = {
        0: np.array([2.0, 1.0, 12.0]),
        1: np.array([3.0, 2.0, 11.0]),
        2: np.array([4.0, -1.0, 12.0]),
    }

    result = PlaneInitializer.init_window_planes_from_cameras(
        cam_params=cam_params,
        cam_to_window=cam_to_window,
        window_media=window_media,
        err_px=err_px,
        X_A_list=x_a_list,
        X_B_list=x_b_list,
        active_cam_ids=[0, 1],
    )

    plane = result[0]
    centers = np.array([[-120.0, -20.0, 0.0], [-120.0, 20.0, 0.0]], dtype=np.float64)
    c_mean = np.mean(centers, axis=0)
    x_mids = np.array([0.5 * (x_a_list[fid] + x_b_list[fid]) for fid in sorted(x_a_list)], dtype=np.float64)
    expected_normal = np.mean(x_mids, axis=0) - c_mean
    expected_normal = expected_normal / np.linalg.norm(expected_normal)
    expected_depth = float(np.median(np.linalg.norm(x_mids - c_mean.reshape(1, 3), axis=1)))

    assert _angle_deg(plane["plane_n"], expected_normal) < 10.0
    assert np.linalg.norm(np.asarray(plane["plane_pt"]) - c_mean) == pytest.approx(expected_depth / 1.333, abs=1e-6)


def test_init_window_planes_plane_point_uses_depth_over_n_object():
    cam_params = {
        0: _make_cam_params(center=[-120.0, -20.0, 0.0]),
        1: _make_cam_params(center=[-120.0, 20.0, 0.0]),
    }
    cam_to_window = {0: 0, 1: 0}
    window_media = {0: {"thickness": 10.0, "n_object": 2.0}}
    err_px = {0: 0.1, 1: 0.1}
    x_a_list = {
        0: np.array([-2.0, -1.0, 8.0]),
        1: np.array([1.0, 1.0, 9.0]),
        2: np.array([0.0, -2.0, 10.0]),
    }
    x_b_list = {
        0: np.array([2.0, 1.0, 12.0]),
        1: np.array([3.0, 2.0, 11.0]),
        2: np.array([4.0, -1.0, 12.0]),
    }

    result = PlaneInitializer.init_window_planes_from_cameras(
        cam_params=cam_params,
        cam_to_window=cam_to_window,
        window_media=window_media,
        err_px=err_px,
        X_A_list=x_a_list,
        X_B_list=x_b_list,
        active_cam_ids=[0, 1],
    )

    plane = result[0]
    centers = np.array([[-120.0, -20.0, 0.0], [-120.0, 20.0, 0.0]], dtype=np.float64)
    c_mean = np.mean(centers, axis=0)
    x_mids = np.array([0.5 * (x_a_list[fid] + x_b_list[fid]) for fid in sorted(x_a_list)], dtype=np.float64)
    expected_depth = float(np.median(np.linalg.norm(x_mids - c_mean.reshape(1, 3), axis=1)))

    assert np.linalg.norm(np.asarray(plane["plane_pt"]) - c_mean) == pytest.approx(expected_depth / 2.0, abs=1e-6)


def test_init_window_planes_plane_point_falls_back_to_n3_then_default_n_object():
    cam_params = {
        0: _make_cam_params(center=[-120.0, -20.0, 0.0]),
        1: _make_cam_params(center=[-120.0, 20.0, 0.0]),
    }
    cam_to_window = {0: 0, 1: 0}
    err_px = {0: 0.1, 1: 0.1}
    x_a_list = {
        0: np.array([-2.0, -1.0, 8.0]),
        1: np.array([1.0, 1.0, 9.0]),
        2: np.array([0.0, -2.0, 10.0]),
    }
    x_b_list = {
        0: np.array([2.0, 1.0, 12.0]),
        1: np.array([3.0, 2.0, 11.0]),
        2: np.array([4.0, -1.0, 12.0]),
    }
    centers = np.array([[-120.0, -20.0, 0.0], [-120.0, 20.0, 0.0]], dtype=np.float64)
    c_mean = np.mean(centers, axis=0)
    x_mids = np.array([0.5 * (x_a_list[fid] + x_b_list[fid]) for fid in sorted(x_a_list)], dtype=np.float64)
    expected_depth = float(np.median(np.linalg.norm(x_mids - c_mean.reshape(1, 3), axis=1)))

    n3_result = PlaneInitializer.init_window_planes_from_cameras(
        cam_params=cam_params,
        cam_to_window=cam_to_window,
        window_media={0: {"thickness": 10.0, "n3": 4.0}},
        err_px=err_px,
        X_A_list=x_a_list,
        X_B_list=x_b_list,
        active_cam_ids=[0, 1],
    )
    default_result = PlaneInitializer.init_window_planes_from_cameras(
        cam_params=cam_params,
        cam_to_window=cam_to_window,
        window_media={0: {"thickness": 10.0}},
        err_px=err_px,
        X_A_list=x_a_list,
        X_B_list=x_b_list,
        active_cam_ids=[0, 1],
    )

    assert np.linalg.norm(np.asarray(n3_result[0]["plane_pt"]) - c_mean) == pytest.approx(expected_depth / 4.0, abs=1e-6)
    assert np.linalg.norm(np.asarray(default_result[0]["plane_pt"]) - c_mean) == pytest.approx(expected_depth / 1.333, abs=1e-6)


def test_init_window_planes_camera_centers_override_takes_precedence_over_cam_params():
    cam_params = {
        0: _make_cam_params(center=[-20.0, -5.0, 0.0]),
        1: _make_cam_params(center=[-20.0, 5.0, 0.0]),
    }
    cam_to_window = {0: 0, 1: 0}
    window_media = {0: {"thickness": 10.0}}
    err_px = {0: 0.1, 1: 0.1}
    camera_centers_override = {
        0: np.array([-120.0, -20.0, 0.0], dtype=np.float64),
        1: np.array([-120.0, 20.0, 0.0], dtype=np.float64),
    }
    x_a_list = {
        0: np.array([-2.0, -1.0, 8.0]),
        1: np.array([1.0, 1.0, 9.0]),
        2: np.array([0.0, -2.0, 10.0]),
    }
    x_b_list = {
        0: np.array([2.0, 1.0, 12.0]),
        1: np.array([3.0, 2.0, 11.0]),
        2: np.array([4.0, -1.0, 12.0]),
    }

    result = PlaneInitializer.init_window_planes_from_cameras(
        cam_params=cam_params,
        cam_to_window=cam_to_window,
        window_media=window_media,
        err_px=err_px,
        X_A_list=x_a_list,
        X_B_list=x_b_list,
        active_cam_ids=[0, 1],
        camera_centers_override=camera_centers_override,
    )

    plane = result[0]
    centers = np.array([camera_centers_override[0], camera_centers_override[1]], dtype=np.float64)
    c_mean = np.mean(centers, axis=0)
    x_mids = np.array([0.5 * (x_a_list[fid] + x_b_list[fid]) for fid in sorted(x_a_list)], dtype=np.float64)
    expected_normal = np.mean(x_mids, axis=0) - c_mean
    expected_normal = expected_normal / np.linalg.norm(expected_normal)
    expected_depth = float(np.median(np.linalg.norm(x_mids - c_mean.reshape(1, 3), axis=1)))

    assert _angle_deg(plane["plane_n"], expected_normal) < 10.0
    assert np.linalg.norm(np.asarray(plane["plane_pt"]) - c_mean) == pytest.approx(expected_depth / 1.333, abs=1e-6)


def test_should_pause_for_stall_detects_three_small_repeated_improvements():
    assert hasattr(case_026_plane_debug_loop, "should_pause_for_stall")
    assert case_026_plane_debug_loop.should_pause_for_stall(
        [15.0, 14.5, 14.2, 14.0],
        ["fix_normal_seed", "fix_normal_seed", "fix_normal_seed"],
    ) is True


def _seed_iterate_inputs(results_dir: Path):
    iterations_dir = results_dir / "iterations"
    iterations_dir.mkdir(parents=True)
    baseline_summary = {
        "artifact": "baseline_divergence_summary",
        "selected_branch": "A",
        "branch_rationale": "INIT diverges first.",
        "earliest_divergence_stage": "INIT",
        "earliest_divergence_sub_stage": None,
        "primary_metric": "max_angular_error_deg",
        "secondary_metric": "max_point_distance_mm",
        "primary_threshold_deg": 2.0,
        "secondary_threshold_mm": 1.0,
        "stable_budget": 25,
        "tested_strategies": ["sequential", "evenly_spaced"],
        "tested_budgets": [25],
        "per_stage_metrics": {
            "INIT": {
                "max_angular_error_deg": 90.0,
                "max_point_distance_mm": 156.5,
                "per_window": {
                    "0": {"angular_error_deg": 90.0, "point_distance_mm": 62.3, "stage_present": True, "gt_present": True},
                    "1": {"angular_error_deg": 90.0, "point_distance_mm": 156.5, "stage_present": True, "gt_present": True},
                },
            }
        },
    }
    request = {
        "artifact": "metis_request",
        "iteration": 0,
        "task": "Analyze baseline divergence and propose first bounded fix",
        "selected_branch": "A",
        "earliest_divergence_stage": "INIT",
        "primary_metric_at_divergence": 90.0,
        "secondary_metric_at_divergence": 156.5,
        "target_primary_metric_deg": 2.0,
        "target_secondary_metric_mm": 1.0,
        "stable_budget": 25,
        "trace_results_dir": str(results_dir / "trace"),
        "gt_planes_path": str(results_dir / "gt_planes.json"),
        "baseline_summary_path": str(results_dir / "baseline_divergence_summary.json"),
        "allowed_fix_files": [
            "modules/camera_calibration/wand_calibration/refraction_wand_calibrator.py",
            "modules/camera_calibration/wand_calibration/refractive_geometry.py",
            "modules/camera_calibration/wand_calibration/refraction_calibration_BA.py",
        ],
    }
    (results_dir / "baseline_divergence_summary.json").write_text(json.dumps(baseline_summary), encoding="utf-8")
    (iterations_dir / "iteration_000_metis_request.json").write_text(json.dumps(request), encoding="utf-8")
    return iterations_dir


def _patch_trace_inputs_for_live_init(
    monkeypatch,
    *,
    live_init_plane,
    bundle_plane,
    gt_window=None,
    initializer_impl=None,
    case_meta_override=None,
    bundle_cache_override=None,
):
    case_meta = case_meta_override or {
        "wand": {"n_frames": 2},
        "cameras": [
            {"cam_id": 0, "plane_id": 0, "C_world": [-120.0, -20.0, 0.0]},
            {"cam_id": 1, "plane_id": 0, "C_world": [-120.0, 20.0, 0.0]},
        ],
        "planes": [{"id": 0, "pt_far": [12.0, 0.0, 0.0], "n": [1.0, 0.0, 0.0]}],
    }
    bundle_cache = bundle_cache_override or {
        "cam_ids": [0, 1],
        "window_ids": [0],
        "cam_params": {
            "0": _make_cam_params(center=[-20.0, -5.0, 0.0]).tolist(),
            "1": _make_cam_params(center=[-20.0, 5.0, 0.0]).tolist(),
        },
        "window_media": {
            "0": {
                "n_air": 1.0,
                "n_window": 1.49,
                "n_object": 1.333,
                "thickness": 2.0,
            }
        },
        "planes": {
            "0": {
                "plane_pt": bundle_plane["plane_pt"],
                "plane_n": bundle_plane["plane_n"],
            }
        },
        "points_3d": [
            10.0, 0.0, 0.0,
            14.0, 0.0, 0.0,
            11.0, 1.0, 0.0,
            15.0, 1.0, 0.0,
        ],
    }
    gt_payload = {
        "artifact": "gt_planes",
        "interface_convention": "farthest",
        "windows": {
            "0": gt_window or _make_gt_window(plane_pt_far=[12.0, 0.0, 0.0], plane_n=[1.0, 0.0, 0.0], thick_mm=2.0)
        },
    }

    monkeypatch.setattr(case_026_plane_debug_loop, "_resolve_case_inputs", lambda **kwargs: (Path("fake_case"), Path("fake_bundle"), Path("fake_gt.csv")))
    monkeypatch.setattr(case_026_plane_debug_loop, "_load_case_meta", lambda path: case_meta)
    monkeypatch.setattr(case_026_plane_debug_loop, "_load_bundle_cache", lambda path: bundle_cache)
    monkeypatch.setattr(case_026_plane_debug_loop, "_calibrated_export_dir_from_bundle_cache_path", lambda path: Path("fake_export"))
    monkeypatch.setattr(case_026_plane_debug_loop, "_discover_calibrated_cam_files", lambda path: [])
    monkeypatch.setattr(case_026_plane_debug_loop, "_frame_ids_from_csv", lambda path: [0, 1])
    monkeypatch.setattr(case_026_plane_debug_loop, "_derive_gt_planes_artifact", lambda case_dir, meta: gt_payload)
    monkeypatch.setattr(case_026_plane_debug_loop, "load_refractive_dataset", lambda *args, **kwargs: {"frames": [0, 1], "cam_ids": [0, 1]})
    monkeypatch.setattr(case_026_plane_debug_loop, "build_refractive_cams_cpp", lambda *args, **kwargs: {0: object(), 1: object()})
    monkeypatch.setattr(case_026_plane_debug_loop, "RefractiveBAConfig", lambda **kwargs: SimpleNamespace(**kwargs), raising=False)
    monkeypatch.setattr(
        case_026_plane_debug_loop,
        "RefractiveBAOptimizer",
        lambda **kwargs: SimpleNamespace(optimize=lambda skip_optimization=False, stage=None: (kwargs["window_planes"], kwargs["cam_params"])),
        raising=False,
    )
    if initializer_impl is None:
        initializer_impl = lambda **kwargs: {0: live_init_plane}
    monkeypatch.setattr(
        case_026_plane_debug_loop,
        "PlaneInitializer",
        SimpleNamespace(init_window_planes_from_cameras=initializer_impl),
        raising=False,
    )


def test_run_iterate_reports_continue_needed_with_computed_post_fix_metrics(tmp_path):
    results_dir = tmp_path / "case_026_plane_debug"
    iterations_dir = _seed_iterate_inputs(results_dir)

    original_run_trace = case_026_plane_debug_loop.run_trace

    def fake_run_trace(**kwargs):
        root = Path(kwargs["results_dir"])
        trace_dir = root / "trace"
        trace_dir.mkdir(parents=True, exist_ok=True)
        gt_path = root / "gt_planes.json"
        gt_payload = {
            "artifact": "gt_planes",
            "interface_convention": "farthest",
            "windows": {
                "0": _make_gt_window(plane_pt_far=[10.0, 0.0, 0.0], plane_n=[1.0, 0.0, 0.0], thick_mm=2.0),
            },
        }
        gt_path.write_text(json.dumps(gt_payload), encoding="utf-8")
        stage_path = trace_dir / "stage_INIT.json"
        stage_path.write_text(
            json.dumps(
                {
                    "stage": "INIT",
                    "interface_convention": "closest",
                    "windows": {
                        "0": {"plane_pt": [0.0, 0.0, 0.0], "plane_n": [0.0, 1.0, 0.0], "thick_mm": 2.0},
                    },
                }
            ),
            encoding="utf-8",
        )
        return {
            "mode": "trace",
            "ba_frame_budget": kwargs.get("frame_budget"),
            "ba_frame_selection": kwargs.get("frame_selection"),
            "gt_planes_path": str(gt_path),
            "artifacts": {"INIT": str(stage_path)},
        }

    case_026_plane_debug_loop.run_trace = fake_run_trace
    try:
        result = case_026_plane_debug_loop.run_iterate(results_dir=results_dir)
    finally:
        case_026_plane_debug_loop.run_trace = original_run_trace

    response_path = iterations_dir / "iteration_001_metis_response.json"
    assert response_path.exists()
    response = json.loads(response_path.read_text(encoding="utf-8"))
    assert response["verdict"] == "continue_needed"
    assert response["termination_state"] == "continue_needed"
    assert response["post_fix_metrics"]["max_angular_error_deg"] == pytest.approx(90.0)
    assert response["post_fix_metrics"]["max_point_distance_mm"] == pytest.approx(8.0)
    assert response["metric_improvement"]["max_angular_error_deg_pct"] == pytest.approx(0.0)
    assert response["full_calibration_gate"] is False
    assert (iterations_dir / "iteration_002_metis_request.json").exists()
    assert result["metis_response_path"] == str(response_path)
    assert result["verdict"] == "continue_needed"
    assert result["trace_manifest"]["ba_frame_budget"] == 25


def test_run_trace_init_stage_uses_live_initializer_output_and_source(tmp_path, monkeypatch):
    live_init_plane = {
        "plane_pt": np.array([10.0, 0.0, 0.0], dtype=np.float64),
        "plane_n": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "thick_mm": 2.0,
        "initialized": True,
    }
    bundle_plane = {
        "plane_pt": [0.0, 4.0, 0.0],
        "plane_n": [0.0, 1.0, 0.0],
    }
    _patch_trace_inputs_for_live_init(monkeypatch, live_init_plane=live_init_plane, bundle_plane=bundle_plane)

    manifest = case_026_plane_debug_loop.run_trace(results_dir=tmp_path)
    init_stage = _read_json(Path(manifest["artifacts"]["INIT"]))
    alt_loop_stage = _read_json(Path(manifest["artifacts"]["ALT_LOOP"]))

    assert init_stage["source"] == "live_init_window_planes_from_cameras"
    np.testing.assert_allclose(init_stage["windows"]["0"]["plane_pt"], [10.0, 0.0, 0.0], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(init_stage["windows"]["0"]["plane_n"], [1.0, 0.0, 0.0], atol=1e-12, rtol=0.0)
    assert alt_loop_stage["source"] == "bundle_cache_proxy"
    np.testing.assert_allclose(alt_loop_stage["windows"]["0"]["plane_pt"], [0.0, 4.0, 0.0], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(alt_loop_stage["windows"]["0"]["plane_n"], [0.0, 1.0, 0.0], atol=1e-12, rtol=0.0)


def test_run_trace_passes_case_meta_world_camera_centers_override_to_live_init(tmp_path, monkeypatch):
    live_init_plane = {
        "plane_pt": np.array([10.0, 0.0, 0.0], dtype=np.float64),
        "plane_n": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "thick_mm": 2.0,
        "initialized": True,
    }
    bundle_plane = {
        "plane_pt": [0.0, 4.0, 0.0],
        "plane_n": [0.0, 1.0, 0.0],
    }
    captured = {}

    def fake_initializer(**kwargs):
        captured.update(kwargs)
        return {0: live_init_plane}

    _patch_trace_inputs_for_live_init(
        monkeypatch,
        live_init_plane=live_init_plane,
        bundle_plane=bundle_plane,
        initializer_impl=fake_initializer,
    )

    manifest = case_026_plane_debug_loop.run_trace(results_dir=tmp_path)
    init_stage = _read_json(Path(manifest["artifacts"]["INIT"]))

    assert init_stage["source"] == "live_init_window_planes_from_cameras"
    np.testing.assert_allclose(captured["camera_centers_override"][0], [-120.0, -20.0, 0.0], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(captured["camera_centers_override"][1], [-120.0, 20.0, 0.0], atol=1e-12, rtol=0.0)
    assert "C_world" in init_stage["metadata"]["init_inputs"]["case_meta_keys"]


def test_run_trace_transforms_live_init_points_into_world_center_frame(tmp_path, monkeypatch):
    live_init_plane = {
        "plane_pt": np.array([112.0, 0.0, 0.0], dtype=np.float64),
        "plane_n": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "thick_mm": 2.0,
        "initialized": True,
    }
    bundle_plane = {
        "plane_pt": [0.0, 4.0, 0.0],
        "plane_n": [0.0, 1.0, 0.0],
    }
    captured = {}
    case_meta = {
        "wand": {"n_frames": 2},
        "cameras": [
            {"cam_id": 0, "plane_id": 0, "C_world": [80.0, -5.0, 0.0]},
            {"cam_id": 1, "plane_id": 0, "C_world": [80.0, 5.0, 0.0]},
        ],
        "planes": [{"id": 0, "pt_far": [112.0, 0.0, 0.0], "n": [1.0, 0.0, 0.0]}],
    }

    def fake_initializer(**kwargs):
        captured.update(kwargs)
        return {0: live_init_plane}

    _patch_trace_inputs_for_live_init(
        monkeypatch,
        live_init_plane=live_init_plane,
        bundle_plane=bundle_plane,
        case_meta_override=case_meta,
        initializer_impl=fake_initializer,
    )

    manifest = case_026_plane_debug_loop.run_trace(results_dir=tmp_path)
    init_stage = _read_json(Path(manifest["artifacts"]["INIT"]))

    np.testing.assert_allclose(captured["X_A_list"][0], [110.0, 0.0, 0.0], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(captured["X_B_list"][0], [114.0, 0.0, 0.0], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(captured["X_A_list"][1], [111.0, 1.0, 0.0], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(captured["X_B_list"][1], [115.0, 1.0, 0.0], atol=1e-12, rtol=0.0)
    assert init_stage["metadata"]["init_inputs"]["points_source"] == "bundle_cache points_3d rigid-aligned into case_meta C_world frame"


def test_run_trace_records_live_ba_stage_after_init(tmp_path, monkeypatch):
    live_init_plane = {
        "plane_pt": np.array([10.0, 0.0, 0.0], dtype=np.float64),
        "plane_n": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "thick_mm": 2.0,
        "initialized": True,
    }
    bundle_plane = {
        "plane_pt": [0.0, 4.0, 0.0],
        "plane_n": [0.0, 1.0, 0.0],
    }
    ba_window_planes = {
        0: {
            "plane_pt": np.array([12.0, 0.0, 0.0], dtype=np.float64),
            "plane_n": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        }
    }
    captured = {}

    class FakeBAOptimizer:
        def __init__(self, **kwargs):
            captured["init_kwargs"] = kwargs
            self.window_planes = kwargs["window_planes"]
            self.cam_params = kwargs["cam_params"]

        def optimize(self, skip_optimization=False, stage=None):
            captured["optimize_kwargs"] = {"skip_optimization": skip_optimization, "stage": stage}
            return ba_window_planes, self.cam_params

    _patch_trace_inputs_for_live_init(monkeypatch, live_init_plane=live_init_plane, bundle_plane=bundle_plane)
    monkeypatch.setattr(case_026_plane_debug_loop, "RefractiveBAOptimizer", FakeBAOptimizer, raising=False)
    monkeypatch.setattr(case_026_plane_debug_loop, "RefractiveBAConfig", lambda **kwargs: SimpleNamespace(**kwargs), raising=False)

    manifest = case_026_plane_debug_loop.run_trace(results_dir=tmp_path, frame_budget=2)
    joint_stage = _read_json(Path(manifest["artifacts"]["JOINT_BA"]))
    gt_planes = _read_json(Path(manifest["gt_planes_path"]))
    comparison = compare_planes(joint_stage, gt_planes)

    assert joint_stage["source"] == "live_ba_after_init"
    np.testing.assert_allclose(joint_stage["windows"]["0"]["plane_pt"], [10.0, 0.0, 0.0], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(joint_stage["windows"]["0"]["plane_n"], [1.0, 0.0, 0.0], atol=1e-12, rtol=0.0)
    assert joint_stage["metadata"]["ba_execution"]["executed"] is True
    assert joint_stage["metadata"]["ba_execution"]["stage"] == 3
    assert joint_stage["metadata"]["ba_execution"]["seed_stage"] == "INIT"
    assert comparison["windows"]["0"]["point_distance_mm"] == pytest.approx(0.0)
    assert captured["init_kwargs"]["window_planes"][0]["plane_pt"].tolist() == [10.0, 0.0, 0.0]
    assert captured["optimize_kwargs"] == {"skip_optimization": False, "stage": 3}


def test_ba_optimize_records_alt_loop_trace_snapshots(monkeypatch):
    optimizer = RefractiveBAOptimizer(
        dataset={"obsA": {}, "obsB": {}, "frames": []},
        cam_params={0: _make_cam_params(center=[-120.0, 0.0, 0.0])},
        cams_cpp={},
        cam_to_window={0: 0},
        window_media={0: {"thickness": 2.0, "n_object": 1.333}},
        window_planes={0: {"plane_pt": [10.0, 0.0, 0.0], "plane_n": [1.0, 0.0, 0.0]}},
        wand_length=4.0,
        config=RefractiveBAConfig(stage=3, verbosity=0),
    )

    weakwin_plane_pt = np.array([20.0, 0.0, 0.0], dtype=np.float64)
    plane_solve_plane_pt = np.array([30.0, 0.0, 0.0], dtype=np.float64)
    align_plane_pt = np.array([40.0, 0.0, 0.0], dtype=np.float64)
    plane_n = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    monkeypatch.setattr(optimizer, "_compute_physical_sigmas", lambda: None)
    monkeypatch.setattr(optimizer, "_build_step_a_plane_d_bounds", lambda loop_iter: {})
    monkeypatch.setattr(optimizer, "_get_chunk_schedule_for_mode", lambda mode: [])
    monkeypatch.setattr(optimizer, "_print_plane_diagnostics", lambda *args, **kwargs: None)
    monkeypatch.setattr(optimizer, "evaluate_residuals", lambda *args, **kwargs: (np.zeros(0), 0.0, 0.0, 0, 0, 0.0, 0))
    monkeypatch.setattr(optimizer, "print_diagnostics", lambda: None)
    monkeypatch.setattr(optimizer, "sync_cpp_state", lambda **kwargs: None)

    def fake_detect_weak_windows():
        optimizer.window_planes[0]["plane_pt"] = weakwin_plane_pt.copy()
        optimizer.window_planes[0]["plane_n"] = plane_n.copy()

    def fake_optimize_generic(**kwargs):
        if kwargs["mode"] == "loop_1_planes":
            optimizer.window_planes[0]["plane_pt"] = plane_solve_plane_pt.copy()
            optimizer.window_planes[0]["plane_n"] = plane_n.copy()
            return SimpleNamespace(active_mask=np.zeros(3, dtype=int)), [
                ("plane_d", 0, 0),
                ("plane_a", 0, 0),
                ("plane_b", 0, 0),
            ]
        if kwargs["mode"] == "loop_1_cams":
            return SimpleNamespace(active_mask=np.zeros(0, dtype=int)), []
        if kwargs["mode"] == "joint":
            return SimpleNamespace(active_mask=np.zeros(0, dtype=int)), []
        raise AssertionError(f"Unexpected mode: {kwargs['mode']}")

    def fake_apply_coordinate_alignment(tag, refresh_initial=True, align_mode="yz", **kwargs):
        optimizer.window_planes[0]["plane_pt"] = align_plane_pt.copy()
        optimizer.window_planes[0]["plane_n"] = plane_n.copy()
        return True

    monkeypatch.setattr(optimizer, "_detect_weak_windows", fake_detect_weak_windows)
    monkeypatch.setattr(optimizer, "_optimize_generic", fake_optimize_generic)
    monkeypatch.setattr(optimizer, "_apply_coordinate_alignment", fake_apply_coordinate_alignment)

    optimizer.optimize(stage=3)

    snapshots = cast(dict[str, dict[str, Any]], optimizer.trace_stage_snapshots)

    assert set(snapshots) >= {
        "ALT_LOOP_POST_WEAKWIN",
        "ALT_LOOP_POST_PLANE_SOLVE",
        "ALT_LOOP_POST_ALIGN",
    }
    np.testing.assert_allclose(
        snapshots["ALT_LOOP_POST_WEAKWIN"]["window_planes"][0]["plane_pt"],
        weakwin_plane_pt,
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        snapshots["ALT_LOOP_POST_PLANE_SOLVE"]["window_planes"][0]["plane_pt"],
        plane_solve_plane_pt,
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        snapshots["ALT_LOOP_POST_ALIGN"]["window_planes"][0]["plane_pt"],
        align_plane_pt,
        atol=1e-12,
        rtol=0.0,
    )


def test_closest_windows_payload_can_shift_plane_point_to_closest_interface():
    windows = case_026_plane_debug_loop._closest_windows_payload(
        [0],
        {0: {"plane_pt": np.array([13.0, 0.0, 0.0], dtype=np.float64), "plane_n": np.array([1.0, 0.0, 0.0], dtype=np.float64)}},
        {0: {"thickness": 2.0}},
        apply_closest_interface_shift=True,
    )

    np.testing.assert_allclose(windows["0"]["plane_pt"], [11.0, 0.0, 0.0], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(windows["0"]["plane_n"], [1.0, 0.0, 0.0], atol=1e-12, rtol=0.0)
    assert windows["0"]["thick_mm"] == pytest.approx(2.0)


def test_run_trace_restores_joint_ba_planes_back_to_case_frame(tmp_path, monkeypatch):
    live_init_plane = {
        "plane_pt": np.array([10.0, 0.0, 0.0], dtype=np.float64),
        "plane_n": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "thick_mm": 2.0,
        "initialized": True,
    }
    bundle_plane = {
        "plane_pt": [0.0, 4.0, 0.0],
        "plane_n": [0.0, 1.0, 0.0],
    }
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    translation = np.array([5.0, 7.0, 0.0], dtype=np.float64)
    ba_window_planes = {
        0: {
            "plane_pt": np.array([5.0, 19.0, 0.0], dtype=np.float64),
            "plane_n": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        }
    }

    class FakeBAOptimizer:
        def __init__(self, **kwargs):
            self.cam_params = kwargs["cam_params"]
            self.cumulative_alignment_rotation = rotation
            self.cumulative_alignment_translation = translation

        def optimize(self, skip_optimization=False, stage=None):
            return ba_window_planes, self.cam_params

    _patch_trace_inputs_for_live_init(monkeypatch, live_init_plane=live_init_plane, bundle_plane=bundle_plane)
    monkeypatch.setattr(case_026_plane_debug_loop, "RefractiveBAOptimizer", FakeBAOptimizer, raising=False)
    monkeypatch.setattr(case_026_plane_debug_loop, "RefractiveBAConfig", lambda **kwargs: SimpleNamespace(**kwargs), raising=False)

    manifest = case_026_plane_debug_loop.run_trace(results_dir=tmp_path, frame_budget=2)
    joint_stage = _read_json(Path(manifest["artifacts"]["JOINT_BA"]))

    np.testing.assert_allclose(joint_stage["windows"]["0"]["plane_pt"], [10.0, 0.0, 0.0], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(joint_stage["windows"]["0"]["plane_n"], [1.0, 0.0, 0.0], atol=1e-12, rtol=0.0)
    assert joint_stage["metadata"]["ba_execution"]["original_frame_restored"] is True
    np.testing.assert_allclose(joint_stage["metadata"]["ba_execution"]["cumulative_alignment_rotation"], rotation, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(joint_stage["metadata"]["ba_execution"]["cumulative_alignment_translation"], translation, atol=1e-12, rtol=0.0)


def test_run_trace_uses_live_alt_loop_substage_snapshots_instead_of_proxies(tmp_path, monkeypatch):
    live_init_plane = {
        "plane_pt": np.array([10.0, 0.0, 0.0], dtype=np.float64),
        "plane_n": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "thick_mm": 2.0,
        "initialized": True,
    }
    bundle_plane = {
        "plane_pt": [0.0, 4.0, 0.0],
        "plane_n": [0.0, 1.0, 0.0],
    }
    plane_n = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    alt_post_weakwin = {0: {"plane_pt": np.array([20.0, 0.0, 0.0], dtype=np.float64), "plane_n": plane_n.copy()}}
    alt_post_plane = {0: {"plane_pt": np.array([30.0, 0.0, 0.0], dtype=np.float64), "plane_n": plane_n.copy()}}
    alt_post_align = {0: {"plane_pt": np.array([40.0, 0.0, 0.0], dtype=np.float64), "plane_n": plane_n.copy()}}
    joint_final = {0: {"plane_pt": np.array([50.0, 0.0, 0.0], dtype=np.float64), "plane_n": plane_n.copy()}}

    class FakeBAOptimizer:
        def __init__(self, **kwargs):
            self.cam_params = kwargs["cam_params"]
            self.trace_stage_snapshots = {
                "ALT_LOOP_POST_WEAKWIN": {
                    "window_planes": alt_post_weakwin,
                    "cumulative_alignment_rotation": np.eye(3, dtype=np.float64),
                    "cumulative_alignment_translation": np.zeros(3, dtype=np.float64),
                    "alignment_history": [],
                },
                "ALT_LOOP_POST_PLANE_SOLVE": {
                    "window_planes": alt_post_plane,
                    "cumulative_alignment_rotation": np.eye(3, dtype=np.float64),
                    "cumulative_alignment_translation": np.zeros(3, dtype=np.float64),
                    "alignment_history": [],
                },
                "ALT_LOOP_POST_ALIGN": {
                    "window_planes": alt_post_align,
                    "cumulative_alignment_rotation": np.eye(3, dtype=np.float64),
                    "cumulative_alignment_translation": np.zeros(3, dtype=np.float64),
                    "alignment_history": [{"tag": "pre-last-loop-cam"}],
                },
            }

        def optimize(self, skip_optimization=False, stage=None):
            return joint_final, self.cam_params

    _patch_trace_inputs_for_live_init(monkeypatch, live_init_plane=live_init_plane, bundle_plane=bundle_plane)
    monkeypatch.setattr(case_026_plane_debug_loop, "RefractiveBAOptimizer", FakeBAOptimizer, raising=False)
    monkeypatch.setattr(case_026_plane_debug_loop, "RefractiveBAConfig", lambda **kwargs: SimpleNamespace(**kwargs), raising=False)

    manifest = case_026_plane_debug_loop.run_trace(results_dir=tmp_path, frame_budget=2)
    weakwin_stage = _read_json(Path(manifest["artifacts"]["ALT_LOOP_POST_WEAKWIN"]))
    plane_stage = _read_json(Path(manifest["artifacts"]["ALT_LOOP_POST_PLANE_SOLVE"]))
    align_stage = _read_json(Path(manifest["artifacts"]["ALT_LOOP_POST_ALIGN"]))
    joint_stage = _read_json(Path(manifest["artifacts"]["JOINT_BA"]))

    assert weakwin_stage["source"] == "live_ba_alt_loop_snapshot"
    assert plane_stage["source"] == "live_ba_alt_loop_snapshot"
    assert align_stage["source"] == "live_ba_alt_loop_snapshot"
    np.testing.assert_allclose(weakwin_stage["windows"]["0"]["plane_pt"], [18.0, 0.0, 0.0], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(plane_stage["windows"]["0"]["plane_pt"], [28.0, 0.0, 0.0], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(align_stage["windows"]["0"]["plane_pt"], [38.0, 0.0, 0.0], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(joint_stage["windows"]["0"]["plane_pt"], [48.0, 0.0, 0.0], atol=1e-12, rtol=0.0)
    assert weakwin_stage["windows"]["0"]["plane_pt"] != [0.0, 4.0, 0.0]
    assert align_stage["windows"]["0"]["plane_pt"] != joint_stage["windows"]["0"]["plane_pt"]


def test_ba_alternating_loop_step_b_uses_tightened_camera_bounds(monkeypatch):
    optimizer = RefractiveBAOptimizer(
        dataset={"obsA": {}, "obsB": {}, "frames": []},
        cam_params={0: _make_cam_params(center=[-120.0, 0.0, 0.0])},
        cams_cpp={},
        cam_to_window={0: 0},
        window_media={0: {"thickness": 10.0, "n_object": 1.333}},
        window_planes={0: {"plane_pt": [10.0, 0.0, 0.0], "plane_n": [1.0, 0.0, 0.0]}},
        wand_length=4.0,
        config=RefractiveBAConfig(stage=2, verbosity=0),
    )
    captured = {}

    monkeypatch.setattr(optimizer, "_compute_physical_sigmas", lambda: None)
    monkeypatch.setattr(optimizer, "_detect_weak_windows", lambda: None)
    monkeypatch.setattr(optimizer, "_sync_initial_state", lambda: None)
    monkeypatch.setattr(optimizer, "_build_step_a_plane_d_bounds", lambda loop_iter: None)
    monkeypatch.setattr(optimizer, "_print_plane_diagnostics", lambda *args, **kwargs: None)
    monkeypatch.setattr(optimizer, "_apply_coordinate_alignment", lambda *args, **kwargs: True)
    monkeypatch.setattr(optimizer, "_set_barrier_profile_for_mode", lambda *args, **kwargs: None)
    monkeypatch.setattr(optimizer, "evaluate_residuals", lambda *args, **kwargs: (np.zeros(0), 0.0, 0.0, 0, 0, 0.0, 0))
    monkeypatch.setattr(optimizer, "print_diagnostics", lambda: None)

    def fake_optimize_generic(**kwargs):
        if kwargs["mode"].endswith("_planes"):
            return SimpleNamespace(active_mask=np.zeros(3, dtype=int)), [
                ("plane_d", 0, 0),
                ("plane_a", 0, 0),
                ("plane_b", 0, 0),
            ]
        if kwargs["mode"].endswith("_cams"):
            captured.update(
                limit_rot_rad=kwargs["limit_rot_rad"],
                limit_trans_mm=kwargs["limit_trans_mm"],
            )
            return SimpleNamespace(active_mask=np.zeros(0, dtype=int)), []
        raise AssertionError(f"Unexpected mode: {kwargs['mode']}")

    monkeypatch.setattr(optimizer, "_optimize_generic", fake_optimize_generic)

    optimizer.optimize(stage=2)

    assert captured["limit_rot_rad"] == pytest.approx(np.deg2rad(5.0))
    assert captured["limit_trans_mm"] == pytest.approx(50.0)


@pytest.mark.parametrize("strategy", ["sequence", "bundle"])
def test_adaptive_plane_d_cap_uses_explicit_mode_over_stale_diag_mode_for_step_a(monkeypatch, strategy):
    optimizer = RefractiveBAOptimizer(
        dataset={"obsA": {}, "obsB": {}, "frames": []},
        cam_params={0: _make_cam_params(center=[-120.0, 0.0, 0.0])},
        cams_cpp={},
        cam_to_window={0: 0},
        window_media={0: {"thickness": 10.0, "n_object": 1.333}},
        window_planes={0: {"plane_pt": [80.0, 0.0, 0.0], "plane_n": [1.0, 0.0, 0.0]}},
        wand_length=4.0,
        config=RefractiveBAConfig(verbosity=0),
    )
    optimizer._diag_current_mode = "loop_1_cams"

    monkeypatch.setattr(optimizer, "_compute_physical_sigmas", lambda: None)
    monkeypatch.setattr(optimizer, "_set_barrier_profile_for_mode", lambda *args, **kwargs: None)
    monkeypatch.setattr(optimizer, "sync_cpp_state", lambda **kwargs: None)
    monkeypatch.setattr(
        optimizer,
        "evaluate_residuals",
        lambda *args, **kwargs: (np.zeros(0, dtype=np.float64), 0.0, 0.0, 0, 0, 0.0, 0),
    )

    captured = {}

    def fake_least_squares(*args, **kwargs):
        captured["bounds"] = kwargs["bounds"]
        x0 = np.asarray(args[1], dtype=np.float64)
        return SimpleNamespace(x=x0.copy(), status=1, message="gtol termination", nfev=0)

    monkeypatch.setattr(
        "modules.camera_calibration.wand_calibration.refraction_calibration_BA.least_squares",
        fake_least_squares,
    )

    optimizer._optimize_generic(
        mode="loop_1_planes",
        description=f"unit-{strategy}",
        enable_planes=True,
        enable_cam_t=False,
        enable_cam_r=False,
        limit_rot_rad=0.0,
        limit_trans_mm=0.0,
        limit_plane_d_mm=500.0,
        limit_plane_angle_rad=np.deg2rad(5.0),
        strategy_override=strategy,
    )

    lower_bounds, upper_bounds = captured["bounds"]
    assert lower_bounds[0] == pytest.approx(-6.0)
    assert upper_bounds[0] == pytest.approx(6.0)


@pytest.mark.parametrize("strategy", ["sequence", "bundle"])
def test_loop_camera_translation_bounds_scale_with_camera_plane_depth(monkeypatch, strategy):
    optimizer = RefractiveBAOptimizer(
        dataset={"obsA": {}, "obsB": {}, "frames": []},
        cam_params={
            0: _make_cam_params(center=[-120.0, 0.0, 0.0]),
            1: _make_cam_params(center=[-20.0, 0.0, 0.0]),
        },
        cams_cpp={},
        cam_to_window={0: 0, 1: 0},
        window_media={0: {"thickness": 10.0, "n_object": 1.333}},
        window_planes={0: {"plane_pt": [80.0, 0.0, 0.0], "plane_n": [1.0, 0.0, 0.0]}},
        wand_length=4.0,
        config=RefractiveBAConfig(verbosity=0),
    )

    monkeypatch.setattr(optimizer, "_compute_physical_sigmas", lambda: None)
    monkeypatch.setattr(optimizer, "_set_barrier_profile_for_mode", lambda *args, **kwargs: None)
    monkeypatch.setattr(optimizer, "sync_cpp_state", lambda **kwargs: None)
    monkeypatch.setattr(
        optimizer,
        "evaluate_residuals",
        lambda *args, **kwargs: (np.zeros(0, dtype=np.float64), 0.0, 0.0, 0, 0, 0.0, 0),
    )

    captured = {}

    def fake_least_squares(*args, **kwargs):
        captured["bounds"] = kwargs["bounds"]
        x0 = np.asarray(args[1], dtype=np.float64)
        return SimpleNamespace(x=x0.copy(), status=1, message="gtol termination", nfev=0)

    monkeypatch.setattr(
        "modules.camera_calibration.wand_calibration.refraction_calibration_BA.least_squares",
        fake_least_squares,
    )

    optimizer._optimize_generic(
        mode="loop_1_cams",
        description=f"cam-bound-{strategy}",
        enable_planes=False,
        enable_cam_t=True,
        enable_cam_r=False,
        limit_rot_rad=0.0,
        limit_trans_mm=50.0,
        limit_plane_d_mm=0.0,
        limit_plane_angle_rad=0.0,
        strategy_override=strategy,
    )

    lower_bounds, upper_bounds = captured["bounds"]
    assert lower_bounds.shape[0] == 6
    assert upper_bounds.shape[0] == 6

    expected_limits = [20.0, 20.0, 20.0, 10.0, 10.0, 10.0]
    assert list(lower_bounds) == pytest.approx([-v for v in expected_limits])
    assert list(upper_bounds) == pytest.approx(expected_limits)


def test_adaptive_plane_d_limit_uses_diag_mode_when_explicit_mode_missing():
    optimizer = RefractiveBAOptimizer(
        dataset={"obsA": {}, "obsB": {}, "frames": []},
        cam_params={0: _make_cam_params(center=[-120.0, 0.0, 0.0])},
        cams_cpp={},
        cam_to_window={0: 0},
        window_media={0: {"thickness": 10.0, "n_object": 1.333}},
        window_planes={0: {"plane_pt": [80.0, 0.0, 0.0], "plane_n": [1.0, 0.0, 0.0]}},
        wand_length=4.0,
        config=RefractiveBAConfig(verbosity=0),
    )
    optimizer._diag_current_mode = "loop_1_planes"

    assert optimizer._get_adaptive_plane_d_limit(500.0) == pytest.approx(6.0)


def test_adaptive_plane_d_limit_keeps_non_joint_formula_unchanged():
    optimizer = RefractiveBAOptimizer(
        dataset={"obsA": {}, "obsB": {}, "frames": []},
        cam_params={0: _make_cam_params(center=[-120.0, 0.0, 0.0])},
        cams_cpp={},
        cam_to_window={0: 0},
        window_media={0: {"thickness": 10.0, "n_object": 1.333}},
        window_planes={0: {"plane_pt": [80.0, 0.0, 0.0], "plane_n": [1.0, 0.0, 0.0]}},
        wand_length=4.0,
        config=RefractiveBAConfig(verbosity=0),
    )
    optimizer._plane_d0 = {0: 358.835, 1: 357.547, 2: 349.411}

    assert optimizer._get_adaptive_plane_d_limit(50.0, mode="loop_1_planes") == pytest.approx(10.48233)


def test_adaptive_plane_d_limit_logs_rich_joint_diagnostics():
    optimizer = RefractiveBAOptimizer(
        dataset={"obsA": {}, "obsB": {}, "frames": []},
        cam_params={0: _make_cam_params(center=[-120.0, 0.0, 0.0])},
        cams_cpp={},
        cam_to_window={0: 0},
        window_media={0: {"thickness": 10.0, "n_object": 1.333}},
        window_planes={0: {"plane_pt": [80.0, 0.0, 0.0], "plane_n": [1.0, 0.0, 0.0]}},
        wand_length=4.0,
        config=RefractiveBAConfig(verbosity=0),
    )
    optimizer._plane_d0 = {0: 400.0, 1: -200.0, 2: np.nan}
    messages = []
    optimizer.reporter.detail = messages.append

    limit = optimizer._get_adaptive_plane_d_limit(50.0, mode="joint")

    assert limit == pytest.approx(6.0)
    assert len(messages) == 1
    assert "mode=joint" in messages[0]
    assert "d0={w0:+400.000mm, w1:-200.000mm}" in messages[0]
    assert "min|d0|=200.000mm" in messages[0]
    assert "base=50.000mm" in messages[0]
    assert "adaptive=6.000mm" in messages[0]
    assert "limit=6.000mm" in messages[0]


def test_adaptive_plane_d_limit_keeps_non_joint_logging_concise():
    optimizer = RefractiveBAOptimizer(
        dataset={"obsA": {}, "obsB": {}, "frames": []},
        cam_params={0: _make_cam_params(center=[-120.0, 0.0, 0.0])},
        cams_cpp={},
        cam_to_window={0: 0},
        window_media={0: {"thickness": 10.0, "n_object": 1.333}},
        window_planes={0: {"plane_pt": [80.0, 0.0, 0.0], "plane_n": [1.0, 0.0, 0.0]}},
        wand_length=4.0,
        config=RefractiveBAConfig(verbosity=0),
    )
    optimizer._plane_d0 = {0: 400.0, 1: -200.0}
    messages = []
    optimizer.reporter.detail = messages.append

    limit = optimizer._get_adaptive_plane_d_limit(50.0, mode="loop_1_planes")

    assert limit == pytest.approx(6.0)
    assert messages == [
        "  [plane_d cap] mode=loop_1_planes base=50.000mm min|d0|=200.000mm -> limit=6.000mm"
    ]
def test_ba_coordinate_alignment_records_cumulative_affine_metadata(monkeypatch):
    optimizer = RefractiveBAOptimizer(
        dataset={"obsA": {}, "obsB": {}, "frames": []},
        cam_params={0: _make_cam_params(center=[-120.0, 0.0, 0.0])},
        cams_cpp={},
        cam_to_window={0: 0},
        window_media={0: {"thickness": 10.0, "n_object": 1.333}},
        window_planes={0: {"plane_pt": [10.0, 0.0, 0.0], "plane_n": [1.0, 0.0, 0.0]}},
        wand_length=4.0,
        config=RefractiveBAConfig(stage=3, verbosity=0),
    )
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    shift = np.array([2.0, 3.0, 4.0], dtype=np.float64)
    translation = rotation @ shift
    new_cam_params = {0: optimizer.cam_params[0].copy()}
    new_window_planes = {
        0: {
            "plane_pt": rotation @ (np.array([10.0, 0.0, 0.0], dtype=np.float64) + shift),
            "plane_n": rotation @ np.array([1.0, 0.0, 0.0], dtype=np.float64),
        }
    }
    transformed_points = {}

    monkeypatch.setattr(optimizer, "_collect_points_for_alignment", lambda: [])
    monkeypatch.setattr(
        "modules.camera_calibration.wand_calibration.refraction_calibration_BA.align_world_y_to_plane_intersection",
        lambda window_planes, cam_params, points_3d, align_mode: (
            new_cam_params,
            new_window_planes,
            points_3d,
            rotation,
            shift,
        ),
    )
    monkeypatch.setattr(optimizer, "sync_cpp_state", lambda **kwargs: None)
    monkeypatch.setattr(optimizer, "_sync_initial_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(optimizer, "_compute_physical_sigmas", lambda: None)
    monkeypatch.setattr(
        optimizer,
        "_transform_bundle_points",
        lambda R_world, t_shift: transformed_points.update({"rotation": R_world, "shift": t_shift}),
    )

    applied = optimizer._apply_coordinate_alignment(tag="unit-test", refresh_initial=True, align_mode="yz")

    assert applied is True
    np.testing.assert_allclose(optimizer.cumulative_alignment_rotation, rotation, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(optimizer.cumulative_alignment_translation, translation, atol=1e-12, rtol=0.0)
    assert len(optimizer.alignment_history) == 1
    assert optimizer.alignment_history[0]["tag"] == "unit-test"
    assert optimizer.alignment_history[0]["align_mode"] == "yz"
    np.testing.assert_allclose(transformed_points["rotation"], rotation, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(transformed_points["shift"], shift, atol=1e-12, rtol=0.0)

    restored = optimizer.reverse_cumulative_alignment_on_planes(optimizer.window_planes)
    np.testing.assert_allclose(restored[0]["plane_pt"], [10.0, 0.0, 0.0], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(restored[0]["plane_n"], [1.0, 0.0, 0.0], atol=1e-12, rtol=0.0)


def test_sync_initial_state_default_rederives_plane_reference_basis_from_current_cameras():
    optimizer = RefractiveBAOptimizer(
        dataset={"obsA": {}, "obsB": {}, "frames": []},
        cam_params={
            0: _make_cam_params(center=[-120.0, 0.0, 0.0]),
            1: _make_cam_params(center=[-140.0, 0.0, 0.0]),
        },
        cams_cpp={},
        cam_to_window={0: 0, 1: 0},
        window_media={0: {"thickness": 10.0, "n_object": 1.333}},
        window_planes={0: {"plane_pt": [10.0, 0.0, 0.0], "plane_n": [1.0, 0.0, 0.0]}},
        wand_length=4.0,
        config=RefractiveBAConfig(verbosity=0),
    )

    optimizer._plane_anchor = {0: np.array([999.0, 0.0, 0.0], dtype=np.float64)}
    optimizer._plane_d0 = {0: 999.0}
    optimizer.window_planes[0]["plane_pt"] = np.array([20.0, 0.0, 0.0], dtype=np.float64)
    optimizer.cam_params[0] = _make_cam_params(center=[-10.0, 0.0, 0.0])
    optimizer.cam_params[1] = _make_cam_params(center=[-30.0, 0.0, 0.0])

    optimizer._sync_initial_state()

    np.testing.assert_allclose(optimizer.initial_planes[0]["plane_pt"], [20.0, 0.0, 0.0], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(optimizer._plane_anchor[0], [-20.0, 0.0, 0.0], atol=1e-12, rtol=0.0)
    assert optimizer._plane_d0[0] == pytest.approx(40.0)


def test_refractive_ba_config_defaults_loop_rounds_to_sequence_strategy():
    config = RefractiveBAConfig()

    assert config.round_strategy["loop_planes"] == "sequence"
    assert config.round_strategy["loop_cams"] == "sequence"
    assert config.round_strategy["joint"] == "sequence"
    assert config.round_strategy["final_refined"] == "sequence"


def test_run_iterate_uses_live_ba_stage_metrics_for_verdict(tmp_path, monkeypatch):
    results_dir = tmp_path / "case_026_plane_debug"
    iterations_dir = _seed_iterate_inputs(results_dir)
    live_init_plane = {
        "plane_pt": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "plane_n": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "thick_mm": 2.0,
        "initialized": True,
    }
    bundle_plane = {
        "plane_pt": [0.0, 4.0, 0.0],
        "plane_n": [0.0, 1.0, 0.0],
    }
    gt_window = _make_gt_window(plane_pt_far=[12.0, 0.0, 0.0], plane_n=[1.0, 0.0, 0.0], thick_mm=2.0)
    ba_window_planes = {
        0: {
            "plane_pt": np.array(gt_window["plane_pt_far"], dtype=np.float64),
            "plane_n": np.array(gt_window["plane_n"], dtype=np.float64),
        }
    }

    class FakeBAOptimizer:
        def __init__(self, **kwargs):
            self.cam_params = kwargs["cam_params"]

        def optimize(self, skip_optimization=False, stage=None):
            return ba_window_planes, self.cam_params

    _patch_trace_inputs_for_live_init(
        monkeypatch,
        live_init_plane=live_init_plane,
        bundle_plane=bundle_plane,
        gt_window=gt_window,
    )
    monkeypatch.setattr(case_026_plane_debug_loop, "RefractiveBAOptimizer", FakeBAOptimizer, raising=False)
    monkeypatch.setattr(case_026_plane_debug_loop, "RefractiveBAConfig", lambda **kwargs: SimpleNamespace(**kwargs), raising=False)

    result = case_026_plane_debug_loop.run_iterate(results_dir=results_dir)

    response = json.loads((iterations_dir / "iteration_001_metis_response.json").read_text(encoding="utf-8"))
    assert result["verdict"] == "fixed"
    assert response["verdict"] == "fixed"
    assert response["post_fix_metrics"]["stage"] == "JOINT_BA"
    assert response["post_fix_metrics"]["max_angular_error_deg"] == pytest.approx(0.0)
    assert response["post_fix_metrics"]["max_point_distance_mm"] == pytest.approx(0.0)
    assert response["comparison"]["stage"] == "JOINT_BA"
    assert not (iterations_dir / "iteration_002_metis_request.json").exists()


def test_run_iterate_reports_fixed_and_skips_next_request_when_thresholds_pass(tmp_path):
    results_dir = tmp_path / "case_026_plane_debug"
    iterations_dir = _seed_iterate_inputs(results_dir)

    original_run_trace = case_026_plane_debug_loop.run_trace

    def fake_run_trace(**kwargs):
        root = Path(kwargs["results_dir"])
        trace_dir = root / "trace"
        trace_dir.mkdir(parents=True, exist_ok=True)
        gt_window = _make_gt_window(plane_pt_far=[10.0, 0.0, 0.0], plane_n=[1.0, 0.0, 0.0], thick_mm=2.0)
        gt_payload = {"artifact": "gt_planes", "interface_convention": "farthest", "windows": {"0": gt_window}}
        gt_path = root / "gt_planes.json"
        gt_path.write_text(json.dumps(gt_payload), encoding="utf-8")
        stage_path = trace_dir / "stage_INIT.json"
        stage_path.write_text(
            json.dumps(
                {
                    "stage": "INIT",
                    "interface_convention": "closest",
                    "windows": {
                        "0": {
                            "plane_pt": gt_window["plane_pt_closest"],
                            "plane_n": gt_window["plane_n"],
                            "thick_mm": gt_window["thick_mm"],
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        return {
            "mode": "trace",
            "ba_frame_budget": kwargs.get("frame_budget"),
            "ba_frame_selection": kwargs.get("frame_selection"),
            "gt_planes_path": str(gt_path),
            "artifacts": {"INIT": str(stage_path)},
        }

    case_026_plane_debug_loop.run_trace = fake_run_trace
    try:
        result = case_026_plane_debug_loop.run_iterate(results_dir=results_dir)
    finally:
        case_026_plane_debug_loop.run_trace = original_run_trace

    response = json.loads((iterations_dir / "iteration_001_metis_response.json").read_text(encoding="utf-8"))
    assert response["verdict"] == "fixed"
    assert response["termination_state"] == "fixed"
    assert response["post_fix_metrics"]["max_angular_error_deg"] == pytest.approx(0.0)
    assert response["post_fix_metrics"]["max_point_distance_mm"] == pytest.approx(0.0)
    assert not (iterations_dir / "iteration_002_metis_request.json").exists()
    assert result["verdict"] == "fixed"


def test_run_iterate_uses_live_init_trace_artifact_for_verdict(tmp_path, monkeypatch):
    results_dir = tmp_path / "case_026_plane_debug"
    iterations_dir = _seed_iterate_inputs(results_dir)
    live_init_plane = {
        "plane_pt": np.array([10.0, 0.0, 0.0], dtype=np.float64),
        "plane_n": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "thick_mm": 2.0,
        "initialized": True,
    }
    bundle_plane = {
        "plane_pt": [0.0, 4.0, 0.0],
        "plane_n": [0.0, 1.0, 0.0],
    }
    _patch_trace_inputs_for_live_init(monkeypatch, live_init_plane=live_init_plane, bundle_plane=bundle_plane)

    class FakeBAOptimizer:
        def __init__(self, **kwargs):
            self.cam_params = kwargs["cam_params"]

        def optimize(self, skip_optimization=False, stage=None):
            return {
                0: {
                    "plane_pt": np.array([12.0, 0.0, 0.0], dtype=np.float64),
                    "plane_n": np.array([1.0, 0.0, 0.0], dtype=np.float64),
                }
            }, self.cam_params

    monkeypatch.setattr(case_026_plane_debug_loop, "RefractiveBAOptimizer", FakeBAOptimizer, raising=False)
    monkeypatch.setattr(case_026_plane_debug_loop, "RefractiveBAConfig", lambda **kwargs: SimpleNamespace(**kwargs), raising=False)

    result = case_026_plane_debug_loop.run_iterate(results_dir=results_dir)

    init_stage = _read_json(Path(result["trace_manifest"]["artifacts"]["INIT"]))
    response = json.loads((iterations_dir / "iteration_001_metis_response.json").read_text(encoding="utf-8"))
    assert init_stage["source"] == "live_init_window_planes_from_cameras"
    assert response["verdict"] == "fixed"
    assert response["post_fix_metrics"]["max_angular_error_deg"] == pytest.approx(0.0)
    assert response["post_fix_metrics"]["max_point_distance_mm"] == pytest.approx(0.0)
    assert not (iterations_dir / "iteration_002_metis_request.json").exists()


def test_run_iterate_writes_response_for_latest_request_iteration(tmp_path):
    results_dir = tmp_path / "case_026_plane_debug"
    iterations_dir = _seed_iterate_inputs(results_dir)
    latest_request = {
        "artifact": "metis_request",
        "iteration": 2,
        "task": "Analyze refreshed trace artifacts and propose the next bounded fix",
        "selected_branch": "A",
        "earliest_divergence_stage": "INIT",
        "target_primary_metric_deg": 2.0,
        "target_secondary_metric_mm": 1.0,
        "stable_budget": 25,
    }
    _write_json(iterations_dir / "iteration_002_metis_request.json", latest_request)

    original_run_trace = case_026_plane_debug_loop.run_trace

    def fake_run_trace(**kwargs):
        root = Path(kwargs["results_dir"])
        trace_dir = root / "trace"
        trace_dir.mkdir(parents=True, exist_ok=True)
        gt_window = _make_gt_window(plane_pt_far=[10.0, 0.0, 0.0], plane_n=[1.0, 0.0, 0.0], thick_mm=2.0)
        gt_path = root / "gt_planes.json"
        _write_json(gt_path, {"artifact": "gt_planes", "interface_convention": "farthest", "windows": {"0": gt_window}})
        stage_path = trace_dir / "stage_INIT.json"
        _write_json(
            stage_path,
            {
                "stage": "INIT",
                "interface_convention": "closest",
                "source": "live_init_window_planes_from_cameras",
                "windows": {"0": {"plane_pt": [0.0, 0.0, 0.0], "plane_n": [0.0, 1.0, 0.0], "thick_mm": 2.0}},
            },
        )
        return {
            "mode": "trace",
            "ba_frame_budget": kwargs.get("frame_budget"),
            "ba_frame_selection": kwargs.get("frame_selection"),
            "gt_planes_path": str(gt_path),
            "artifacts": {"INIT": str(stage_path)},
        }

    case_026_plane_debug_loop.run_trace = fake_run_trace
    try:
        result = case_026_plane_debug_loop.run_iterate(results_dir=results_dir)
    finally:
        case_026_plane_debug_loop.run_trace = original_run_trace

    response_path = iterations_dir / "iteration_002_metis_response.json"
    next_request_path = iterations_dir / "iteration_003_metis_request.json"
    assert response_path.exists()
    assert next_request_path.exists()
    response = _read_json(response_path)
    assert response["iteration"] == 2
    assert result["iteration"] == 2
    assert result["metis_response_path"] == str(response_path)
    assert result["metis_request_path"] == str(next_request_path)


def test_joint_round3_uses_tightened_camera_bounds():
    """
    Regression: Verify JOINT Round 3 uses tighter camera extrinsic bounds:
    - limit_rvec: 5° (not 20°)
    - limit_tvec: 15 mm (not 50 mm)
    while keeping plane bounds unchanged:
    - limit_plane_ang: 10°
    - limit_plane_d: 50 mm (requested, before adaptive cap)
    """
    from modules.camera_calibration.wand_calibration.refraction_calibration_BA import (
        RefractiveBAOptimizer,
        RefractiveBAConfig,
    )
    from unittest.mock import MagicMock, patch
    
    # Minimal test setup
    config = RefractiveBAConfig(stage=3, skip_optimization=False, verbosity=0)
    optimizer = RefractiveBAOptimizer(
        dataset={"obsA": {}, "obsB": {}, "frames": []},
        cam_params={0: np.array([0, 0, 0, 0, 0, 0, 1000, 512, 512, 0, 0], dtype=np.float64)},
        cams_cpp={},
        cam_to_window={0: 0},
        window_media={0: {"thickness": 10.0, "n_object": 1.49}},
        window_planes={0: {"plane_pt": np.array([0, 0, 0]), "plane_n": np.array([0, 0, 1])}},
        wand_length=4.0,
        config=config,
    )
    
    # Capture _optimize_generic calls
    captured_calls = []
    original_optimize_generic = optimizer._optimize_generic
    
    def mock_optimize_generic(*args, **kwargs):
        captured_calls.append(kwargs.copy())
        # Return minimal result structure
        from types import SimpleNamespace
        return SimpleNamespace(x=np.array([]), active_mask=np.array([])), []
    
    with patch.object(optimizer, '_optimize_generic', side_effect=mock_optimize_generic):
        with patch.object(optimizer, '_get_chunk_schedule_for_mode', return_value=None):
            with patch.object(optimizer, '_print_plane_diagnostics'):
                with patch.object(optimizer, 'evaluate_residuals', return_value=(np.array([]), 0, 0, 0, 0, 0, 0)):
                    with patch.object(optimizer, 'print_diagnostics'):
                        optimizer.optimize(skip_optimization=False, stage=3)
    
    # Find the JOINT call
    joint_call = None
    for call in captured_calls:
        if call.get('mode') == 'joint':
            joint_call = call
            break
    
    assert joint_call is not None, "No 'joint' mode call found"
    
    # Verify tightened camera bounds
    assert joint_call['limit_rot_rad'] == pytest.approx(np.radians(5.0)), \
        f"Expected limit_rvec=5°, got {np.degrees(joint_call['limit_rot_rad']):.1f}°"
    assert joint_call['limit_trans_mm'] == 15.0, \
        f"Expected limit_tvec=15mm, got {joint_call['limit_trans_mm']}mm"
    
    # Verify plane bounds unchanged
    assert joint_call['limit_plane_angle_rad'] == pytest.approx(np.radians(10.0)), \
        f"Expected limit_plane_ang=10°, got {np.degrees(joint_call['limit_plane_angle_rad']):.1f}°"
    assert joint_call['limit_plane_d_mm'] == 50.0, \
        f"Expected limit_plane_d=50mm, got {joint_call['limit_plane_d_mm']}mm"
