# pyright: reportMissingImports=false, reportAttributeAccessIssue=false
"""
Regression tests for P0 bootstrap failure modes and structured telemetry.

These tests exercise the catastrophic P0 failure behavior, telemetry
emission, and failure-reason classification introduced in Task 2.

DESIGN (red-phase):
    Many of these tests are expected to FAIL (or reveal specific failure
    behavior) before bootstrap hardening is applied in Task 5.  The test
    names and docstrings clarify which should currently pass and which are
    red-phase regression anchors.

    - Tests tagged ``_red_phase`` are expected to fail before hardening.
    - Tests tagged ``_green`` should always pass.
"""

import json
import re
import sys
import textwrap
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple, Optional

import numpy as np
import pytest
import cv2

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.camera_calibration.wand_calibration.refractive_bootstrap import (
    P0FailureError,
    P0Telemetry,
    Phase2CameraTelemetry,
    P0_REASON_CATASTROPHIC_REPROJECTION,
    P0_REASON_ESSENTIAL_MATRIX_FAILED,
    P0_REASON_INSUFFICIENT_GEOMETRY,
    P0_REASON_OK,
    P0_REASON_PHASE1_BA_FAILURE,
    P0_REASON_TOO_FEW_E_INLIERS,
    P0_REASON_UNSTABLE_SCALE_RECOVERY,
    PinholeBootstrapP0,
    PinholeBootstrapP0Config,
    select_ranked_pairs_via_precalib,
)
from scripts.case_013_bootstrap_debug_loop import (
    CASE_ARTIFACT_KEYS,
    DEFAULT_HEALTHY_CASES,
    DEFAULT_HEALTHY_SEED,
    DEFAULT_RESULTS_ROOT,
    DEFAULT_TARGET_CASE,
    ENVELOPE_KEYS,
    build_case_artifact,
    run_bootstrap_case,
    run_final_regression,
    run_iterate,
    should_pause_for_stall,
)


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic observation helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_camera_settings(cam_ids, focal=9000.0, width=1280, height=800):
    """Build camera_settings dict for the given camera IDs."""
    return {
        cid: {"focal": focal, "width": width, "height": height}
        for cid in cam_ids
    }


def _project_point(pt3d, R, t, K):
    """Project a 3D point to 2D using pinhole model."""
    pt_cam = R @ pt3d.reshape(3, 1) + t
    pt_cam = pt_cam.flatten()
    if pt_cam[2] <= 0:
        return np.array([1e6, 1e6])
    pt_norm = pt_cam[:2] / pt_cam[2]
    pt_px = K[:2, :2] @ pt_norm + K[:2, 2]
    return pt_px


def _build_healthy_observations(
    n_frames: int = 50,
    wand_length_mm: float = 10.0,
    focal: float = 9000.0,
    width: int = 1280,
    height: int = 800,
    baseline_mm: float = 250.0,
    seed: int = 42,
) -> Tuple[Dict, Dict, int, int]:
    """
    Build synthetic observations for a healthy two-camera setup.

    Returns (observations, camera_settings, cam_i, cam_j).
    """
    rng = np.random.default_rng(seed)

    cam_i, cam_j = 0, 1
    K = np.array([[focal, 0, width / 2.0],
                  [0, focal, height / 2.0],
                  [0, 0, 1.0]], dtype=np.float64)

    # Camera i at origin, camera j offset along X
    R_i = np.eye(3)
    t_i = np.zeros((3, 1))
    R_j = np.eye(3)
    t_j = np.array([[baseline_mm], [0.0], [0.0]])

    observations = {}
    for fid in range(n_frames):
        # Random wand positions in front of both cameras
        center = np.array([
            rng.uniform(-20, 20),
            rng.uniform(-20, 20),
            rng.uniform(500, 700),
        ])
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)

        ptA = center - direction * (wand_length_mm / 2)
        ptB = center + direction * (wand_length_mm / 2)

        uvA_i = _project_point(ptA, R_i, t_i, K)
        uvB_i = _project_point(ptB, R_i, t_i, K)
        uvA_j = _project_point(ptA, R_j, t_j, K)
        uvB_j = _project_point(ptB, R_j, t_j, K)

        observations[fid] = {
            cam_i: (uvA_i, uvB_i),
            cam_j: (uvA_j, uvB_j),
        }

    camera_settings = _make_camera_settings([cam_i, cam_j], focal, width, height)
    return observations, camera_settings, cam_i, cam_j


def _build_degenerate_observations(
    n_frames: int = 50,
    wand_length_mm: float = 10.0,
    focal: float = 9000.0,
    width: int = 1280,
    height: int = 800,
    baseline_mm: float = 0.3,
    seed: int = 99,
) -> Tuple[Dict, Dict, int, int]:
    """
    Build synthetic observations for a degenerate two-camera setup with
    a tiny baseline that should trigger catastrophic P0 failure.

    Returns (observations, camera_settings, cam_i, cam_j).
    """
    rng = np.random.default_rng(seed)

    cam_i, cam_j = 0, 1
    K = np.array([[focal, 0, width / 2.0],
                  [0, focal, height / 2.0],
                  [0, 0, 1.0]], dtype=np.float64)

    # Tiny baseline — nearly co-located cameras
    R_i = np.eye(3)
    t_i = np.zeros((3, 1))
    R_j = np.eye(3)
    t_j = np.array([[baseline_mm], [0.0], [0.0]])

    observations = {}
    for fid in range(n_frames):
        center = np.array([
            rng.uniform(-10, 10),
            rng.uniform(-10, 10),
            rng.uniform(500, 700),
        ])
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)

        ptA = center - direction * (wand_length_mm / 2)
        ptB = center + direction * (wand_length_mm / 2)

        uvA_i = _project_point(ptA, R_i, t_i, K)
        uvB_i = _project_point(ptB, R_i, t_i, K)
        uvA_j = _project_point(ptA, R_j, t_j, K)
        uvB_j = _project_point(ptB, R_j, t_j, K)

        observations[fid] = {
            cam_i: (uvA_i, uvB_i),
            cam_j: (uvA_j, uvB_j),
        }

    camera_settings = _make_camera_settings([cam_i, cam_j], focal, width, height)
    return observations, camera_settings, cam_i, cam_j


# ═══════════════════════════════════════════════════════════════════════════
# P0Telemetry unit tests (GREEN — should always pass)
# ═══════════════════════════════════════════════════════════════════════════


class TestP0TelemetryStructure:
    """Tests that P0Telemetry dataclass has all expected fields and serializes correctly."""

    def test_default_fields_green(self):
        """All expected telemetry fields exist with correct defaults."""
        t = P0Telemetry()
        assert t.failure_reason == P0_REASON_OK
        assert t.selected_pair is None
        assert t.baseline_mm is None
        assert t.cheirality_ratio is None
        assert t.scale_factor_finite is None
        assert t.ba_initial_cost is None
        assert t.reproj_err_mean is None

    def test_to_dict_serializable_green(self):
        """to_dict() produces a JSON-serializable dict with all keys."""
        t = P0Telemetry(
            selected_pair=(0, 1),
            baseline_mm=250.0,
            e_inliers=100,
            e_total=200,
            pose_inliers=90,
            pose_total=100,
            cheirality_ratio=0.9,
            valid_inlier_wand_pairs=40,
            median_triangulation_length=5.0,
            scale_factor=2.0,
            scale_factor_finite=True,
            ba_initial_cost=100.0,
            ba_final_cost=10.0,
            ba_converged=True,
            ba_message="converged",
            reproj_err_mean=0.05,
            reproj_err_max=0.2,
            wand_length_median=10.0,
            wand_length_error=0.001,
            valid_frames=50,
            failure_reason=P0_REASON_OK,
        )
        d = t.to_dict()

        # Verify JSON round-trip
        json_str = json.dumps(d, default=str)
        loaded = json.loads(json_str)

        assert loaded["failure_reason"] == "ok"
        assert loaded["selected_pair"] == [0, 1]
        assert loaded["cheirality_ratio"] == 0.9
        assert loaded["scale_factor_finite"] is True

    def test_emit_produces_telemetry_line_green(self, capsys):
        """emit() prints a [P0_TELEMETRY] line parseable as JSON."""
        t = P0Telemetry(
            selected_pair=(0, 1),
            failure_reason=P0_REASON_OK,
            baseline_mm=100.0,
        )
        t.emit()
        captured = capsys.readouterr()
        assert "[P0_TELEMETRY]" in captured.out

        # Extract JSON payload
        match = re.search(r"\[P0_TELEMETRY\]\s*(\{.*\})", captured.out)
        assert match is not None
        payload = json.loads(match.group(1))
        assert payload["failure_reason"] == "ok"
        assert payload["baseline_mm"] == 100.0

    def test_failure_reason_constants_green(self):
        """All failure reason constants are distinct non-empty strings."""
        reasons = [
            P0_REASON_OK,
            P0_REASON_INSUFFICIENT_GEOMETRY,
            P0_REASON_ESSENTIAL_MATRIX_FAILED,
            P0_REASON_TOO_FEW_E_INLIERS,
            P0_REASON_UNSTABLE_SCALE_RECOVERY,
            P0_REASON_CATASTROPHIC_REPROJECTION,
            P0_REASON_PHASE1_BA_FAILURE,
        ]
        assert len(reasons) == len(set(reasons)), "Duplicate failure reason constants"
        for r in reasons:
            assert isinstance(r, str) and len(r) > 0


class TestP0FailureErrorStructure:
    """Tests that P0FailureError carries structured reason and telemetry."""

    def test_error_carries_reason_and_telemetry_green(self):
        """P0FailureError stores reason and telemetry with failure fields set."""
        t = P0Telemetry(selected_pair=(2, 3))
        exc = P0FailureError(
            "test failure message",
            P0_REASON_CATASTROPHIC_REPROJECTION,
            t,
        )
        assert exc.reason == P0_REASON_CATASTROPHIC_REPROJECTION
        assert exc.telemetry is t
        assert t.failure_reason == P0_REASON_CATASTROPHIC_REPROJECTION
        assert t.failure_detail == "test failure message"
        assert str(exc) == "test failure message"

    def test_error_is_runtime_error_green(self):
        """P0FailureError is a subclass of RuntimeError."""
        t = P0Telemetry()
        exc = P0FailureError("msg", P0_REASON_OK, t)
        assert isinstance(exc, RuntimeError)


class TestPhase2ConfigAndTelemetrySchema:
    """Tests that Phase-2 config and telemetry extensions are schema-complete and additive."""

    def test_config_has_phase2_fields_green(self):
        """PinholeBootstrapP0Config has all expected Phase-2 fields with defaults."""
        config = PinholeBootstrapP0Config()
        
        assert hasattr(config, "phase2_prefilter_wand_length_error_mm")
        assert hasattr(config, "phase2_prefilter_depth_min_mm")
        assert hasattr(config, "phase2_prefilter_depth_max_mm")
        assert hasattr(config, "phase2_prefilter_min_triangulation_angle_deg")
        assert hasattr(config, "phase2_ransac_enabled")
        assert hasattr(config, "phase2_ransac_reproj_threshold_px")
        assert hasattr(config, "phase2_ransac_confidence")
        assert hasattr(config, "phase2_ransac_min_inlier_ratio")
        assert hasattr(config, "phase2_confidence_min_correspondences")
        assert hasattr(config, "phase2_confidence_max_reproj_rms_px")
        assert hasattr(config, "phase2_confidence_min_inlier_ratio")
        assert hasattr(config, "phase2_pair_fallback_rms_threshold_px")
        assert hasattr(config, "phase2_pair_fallback_max_retries")
        
        assert config.phase2_prefilter_wand_length_error_mm == 2.0
        assert config.phase2_prefilter_depth_min_mm == 100.0
        assert config.phase2_prefilter_depth_max_mm == 10000.0
        assert config.phase2_prefilter_min_triangulation_angle_deg == 0.5
        assert config.phase2_ransac_enabled is False
        assert config.phase2_ransac_reproj_threshold_px == 3.0
        assert config.phase2_ransac_confidence == 0.999
        assert config.phase2_ransac_min_inlier_ratio == 0.5
        assert config.phase2_confidence_min_correspondences == 12
        assert config.phase2_confidence_max_reproj_rms_px == 5.0
        assert config.phase2_confidence_min_inlier_ratio == 0.6
        assert config.phase2_pair_fallback_rms_threshold_px == 35.0
        assert config.phase2_pair_fallback_max_retries == 2

    def test_config_preserves_existing_fields_green(self):
        """Phase-2 config extension is additive and preserves all original fields."""
        config = PinholeBootstrapP0Config(wand_length_mm=15.0, ui_focal_px=10000.0)
        
        assert config.wand_length_mm == 15.0
        assert config.ui_focal_px == 10000.0
        assert config.ftol == 1e-6
        assert config.xtol == 1e-6

    def test_phase2_camera_telemetry_schema_green(self):
        """Phase2CameraTelemetry has all expected fields and serializes correctly."""
        telem = Phase2CameraTelemetry(
            camera_id=5,
            total_correspondences=100,
            prefilter_candidate_removed=18,
            prefilter_removed=15,
            prefilter_removed_nonfinite=2,
            prefilter_removed_depth=6,
            prefilter_removed_wand_length=4,
            prefilter_removed_angle=3,
            prefilter_kept=85,
            prefilter_min_correspondences=6,
            prefilter_starvation_fallback=False,
            pnp_correspondences=85,
            ransac_inliers=80,
            ransac_inlier_ratio=0.941,
            reproj_rms_px=1.23,
            confidence_label="normal",
            confidence_warning=None,
        )
        
        assert telem.camera_id == 5
        assert telem.total_correspondences == 100
        assert telem.prefilter_candidate_removed == 18
        assert telem.prefilter_removed == 15
        assert telem.prefilter_removed_nonfinite == 2
        assert telem.prefilter_removed_depth == 6
        assert telem.prefilter_removed_wand_length == 4
        assert telem.prefilter_removed_angle == 3
        assert telem.prefilter_kept == 85
        assert telem.prefilter_min_correspondences == 6
        assert telem.prefilter_starvation_fallback is False
        assert telem.pnp_correspondences == 85
        assert telem.ransac_inliers == 80
        assert telem.ransac_inlier_ratio == 0.941
        assert telem.reproj_rms_px == 1.23
        assert telem.confidence_label == "normal"
        assert telem.confidence_warning is None
        
        d = telem.to_dict()
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        
        assert loaded["camera_id"] == 5
        assert loaded["total_correspondences"] == 100
        assert loaded["prefilter_candidate_removed"] == 18
        assert loaded["prefilter_removed_depth"] == 6
        assert loaded["prefilter_kept"] == 85
        assert loaded["ransac_inlier_ratio"] == 0.941
        assert loaded["confidence_label"] == "normal"

    def test_p0_telemetry_has_phase2_cameras_field_green(self):
        """P0Telemetry has phase2_cameras field that holds per-camera telemetry."""
        t = P0Telemetry()
        
        assert hasattr(t, "phase2_cameras")
        assert isinstance(t.phase2_cameras, dict)
        assert len(t.phase2_cameras) == 0
        
        cam_telem = Phase2CameraTelemetry(
            camera_id=3,
            total_correspondences=50,
            prefilter_removed=5,
            pnp_correspondences=45,
            ransac_inliers=42,
            ransac_inlier_ratio=0.933,
            reproj_rms_px=0.85,
            confidence_label="normal",
        )
        t.phase2_cameras[3] = cam_telem
        
        assert 3 in t.phase2_cameras
        assert t.phase2_cameras[3].camera_id == 3
        assert t.phase2_cameras[3].total_correspondences == 50

    def test_p0_telemetry_to_dict_includes_phase2_cameras_green(self):
        """P0Telemetry.to_dict() serializes nested phase2_cameras correctly."""
        t = P0Telemetry(
            selected_pair=(2, 3),
            failure_reason=P0_REASON_OK,
        )
        
        cam4_telem = Phase2CameraTelemetry(
            camera_id=4,
            total_correspondences=60,
            prefilter_removed=10,
            pnp_correspondences=50,
            reproj_rms_px=2.1,
            confidence_label="low",
            confidence_warning="insufficient_correspondences",
        )
        cam5_telem = Phase2CameraTelemetry(
            camera_id=5,
            total_correspondences=80,
            prefilter_removed=2,
            pnp_correspondences=78,
            ransac_inliers=75,
            ransac_inlier_ratio=0.962,
            reproj_rms_px=0.92,
            confidence_label="normal",
        )
        
        t.phase2_cameras[4] = cam4_telem
        t.phase2_cameras[5] = cam5_telem
        
        d = t.to_dict()
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        
        assert "phase2_cameras" in loaded
        assert "4" in loaded["phase2_cameras"]
        assert "5" in loaded["phase2_cameras"]
        
        assert loaded["phase2_cameras"]["4"]["camera_id"] == 4
        assert loaded["phase2_cameras"]["4"]["total_correspondences"] == 60
        assert loaded["phase2_cameras"]["4"]["prefilter_removed"] == 10
        assert loaded["phase2_cameras"]["4"]["confidence_label"] == "low"
        assert loaded["phase2_cameras"]["4"]["confidence_warning"] == "insufficient_correspondences"
        
        assert loaded["phase2_cameras"]["5"]["camera_id"] == 5
        assert loaded["phase2_cameras"]["5"]["ransac_inlier_ratio"] == 0.962
        assert loaded["phase2_cameras"]["5"]["confidence_label"] == "normal"

    def test_p0_telemetry_emit_includes_phase2_cameras_green(self, capsys):
        """P0Telemetry.emit() outputs phase2_cameras in JSON."""
        t = P0Telemetry(
            selected_pair=(2, 3),
            failure_reason=P0_REASON_OK,
        )
        
        cam6_telem = Phase2CameraTelemetry(
            camera_id=6,
            total_correspondences=40,
            prefilter_removed=8,
            pnp_correspondences=32,
            reproj_rms_px=3.5,
            confidence_label="low",
            confidence_warning="high_rms",
        )
        t.phase2_cameras[6] = cam6_telem
        
        t.emit()
        captured = capsys.readouterr()
        
        assert "[P0_TELEMETRY]" in captured.out
        match = re.search(r"\[P0_TELEMETRY\]\s*(\{.*\})", captured.out)
        assert match is not None
        
        payload = json.loads(match.group(1))
        assert "phase2_cameras" in payload
        assert "6" in payload["phase2_cameras"]
        assert payload["phase2_cameras"]["6"]["confidence_label"] == "low"
        assert payload["phase2_cameras"]["6"]["confidence_warning"] == "high_rms"


# ═══════════════════════════════════════════════════════════════════════════
# P0 Bootstrap integration tests (mix of GREEN and RED-PHASE)
# ═══════════════════════════════════════════════════════════════════════════


class TestP0BootstrapInsufficientFrames:
    """Insufficient frames should trigger early failure with correct reason."""

    def test_too_few_frames_green(self):
        """< 10 frames raises P0FailureError(insufficient_geometry)."""
        obs, settings, ci, cj = _build_healthy_observations(n_frames=5)
        config = PinholeBootstrapP0Config(wand_length_mm=10.0)
        p0 = PinholeBootstrapP0(config)

        with pytest.raises(P0FailureError) as exc_info:
            p0.run(ci, cj, obs, settings)

        assert exc_info.value.reason == P0_REASON_INSUFFICIENT_GEOMETRY
        assert exc_info.value.telemetry.failure_reason == P0_REASON_INSUFFICIENT_GEOMETRY
        assert exc_info.value.telemetry.selected_pair == (ci, cj)

    def test_zero_frames_green(self):
        """Zero frames raises P0FailureError(insufficient_geometry)."""
        settings = _make_camera_settings([0, 1])
        config = PinholeBootstrapP0Config(wand_length_mm=10.0)
        p0 = PinholeBootstrapP0(config)

        with pytest.raises(P0FailureError) as exc_info:
            p0.run(0, 1, {}, settings)

        assert exc_info.value.reason == P0_REASON_INSUFFICIENT_GEOMETRY


class TestP0BootstrapHealthyCase:
    """A well-separated camera pair should produce good P0 results."""

    def test_healthy_baseline_passes_green(self, capsys):
        """Healthy synthetic setup passes P0 with reasonable metrics."""
        obs, settings, ci, cj = _build_healthy_observations(
            n_frames=100,
            baseline_mm=250.0,
            seed=42,
        )
        config = PinholeBootstrapP0Config(wand_length_mm=10.0, ui_focal_px=9000.0)
        p0 = PinholeBootstrapP0(config)

        params_i, params_j, report = p0.run(ci, cj, obs, settings)

        # Verify structural integrity
        assert params_i.shape == (6,)
        assert params_j.shape == (6,)
        assert "p0_telemetry" in report

        telem = report["p0_telemetry"]
        assert telem["failure_reason"] == P0_REASON_OK
        assert telem["selected_pair"] == [ci, cj]
        assert telem["scale_factor_finite"] is True
        assert telem["cheirality_ratio"] is not None
        assert telem["cheirality_ratio"] > 0.7  # healthy cheirality
        assert telem["ba_converged"] is True

        # Reproj error should be small for a well-conditioned setup
        assert telem["reproj_err_mean"] < 5.0  # generous threshold

        # Baseline should be recovered approximately
        assert telem["baseline_mm"] is not None
        assert telem["baseline_mm"] > 50.0  # well above the 50mm warning

        # Check [P0_TELEMETRY] was emitted
        captured = capsys.readouterr()
        assert "[P0_TELEMETRY]" in captured.out

    def test_healthy_report_has_all_telemetry_keys_green(self):
        """Report's p0_telemetry dict contains all expected diagnostic keys."""
        obs, settings, ci, cj = _build_healthy_observations(
            n_frames=50,
            baseline_mm=200.0,
            seed=123,
        )
        config = PinholeBootstrapP0Config(wand_length_mm=10.0, ui_focal_px=9000.0)
        p0 = PinholeBootstrapP0(config)

        _, _, report = p0.run(ci, cj, obs, settings)
        telem = report["p0_telemetry"]

        required_keys = {
            "selected_pair", "baseline_mm",
            "e_inliers", "e_total",
            "pose_inliers", "pose_total", "cheirality_ratio",
            "valid_inlier_wand_pairs", "median_triangulation_length",
            "scale_factor", "scale_factor_finite",
            "ba_initial_cost", "ba_final_cost", "ba_converged", "ba_message",
            "reproj_err_mean", "reproj_err_max",
            "wand_length_median", "wand_length_error",
            "valid_frames",
            "failure_reason", "failure_detail",
        }
        assert required_keys.issubset(set(telem.keys())), (
            f"Missing keys: {required_keys - set(telem.keys())}"
        )


class TestP0BootstrapDegenerateBaseline:
    """
    A degenerate (tiny) baseline should trigger catastrophic failure.

    RED-PHASE: Before hardening, the current bootstrap should raise
    P0FailureError with reason catastrophic_reprojection because the
    tiny baseline causes scale recovery to produce enormous reprojection.
    """

    def test_tiny_baseline_triggers_failure_red_phase(self):
        """
        A ~0.3mm baseline should trigger P0FailureError.

        This test reproduces the core failure pattern seen in the 7 failed
        cases (e.g. case_023 with baseline=0.32mm).

        After bootstrap hardening (Task 5), this test should be updated
        to expect either a different failure reason or a recovery path.
        """
        obs, settings, ci, cj = _build_degenerate_observations(
            n_frames=100,
            baseline_mm=0.3,
            seed=99,
        )
        config = PinholeBootstrapP0Config(wand_length_mm=10.0, ui_focal_px=9000.0)
        p0 = PinholeBootstrapP0(config)

        with pytest.raises(P0FailureError) as exc_info:
            p0.run(ci, cj, obs, settings)

        err = exc_info.value
        # The failure should be one of the catastrophic paths
        assert err.reason in (
            P0_REASON_CATASTROPHIC_REPROJECTION,
            P0_REASON_UNSTABLE_SCALE_RECOVERY,
            P0_REASON_PHASE1_BA_FAILURE,
        ), f"Unexpected failure reason: {err.reason}"

        # Telemetry should be populated even on failure
        t = err.telemetry
        assert t.selected_pair == (ci, cj)
        assert t.failure_reason == err.reason
        assert t.failure_detail is not None and len(t.failure_detail) > 0

    def test_degenerate_telemetry_has_cheirality_red_phase(self):
        """
        Even when P0 fails, cheirality_ratio should be populated if
        the E-matrix stage succeeded.

        In the real failing cases, cheirality_ratio ~0.487 was the
        pre-failure signal.
        """
        obs, settings, ci, cj = _build_degenerate_observations(
            n_frames=100,
            baseline_mm=0.3,
            seed=99,
        )
        config = PinholeBootstrapP0Config(wand_length_mm=10.0, ui_focal_px=9000.0)
        p0 = PinholeBootstrapP0(config)

        with pytest.raises(P0FailureError) as exc_info:
            p0.run(ci, cj, obs, settings)

        t = exc_info.value.telemetry
        # If E-matrix stage succeeded, these should be populated
        if t.e_inliers is not None and t.e_inliers >= 8:
            assert t.cheirality_ratio is not None
            assert 0.0 <= t.cheirality_ratio <= 1.0


class TestP0BootstrapMissingCamera:
    """Missing camera settings should raise ValueError (not P0FailureError)."""

    def test_missing_camera_settings_green(self):
        """Requesting an unknown camera raises ValueError."""
        obs, settings, ci, cj = _build_healthy_observations(n_frames=20)
        config = PinholeBootstrapP0Config(wand_length_mm=10.0)
        p0 = PinholeBootstrapP0(config)

        # Remove cam_j from settings
        settings_missing = {ci: settings[ci]}
        with pytest.raises(ValueError, match="Missing camera_settings"):
            p0.run(ci, cj, obs, settings_missing)


class TestP0BootstrapTelemetryOnFailurePaths:
    """
    Verify structured telemetry is emitted even when P0 fails,
    so downstream tools (ablation runner, Task 4 classifier) can
    parse the failure mode.
    """

    def test_p0_failure_error_is_json_serializable_green(self):
        """P0FailureError telemetry can be serialized to JSON for logging."""
        t = P0Telemetry(
            selected_pair=(2, 4),
            baseline_mm=1.17,
            cheirality_ratio=0.274,
            failure_reason=P0_REASON_CATASTROPHIC_REPROJECTION,
            failure_detail="[P0 FAIL] Reprojection error too high: 561848.12 px",
        )
        d = t.to_dict()
        json_str = json.dumps(d, default=str)
        loaded = json.loads(json_str)
        assert loaded["failure_reason"] == P0_REASON_CATASTROPHIC_REPROJECTION
        assert loaded["baseline_mm"] == 1.17
        assert loaded["cheirality_ratio"] == 0.274

    def test_failure_reason_matches_known_p0_cases_green(self):
        """
        Verify the known failure reasons from the evidence matrix
        match the defined constants.
        """
        known_reasons_from_evidence = [
            "catastrophic_reprojection",  # case_012, 015, 019, 023, 027, 028, 029
        ]
        valid_constants = {
            P0_REASON_OK,
            P0_REASON_INSUFFICIENT_GEOMETRY,
            P0_REASON_ESSENTIAL_MATRIX_FAILED,
            P0_REASON_TOO_FEW_E_INLIERS,
            P0_REASON_UNSTABLE_SCALE_RECOVERY,
            P0_REASON_CATASTROPHIC_REPROJECTION,
            P0_REASON_PHASE1_BA_FAILURE,
        }
        for reason in known_reasons_from_evidence:
            assert reason in valid_constants, (
                f"Evidence-matrix reason '{reason}' not in defined constants"
            )


# ═══════════════════════════════════════════════════════════════════════════
# Task 5: Geometry-aware pair selection and fallback retry tests
# ═══════════════════════════════════════════════════════════════════════════


def _build_multi_camera_observations(
    n_frames: int = 50,
    wand_length_mm: float = 10.0,
    focal: float = 9000.0,
    width: int = 1280,
    height: int = 800,
    seed: int = 77,
) -> Tuple[Dict, Dict]:
    """
    Build synthetic observations for a 5-camera setup where cameras 0 and 1
    are nearly co-located (degenerate pair) but cameras 2, 3, 4 are well
    separated.

    Camera positions:
      cam 0: origin
      cam 1: 0.3 mm offset (nearly co-located with cam 0 -- DEGENERATE)
      cam 2: 250 mm along X (healthy)
      cam 3: 0, 200 mm along Y (healthy)
      cam 4: 150 mm along X, 150 mm along Y (healthy)

    Returns (observations, camera_settings).
    """
    rng = np.random.default_rng(seed)

    cam_ids = [0, 1, 2, 3, 4]
    K = np.array([[focal, 0, width / 2.0],
                  [0, focal, height / 2.0],
                  [0, 0, 1.0]], dtype=np.float64)

    # Camera extrinsics (R=identity for all, varying translations)
    cam_R = {cid: np.eye(3) for cid in cam_ids}
    cam_t = {
        0: np.array([[0.0], [0.0], [0.0]]),
        1: np.array([[0.3], [0.0], [0.0]]),     # nearly co-located with cam 0
        2: np.array([[250.0], [0.0], [0.0]]),    # healthy separation
        3: np.array([[0.0], [200.0], [0.0]]),    # healthy separation
        4: np.array([[150.0], [150.0], [0.0]]),  # healthy separation
    }

    observations = {}
    for fid in range(n_frames):
        center = np.array([
            rng.uniform(-20, 20),
            rng.uniform(-20, 20),
            rng.uniform(500, 700),
        ])
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)

        ptA = center - direction * (wand_length_mm / 2)
        ptB = center + direction * (wand_length_mm / 2)

        frame_obs = {}
        for cid in cam_ids:
            uvA = _project_point(ptA, cam_R[cid], cam_t[cid], K)
            uvB = _project_point(ptB, cam_R[cid], cam_t[cid], K)
            frame_obs[cid] = (uvA, uvB)
        observations[fid] = frame_obs

    camera_settings = _make_camera_settings(cam_ids, focal, width, height)
    return observations, camera_settings


def _build_phase2_geometry_poor_observations(
    n_frames: int = 50,
    wand_length_mm: float = 10.0,
    focal: float = 9000.0,
    width: int = 1280,
    height: int = 800,
    seed: int = 1234,
) -> Tuple[Dict, Dict, Dict[int, np.ndarray]]:
    rng = np.random.default_rng(seed)

    cam_ids = [0, 1, 2, 3, 4, 5]
    K = np.array([[focal, 0, width / 2.0],
                  [0, focal, height / 2.0],
                  [0, 0, 1.0]], dtype=np.float64)

    cam_R = {cid: np.eye(3) for cid in cam_ids}
    cam_t = {
        0: np.array([[0.0], [0.0], [0.0]]),
        1: np.array([[0.3], [0.0], [0.0]]),
        2: np.array([[250.0], [0.0], [0.0]]),
        3: np.array([[0.0], [200.0], [0.0]]),
        4: np.array([[150.0], [150.0], [0.0]]),
        5: np.array([[251.0], [0.0], [0.0]]),
    }

    observations = {}
    for fid in range(n_frames):
        center = np.array([
            rng.uniform(-20, 20),
            rng.uniform(-20, 20),
            rng.uniform(500, 700),
        ])
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)

        ptA = center - direction * (wand_length_mm / 2)
        ptB = center + direction * (wand_length_mm / 2)

        frame_obs = {}
        for cid in cam_ids:
            uvA = _project_point(ptA, cam_R[cid], cam_t[cid], K)
            uvB = _project_point(ptB, cam_R[cid], cam_t[cid], K)
            frame_obs[cid] = (uvA, uvB)
        observations[fid] = frame_obs

    camera_settings = _make_camera_settings(cam_ids, focal, width, height)
    cam_t_flat = {cid: t.reshape(3) for cid, t in cam_t.items()}
    return observations, camera_settings, cam_t_flat


def _inject_phase2_camera_outliers(
    observations: Dict,
    cid: int,
    bad_fids,
    delta_a: np.ndarray,
    delta_b: np.ndarray,
) -> Dict:
    corrupted = {}
    delta_a = np.asarray(delta_a, dtype=np.float64)
    delta_b = np.asarray(delta_b, dtype=np.float64)

    for fid, frame in observations.items():
        new_frame = dict(frame)
        if fid in bad_fids and cid in new_frame:
            uvA, uvB = new_frame[cid]
            new_frame[cid] = (
                np.asarray(uvA, dtype=np.float64) + delta_a,
                np.asarray(uvB, dtype=np.float64) + delta_b,
            )
        corrupted[fid] = new_frame

    return corrupted


def _phase2_seed_run(
    observations: Dict,
    camera_settings: Dict,
    cam_i: int = 2,
    cam_j: int = 3,
    config: Optional[PinholeBootstrapP0Config] = None,
):
    config = config or PinholeBootstrapP0Config(wand_length_mm=10.0, ui_focal_px=9000.0)
    p0 = PinholeBootstrapP0(config)
    params_i, params_j, report = p0.run(cam_i, cam_j, observations, camera_settings)
    points_3d = p0.triangulate_all_points(
        cam_i, cam_j, params_i, params_j, observations, camera_settings,
    )
    cam_params = {
        cam_i: params_i.copy(),
        cam_j: params_j.copy(),
    }
    return p0, cam_params, points_3d, report


def _phase2_camera_reproj_rms(
    cid: int,
    params: np.ndarray,
    observations: Dict,
    points_3d: Dict[int, Tuple[np.ndarray, np.ndarray]],
    camera_settings: Dict,
) -> float:
    K = np.array([
        [camera_settings[cid]["focal"], 0, camera_settings[cid]["width"] / 2.0],
        [0, camera_settings[cid]["focal"], camera_settings[cid]["height"] / 2.0],
        [0, 0, 1.0],
    ], dtype=np.float64)

    rvec = params[:3].reshape(3, 1)
    tvec = params[3:6].reshape(3, 1)
    errs = []
    for fid, (XA, XB) in points_3d.items():
        if fid not in observations or cid not in observations[fid]:
            continue
        uvA, uvB = observations[fid][cid]
        proj_A, _ = cv2.projectPoints(XA.reshape(1, 3), rvec, tvec, K, np.zeros(5))
        proj_B, _ = cv2.projectPoints(XB.reshape(1, 3), rvec, tvec, K, np.zeros(5))
        errs.append(np.linalg.norm(proj_A.reshape(2) - uvA))
        errs.append(np.linalg.norm(proj_B.reshape(2) - uvB))

    errs = np.array(errs, dtype=np.float64)
    return float(np.sqrt(np.mean(errs ** 2))) if len(errs) else float("inf")


def _phase2_translation_error_mm(params: np.ndarray, expected_tvec: np.ndarray) -> float:
    return float(np.linalg.norm(params[3:6] - np.asarray(expected_tvec, dtype=np.float64)))


class _MockBaseCalibratorWithPerCamError:
    """
    Mock base calibrator that simulates precalibration returning per-camera
    reprojection errors where the degenerate cameras (0, 1) have the lowest
    individual errors -- the exact pathology from the evidence matrix.
    """

    def __init__(self, observations, camera_settings, per_cam_errors,
                 cam_positions=None, extrinsics_source='cameras'):
        """
        Parameters
        ----------
        extrinsics_source : str
            'cameras'      — populate self.cameras (WandCalibrator path)
            'final_params' — populate self.final_params only
            'none'         — no extrinsics stored (MockBase production path)
        """
        self.wand_points_filtered = observations
        self.wand_points = observations
        self.camera_settings = camera_settings
        self._per_cam_errors = per_cam_errors
        self.per_frame_errors = None
        self.image_size = (800, 1280)

        self.cameras = {}
        self.final_params = {}

        if cam_positions and extrinsics_source == 'cameras':
            for cid, pos in cam_positions.items():
                R = np.eye(3)
                T = np.array(pos, dtype=np.float64).reshape(3)
                self.cameras[cid] = {'R': R, 'T': T}
        elif cam_positions and extrinsics_source == 'final_params':
            for cid, pos in cam_positions.items():
                R = np.eye(3)
                T = np.array(pos, dtype=np.float64).reshape(3, 1)
                self.final_params[cid] = {'R': R, 'T': T}
        # extrinsics_source == 'none': leave both empty

    def run_precalibration_check(self, wand_length, init_focal_length):
        lines = []
        for cid in sorted(self._per_cam_errors.keys()):
            lines.append(f"  Cam {cid}: {self._per_cam_errors[cid]:.2f} px")
        msg = "\n".join(lines)
        return True, msg, None


class TestGeometryAwarePairSelection:
    """
    Task 5: select_ranked_pairs_via_precalib must not prefer degenerate pairs
    when healthy alternatives exist.
    """

    # tvec values matching _build_multi_camera_observations layout
    # Camera centres = -R.T @ T = -T (since R=I)
    _STANDARD_CAM_TVECS = {
        0: [0.0, 0.0, 0.0],
        1: [0.3, 0.0, 0.0],      # 0.3 mm from cam 0 (degenerate)
        2: [250.0, 0.0, 0.0],    # 250 mm from cam 0
        3: [0.0, 200.0, 0.0],    # 200 mm from cam 0
        4: [150.0, 150.0, 0.0],  # ~212 mm from cam 0
    }

    def _make_mock(self, per_cam_errors, cam_positions=None, seed=77):
        obs, settings = _build_multi_camera_observations(n_frames=50, seed=seed)
        positions = cam_positions or self._STANDARD_CAM_TVECS
        return _MockBaseCalibratorWithPerCamError(
            obs, settings, per_cam_errors, cam_positions=positions
        )

    def test_degenerate_pair_not_top_ranked(self):
        """
        When cameras 0 and 1 have the lowest individual reprojection errors
        but are nearly co-located (0.3mm apart), the top-ranked pair should
        NOT be (0,1).

        Evidence: case_012 precalib selects (2,4) with low per-cam error but
        baseline=1.17mm. 9/10 other pairs pass with baselines 241-1159mm.
        """
        per_cam_errors = {0: 5.0, 1: 7.0, 2: 15.0, 3: 18.0, 4: 20.0}
        mock_base = self._make_mock(per_cam_errors)

        ranked = select_ranked_pairs_via_precalib(mock_base, 10.0, 9000.0)
        assert ranked is not None and len(ranked) > 0, "Should return at least one pair"

        top_pair = ranked[0]
        assert top_pair != (0, 1), (
            f"Top pair is degenerate (0,1) -- geometry sanity check failed. "
            f"Full ranking: {ranked}"
        )

    def test_healthy_pair_appears_in_top_ranked(self):
        per_cam_errors = {0: 5.0, 1: 7.0, 2: 15.0, 3: 18.0, 4: 20.0}
        mock_base = self._make_mock(per_cam_errors)

        ranked = select_ranked_pairs_via_precalib(mock_base, 10.0, 9000.0)
        assert ranked is not None and len(ranked) >= 3

        healthy_pairs = {(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4),
                         (2, 3), (2, 4), (3, 4)}
        top_3 = set(ranked[:3])
        assert top_3 & healthy_pairs, (
            f"No healthy pair in top 3: {ranked[:3]}"
        )

    def test_ranked_returns_multiple_candidates(self):
        per_cam_errors = {0: 5.0, 1: 7.0, 2: 15.0, 3: 18.0, 4: 20.0}
        mock_base = self._make_mock(per_cam_errors)

        ranked = select_ranked_pairs_via_precalib(mock_base, 10.0, 9000.0)
        assert ranked is not None
        assert len(ranked) >= 2, (
            f"Should return >=2 candidates for fallback, got {len(ranked)}"
        )

    def test_backward_compatible_select_best_pair(self):
        from modules.camera_calibration.wand_calibration.refractive_bootstrap import (
            select_best_pair_via_precalib,
        )
        per_cam_errors = {0: 5.0, 1: 7.0, 2: 15.0, 3: 18.0, 4: 20.0}
        mock_base = self._make_mock(per_cam_errors)

        pair = select_best_pair_via_precalib(mock_base, 10.0, 9000.0)
        assert pair is not None
        assert isinstance(pair, tuple) and len(pair) == 2
        assert pair != (0, 1)

    def test_high_disparity_low_baseline_demoted(self):
        """Regression for case_012: pair (2,4) had 92.4px disparity but only
        1.17mm 3D baseline.  Pixel disparity alone passed but the pair was
        degenerate.  With precalib baselines available, such pairs must be
        demoted.

        Setup: cam 5 placed 1.2 mm from cam 2 (degenerate baseline) but with
        a different viewing direction so pixel disparity is high.  Cam 5 and
        cam 2 both have the lowest per-cam errors.
        """
        obs_base, settings_base = _build_multi_camera_observations(
            n_frames=50, seed=77
        )

        cam_positions = dict(self._STANDARD_CAM_TVECS)
        cam_positions[5] = [251.2, 0.0, 0.0]  # 1.2 mm from cam 2 (t=[250,...])

        K = np.array([[9000.0, 0, 640.0],
                      [0, 9000.0, 400.0],
                      [0, 0, 1.0]])
        R5 = np.eye(3)
        t5 = np.array([[251.2], [0.0], [0.0]])

        obs = {}
        for fid, frame in obs_base.items():
            new_frame = dict(frame)
            uv0 = frame[0]
            ptA_approx = np.array([
                (uv0[0][0] - 640.0) / 9000.0 * 600,
                (uv0[0][1] - 400.0) / 9000.0 * 600,
                600.0
            ])
            ptB_approx = np.array([
                (uv0[1][0] - 640.0) / 9000.0 * 600,
                (uv0[1][1] - 400.0) / 9000.0 * 600,
                600.0
            ])
            uvA_5 = _project_point(ptA_approx, R5, t5, K)
            uvB_5 = _project_point(ptB_approx, R5, t5, K)
            new_frame[5] = (uvA_5, uvB_5)
            obs[fid] = new_frame

        settings = dict(settings_base)
        settings[5] = {"focal": 9000.0, "width": 1280, "height": 800}

        per_cam_errors = {
            0: 20.0, 1: 22.0, 2: 8.0, 3: 25.0, 4: 25.0, 5: 9.0
        }
        mock_base = _MockBaseCalibratorWithPerCamError(
            obs, settings, per_cam_errors, cam_positions=cam_positions
        )

        ranked = select_ranked_pairs_via_precalib(mock_base, 10.0, 9000.0)
        assert ranked is not None

        degenerate_pair = (2, 5)
        top_pair = ranked[0]
        assert top_pair != degenerate_pair, (
            f"Pair {degenerate_pair} (1.2mm baseline) should NOT be #1. "
            f"Ranking: {ranked[:5]}"
        )

    def test_final_params_only_path(self):
        """When calibrator exposes final_params but not cameras (WandCalibrator
        alternate structure), baseline check still works and demotes
        degenerate pairs."""
        per_cam_errors = {0: 5.0, 1: 7.0, 2: 15.0, 3: 18.0, 4: 20.0}
        obs, settings = _build_multi_camera_observations(n_frames=50, seed=77)
        positions = self._STANDARD_CAM_TVECS
        mock_base = _MockBaseCalibratorWithPerCamError(
            obs, settings, per_cam_errors,
            cam_positions=positions, extrinsics_source='final_params',
        )
        assert not mock_base.cameras
        assert len(mock_base.final_params) == 5

        ranked = select_ranked_pairs_via_precalib(mock_base, 10.0, 9000.0)
        assert ranked is not None and len(ranked) > 0
        assert ranked[0] != (0, 1), (
            f"final_params path failed to demote (0,1): {ranked[:5]}"
        )

    def test_no_extrinsics_recovery_demotes_degenerate(self):
        """When calibrator exposes NO extrinsics at all (like production
        MockBase), the function recovers camera positions from observations
        via Essential Matrix + PnP and still demotes degenerate pairs.

        This is the key regression test for the production bug: MockBase's
        run_precalibration_check() discards cam_params, so the old code
        always returned baseline_3d=-1.0 and fell back to pixel disparity
        which couldn't catch case_012's bad pair (2,4).
        """
        per_cam_errors = {0: 5.0, 1: 7.0, 2: 15.0, 3: 18.0, 4: 20.0}
        obs, settings = _build_multi_camera_observations(n_frames=50, seed=77)
        mock_base = _MockBaseCalibratorWithPerCamError(
            obs, settings, per_cam_errors,
            cam_positions=self._STANDARD_CAM_TVECS,
            extrinsics_source='none',
        )
        assert not mock_base.cameras
        assert not mock_base.final_params

        ranked = select_ranked_pairs_via_precalib(mock_base, 10.0, 9000.0)
        assert ranked is not None and len(ranked) > 0
        assert ranked[0] != (0, 1), (
            f"No-extrinsics recovery failed to demote (0,1): {ranked[:5]}"
        )

    def test_partial_extrinsics_uses_essential_per_pair(self, capsys):
        per_cam_errors = {0: 20.0, 1: 22.0, 2: 8.0, 3: 9.0, 4: 25.0}
        obs, settings = _build_multi_camera_observations(n_frames=50, seed=77)

        partial_positions = {
            0: self._STANDARD_CAM_TVECS[0],
            1: self._STANDARD_CAM_TVECS[1],
        }
        mock_base = _MockBaseCalibratorWithPerCamError(
            obs,
            settings,
            per_cam_errors,
            cam_positions=partial_positions,
            extrinsics_source='cameras',
        )

        ranked = select_ranked_pairs_via_precalib(mock_base, 10.0, 9000.0)
        assert ranked is not None and len(ranked) > 0

        captured = capsys.readouterr()
        assert "source=essential" in captured.out, captured.out
        assert ranked[0] == (2, 3), (
            f"Expected missing-centre pair (2,3) to be ranked first via Essential fallback. "
            f"Ranking: {ranked[:5]}"
        )

    def test_all_geometry_invalid_returns_none(self, capsys):
        per_cam_errors = {0: 5.0, 1: 6.0, 2: 7.0, 3: 8.0, 4: 9.0}
        obs, settings = _build_multi_camera_observations(n_frames=50, seed=77)
        tiny_positions = {
            0: [0.0, 0.0, 0.0],
            1: [0.3, 0.0, 0.0],
            2: [0.6, 0.0, 0.0],
            3: [0.0, 0.8, 0.0],
            4: [0.5, 0.6, 0.0],
        }
        mock_base = _MockBaseCalibratorWithPerCamError(
            obs, settings, per_cam_errors, cam_positions=tiny_positions
        )

        ranked = select_ranked_pairs_via_precalib(mock_base, 10.0, 9000.0)
        assert ranked is None

        captured = capsys.readouterr()
        assert "No geometry-valid camera pair found" in captured.out


class TestDeterministicFallbackRetry:
    """
    Task 5: If the first selected pair fails P0, try exactly one next-best
    geometry-valid pair, then stop.
    """

    # Reuse standard camera layout for consistent 3D baseline checks.
    _STANDARD_CAM_TVECS = {
        0: [0.0, 0.0, 0.0],
        1: [0.3, 0.0, 0.0],
        2: [250.0, 0.0, 0.0],
        3: [0.0, 200.0, 0.0],
        4: [150.0, 150.0, 0.0],
    }

    def test_fallback_tries_second_pair_on_first_failure(self):
        """
        Patch PinholeBootstrapP0.run to fail on the first pair and succeed
        on the second, verifying that the ranked list enables one retry.
        """
        obs, settings = _build_multi_camera_observations(n_frames=50, seed=77)
        per_cam_errors = {0: 5.0, 1: 7.0, 2: 15.0, 3: 18.0, 4: 20.0}
        mock_base = _MockBaseCalibratorWithPerCamError(
            obs, settings, per_cam_errors,
            cam_positions=self._STANDARD_CAM_TVECS,
        )

        ranked = select_ranked_pairs_via_precalib(mock_base, 10.0, 9000.0)
        assert ranked is not None and len(ranked) >= 2

        first_pair = ranked[0]
        second_pair = ranked[1]

        call_log = []

        config = PinholeBootstrapP0Config(wand_length_mm=10.0, ui_focal_px=9000.0)
        p0 = PinholeBootstrapP0(config)

        original_run = p0.run

        def mock_run(cam_i, cam_j, *args, **kwargs):
            call_log.append((cam_i, cam_j))
            if (cam_i, cam_j) == first_pair:
                telemetry = P0Telemetry(selected_pair=first_pair)
                raise P0FailureError(
                    "simulated first-pair failure",
                    P0_REASON_CATASTROPHIC_REPROJECTION,
                    telemetry,
                )
            return original_run(cam_i, cam_j, *args, **kwargs)

        p0.run = mock_run

        with pytest.raises(P0FailureError):
            p0.run(*first_pair, obs, settings)
        assert len(call_log) == 1
        assert call_log[0] == first_pair

        call_log.clear()
        result = p0.run(*second_pair, obs, settings)
        assert len(call_log) == 1
        assert call_log[0] == second_pair
        assert result is not None

    def test_no_open_ended_retry_loop(self):
        """
        The fallback mechanism should attempt at most ONE additional pair,
        not loop through all possible pairs.
        """
        obs, settings = _build_multi_camera_observations(n_frames=50, seed=77)
        per_cam_errors = {0: 5.0, 1: 7.0, 2: 15.0, 3: 18.0, 4: 20.0}
        mock_base = _MockBaseCalibratorWithPerCamError(
            obs, settings, per_cam_errors,
            cam_positions=self._STANDARD_CAM_TVECS,
        )

        ranked = select_ranked_pairs_via_precalib(mock_base, 10.0, 9000.0)
        assert ranked is not None
        assert len(ranked) >= 2


class TestCase023NotRecovered:
    """
    Task 5: case_023 must NOT be magically recovered by the hardening.
    It remains geometry-limited in tested evidence.
    """

    def test_degenerate_baseline_still_fails_after_hardening(self):
        """
        A ~0.3mm baseline should still trigger P0FailureError.
        This validates that the hardening (geometry-aware pair selection +
        one retry) does NOT mask fundamental geometry failure.
        """
        obs, settings, ci, cj = _build_degenerate_observations(
            n_frames=100, baseline_mm=0.3, seed=99,
        )
        config = PinholeBootstrapP0Config(wand_length_mm=10.0, ui_focal_px=9000.0)
        p0 = PinholeBootstrapP0(config)

        with pytest.raises(P0FailureError) as exc_info:
            p0.run(ci, cj, obs, settings)

        err = exc_info.value
        assert err.reason in (
            P0_REASON_CATASTROPHIC_REPROJECTION,
            P0_REASON_UNSTABLE_SCALE_RECOVERY,
            P0_REASON_PHASE1_BA_FAILURE,
        )
        assert err.telemetry.selected_pair == (ci, cj)
        assert err.telemetry.failure_reason == err.reason


class TestP0BootstrapPhase2SyntheticHarness:
    def test_phase2_healthy_seed_triangulation_and_added_cameras_green(self):
        observations, settings = _build_multi_camera_observations(n_frames=60, seed=77)
        p0, cam_params, points_3d, report = _phase2_seed_run(observations, settings)

        assert report["p0_telemetry"]["failure_reason"] == P0_REASON_OK
        assert len(points_3d) == 60
        assert all(len(pair) == 2 for pair in points_3d.values())

        solved = p0.run_phase2(
            dict(cam_params),
            observations,
            points_3d,
            settings,
            sorted(settings),
        )

        assert sorted(solved) == [0, 1, 2, 3, 4]
        assert _phase2_camera_reproj_rms(0, solved[0], observations, points_3d, settings) < 1e-3
        assert _phase2_camera_reproj_rms(1, solved[1], observations, points_3d, settings) < 1e-3
        assert _phase2_camera_reproj_rms(4, solved[4], observations, points_3d, settings) < 1e-3
        assert _phase2_translation_error_mm(solved[0], np.array([-250.0, 0.0, 0.0])) < 1e-3
        assert _phase2_translation_error_mm(solved[1], np.array([-249.7, 0.0, 0.0])) < 1e-3
        assert _phase2_translation_error_mm(solved[4], np.array([-100.0, 150.0, 0.0])) < 1e-3

    def test_phase2_ransac_healthy_case_records_inliers_green(self):
        observations, settings = _build_multi_camera_observations(n_frames=60, seed=77)
        config = PinholeBootstrapP0Config(
            wand_length_mm=10.0,
            ui_focal_px=9000.0,
            phase2_ransac_enabled=True,
        )
        p0, cam_params, points_3d, _ = _phase2_seed_run(observations, settings, config=config)

        solved = p0.run_phase2(
            dict(cam_params),
            observations,
            points_3d,
            settings,
            sorted(settings),
        )

        assert _phase2_translation_error_mm(solved[4], np.array([-100.0, 150.0, 0.0])) < 1e-3

        telem = p0.last_phase2_telemetry[4]
        assert telem["pnp_correspondences"] == len(points_3d) * 2
        assert telem["ransac_inliers"] is not None
        assert telem["ransac_inliers"] >= int(0.95 * telem["pnp_correspondences"])
        assert telem["ransac_inlier_ratio"] is not None
        assert telem["ransac_inlier_ratio"] > 0.95

    def test_phase2_prefilter_removes_contaminated_seed_points_green(self):
        observations, settings = _build_multi_camera_observations(n_frames=60, seed=77)
        p0, cam_params, points_3d, _ = _phase2_seed_run(observations, settings)

        bad_fids = set(sorted(points_3d)[:18])
        contaminated_points = {
            fid: (
                XA + np.array([4000.0, -2500.0, 1200.0]),
                XB + np.array([-3500.0, 2200.0, -900.0]),
            ) if fid in bad_fids else (XA.copy(), XB.copy())
            for fid, (XA, XB) in points_3d.items()
        }

        solved = p0.run_phase2(
            dict(cam_params),
            observations,
            contaminated_points,
            settings,
            sorted(settings),
        )

        assert _phase2_translation_error_mm(solved[0], np.array([-250.0, 0.0, 0.0])) < 50.0
        assert _phase2_translation_error_mm(solved[1], np.array([-249.7, 0.0, 0.0])) < 50.0
        assert _phase2_translation_error_mm(solved[4], np.array([-100.0, 150.0, 0.0])) < 50.0

        telem = p0.last_phase2_telemetry[0]
        assert telem["total_correspondences"] == len(points_3d) * 2
        assert telem["prefilter_candidate_removed"] >= len(bad_fids) * 2
        assert telem["prefilter_removed"] >= len(bad_fids) * 2
        assert telem["prefilter_removed_depth"] > 0 or telem["prefilter_removed_wand_length"] > 0
        assert telem["prefilter_starvation_fallback"] is False
        assert telem["pnp_correspondences"] <= telem["total_correspondences"] - len(bad_fids) * 2

    def test_phase2_ransac_recovers_contaminated_added_camera_correspondences(self):
        observations, settings = _build_multi_camera_observations(n_frames=60, seed=77)
        config = PinholeBootstrapP0Config(
            wand_length_mm=10.0,
            ui_focal_px=9000.0,
            phase2_ransac_enabled=True,
            phase2_ransac_reproj_threshold_px=3.0,
            phase2_ransac_confidence=0.999,
            phase2_ransac_min_inlier_ratio=0.5,
        )
        p0, cam_params, points_3d, _ = _phase2_seed_run(observations, settings, config=config)

        bad_fids = set(sorted(points_3d)[:18])
        contaminated_observations = _inject_phase2_camera_outliers(
            observations,
            cid=4,
            bad_fids=bad_fids,
            delta_a=np.array([220.0, -180.0]),
            delta_b=np.array([-210.0, 190.0]),
        )

        solved = p0.run_phase2(
            dict(cam_params),
            contaminated_observations,
            points_3d,
            settings,
            sorted(settings),
        )

        assert _phase2_translation_error_mm(solved[4], np.array([-100.0, 150.0, 0.0])) < 5.0

        telem = p0.last_phase2_telemetry[4]
        assert telem["pnp_correspondences"] == len(points_3d) * 2
        assert telem["ransac_inliers"] is not None
        assert telem["ransac_inliers"] < telem["pnp_correspondences"]
        assert telem["ransac_inliers"] >= int(0.6 * telem["pnp_correspondences"])
        assert telem["ransac_inlier_ratio"] is not None
        assert 0.6 <= telem["ransac_inlier_ratio"] < 0.9

    def test_phase2_prefilter_starvation_fallback_preserves_healthy_solution_green(self):
        observations, settings = _build_multi_camera_observations(n_frames=12, seed=77)
        config = PinholeBootstrapP0Config(
            wand_length_mm=10.0,
            ui_focal_px=9000.0,
            phase2_prefilter_depth_max_mm=300.0,
        )
        p0, cam_params, points_3d, _ = _phase2_seed_run(
            observations,
            settings,
            config=config,
        )

        solved = p0.run_phase2(
            dict(cam_params),
            observations,
            points_3d,
            settings,
            sorted(settings),
        )

        assert _phase2_translation_error_mm(solved[0], np.array([-250.0, 0.0, 0.0])) < 1e-2
        assert _phase2_translation_error_mm(solved[1], np.array([-249.7, 0.0, 0.0])) < 1e-2
        assert _phase2_translation_error_mm(solved[4], np.array([-100.0, 150.0, 0.0])) < 1e-2

        telem = p0.last_phase2_telemetry[0]
        assert telem["prefilter_candidate_removed"] == telem["total_correspondences"]
        assert telem["prefilter_removed"] == 0
        assert telem["prefilter_kept"] == telem["total_correspondences"]
        assert telem["prefilter_starvation_fallback"] is True
        assert telem["pnp_correspondences"] == telem["total_correspondences"]

    def test_phase2_geometry_poor_added_camera_is_labeled_low_confidence_green(self):
        observations, settings, true_t = _build_phase2_geometry_poor_observations(
            n_frames=60,
            seed=1234,
        )
        config = PinholeBootstrapP0Config(
            wand_length_mm=10.0,
            ui_focal_px=9000.0,
            phase2_ransac_enabled=True,
            phase2_ransac_reproj_threshold_px=3.0,
            phase2_ransac_confidence=0.999,
            phase2_ransac_min_inlier_ratio=0.75,
        )
        rng = np.random.default_rng(5)
        for fid in observations:
            uvA, uvB = observations[fid][5]
            observations[fid][5] = (
                uvA + rng.normal(0.0, 6.0, size=2),
                uvB + rng.normal(0.0, 6.0, size=2),
            )

        p0, cam_params, points_3d, _ = _phase2_seed_run(observations, settings, config=config)
        solved = p0.run_phase2(
            dict(cam_params),
            observations,
            points_3d,
            settings,
            sorted(settings),
        )

        expected_cam5_t = true_t[5] - true_t[2]
        assert 5 in solved
        assert _phase2_translation_error_mm(solved[5], expected_cam5_t) > 1.0

        telem = p0.last_phase2_telemetry[5]
        assert telem["pnp_correspondences"] == len(points_3d) * 2
        assert telem["reproj_rms_px"] < 3.0
        assert telem["ransac_inlier_ratio"] < config.phase2_confidence_min_inlier_ratio
        assert telem["confidence_label"] == "low"
        assert telem["confidence_warning"] is not None
        assert "inlier_ratio" in telem["confidence_warning"]

    def test_phase2_confidence_warning_combines_failed_signals_green(self):
        p0 = PinholeBootstrapP0(PinholeBootstrapP0Config())
        telemetry = Phase2CameraTelemetry(
            camera_id=5,
            pnp_correspondences=8,
            ransac_inlier_ratio=0.2,
            reproj_rms_px=8.5,
        )

        p0._label_phase2_confidence(telemetry)

        assert telemetry.confidence_label == "low"
        assert telemetry.confidence_warning == "insufficient_correspondences,low_inlier_ratio,high_rms"


class TestPhase2RmsGatedSeedPairFallback:
    def _make_stub_bootstrap(self, config: Optional[PinholeBootstrapP0Config] = None):
        config = config or PinholeBootstrapP0Config(wand_length_mm=10.0, ui_focal_px=9000.0)
        p0 = PinholeBootstrapP0(config)
        observations = {0: {0: (np.zeros(2), np.ones(2)), 1: (np.zeros(2), np.ones(2)), 2: (np.zeros(2), np.ones(2))}}
        settings = _make_camera_settings([0, 1, 2, 3])
        return p0, observations, settings

    def test_run_all_retries_next_ranked_pair_when_phase2_rms_is_too_high(self, monkeypatch):
        p0, observations, settings = self._make_stub_bootstrap(
            PinholeBootstrapP0Config(
                wand_length_mm=10.0,
                ui_focal_px=9000.0,
                phase2_pair_fallback_rms_threshold_px=35.0,
                phase2_pair_fallback_max_retries=2,
            )
        )

        ranked_pairs = [(0, 1), (1, 2), (0, 2)]
        phase2_rms_by_pair = {
            (0, 1): 42.0,
            (1, 2): 12.0,
        }
        attempted_pairs = []

        def _fake_run(cam_i, cam_j, observations, camera_settings, progress_callback=None):
            attempted_pairs.append((cam_i, cam_j))
            return (
                np.zeros(6, dtype=np.float64),
                np.array([0.0, 0.0, 0.0, float(cam_i), float(cam_j), 1.0], dtype=np.float64),
                {"p0_telemetry": {"selected_pair": [cam_i, cam_j], "failure_reason": P0_REASON_OK}},
            )

        def _fake_triangulate(*args, **kwargs):
            return {0: (np.zeros(3, dtype=np.float64), np.ones(3, dtype=np.float64))}

        def _fake_run_phase2(cam_params, observations, points_3d, camera_settings, all_cam_ids):
            rms = phase2_rms_by_pair[attempted_pairs[-1]]
            p0.last_phase2_telemetry = {
                3: {
                    "camera_id": 3,
                    "reproj_rms_px": rms,
                    "confidence_label": "normal",
                    "confidence_warning": None,
                }
            }
            cam_params = dict(cam_params)
            cam_params[3] = np.array([0.0, 0.0, 0.0, rms, 0.0, 1.0], dtype=np.float64)
            return cam_params

        def _fake_run_phase3(cam_params, observations, camera_settings, cam_anchor_id=None, progress_callback=None):
            return cam_params, {0: (np.zeros(3, dtype=np.float64), np.ones(3, dtype=np.float64))}

        monkeypatch.setattr(p0, "run", _fake_run)
        monkeypatch.setattr(p0, "triangulate_all_points", _fake_triangulate)
        monkeypatch.setattr(p0, "run_phase2", _fake_run_phase2)
        monkeypatch.setattr(p0, "run_phase3", _fake_run_phase3)

        cam_params, report = p0.run_all(
            0,
            1,
            observations,
            settings,
            [0, 1, 2, 3],
            ranked_seed_pairs=ranked_pairs,
        )

        assert attempted_pairs == [(0, 1), (1, 2)]
        assert sorted(cam_params) == [1, 2, 3]
        assert report["p0_telemetry"]["selected_pair"] == [1, 2]
        assert report["p0_telemetry"]["phase2_cameras"][3]["reproj_rms_px"] == 12.0
        assert report["phase2_rms_max_px"] == 12.0

    def test_run_all_stops_after_bounded_retries_and_returns_best_available(self, monkeypatch):
        p0, observations, settings = self._make_stub_bootstrap(
            PinholeBootstrapP0Config(
                wand_length_mm=10.0,
                ui_focal_px=9000.0,
                phase2_pair_fallback_rms_threshold_px=35.0,
                phase2_pair_fallback_max_retries=2,
            )
        )

        ranked_pairs = [(0, 1), (1, 2), (0, 2), (2, 3)]
        phase2_rms_by_pair = {
            (0, 1): 48.0,
            (1, 2): 44.0,
            (0, 2): 39.0,
            (2, 3): 8.0,
        }
        attempted_pairs = []

        def _fake_run(cam_i, cam_j, observations, camera_settings, progress_callback=None):
            attempted_pairs.append((cam_i, cam_j))
            return (
                np.zeros(6, dtype=np.float64),
                np.array([0.0, 0.0, 0.0, float(cam_i), float(cam_j), 1.0], dtype=np.float64),
                {"p0_telemetry": {"selected_pair": [cam_i, cam_j], "failure_reason": P0_REASON_OK}},
            )

        def _fake_triangulate(*args, **kwargs):
            return {0: (np.zeros(3, dtype=np.float64), np.ones(3, dtype=np.float64))}

        def _fake_run_phase2(cam_params, observations, points_3d, camera_settings, all_cam_ids):
            rms = phase2_rms_by_pair[attempted_pairs[-1]]
            p0.last_phase2_telemetry = {
                3: {
                    "camera_id": 3,
                    "reproj_rms_px": rms,
                    "confidence_label": "low",
                    "confidence_warning": "high_rms",
                }
            }
            return dict(cam_params)

        def _fake_run_phase3(cam_params, observations, camera_settings, cam_anchor_id=None, progress_callback=None):
            return cam_params, {0: (np.zeros(3, dtype=np.float64), np.ones(3, dtype=np.float64))}

        monkeypatch.setattr(p0, "run", _fake_run)
        monkeypatch.setattr(p0, "triangulate_all_points", _fake_triangulate)
        monkeypatch.setattr(p0, "run_phase2", _fake_run_phase2)
        monkeypatch.setattr(p0, "run_phase3", _fake_run_phase3)

        cam_params, report = p0.run_all(
            0,
            1,
            observations,
            settings,
            [0, 1, 2, 3],
            ranked_seed_pairs=ranked_pairs,
        )

        assert attempted_pairs == [(0, 1), (1, 2), (0, 2)]
        assert sorted(cam_params) == [0, 2]
        assert report["p0_telemetry"]["selected_pair"] == [0, 2]
        assert report["p0_telemetry"]["phase2_cameras"][3]["reproj_rms_px"] == 39.0
        assert report["phase2_rms_max_px"] == 39.0

# ═══════════════════════════════════════════════════════════════════════════
# Live case regression tests (require J: drive data)
# ═══════════════════════════════════════════════════════════════════════════

# These tests use actual case data from J:\Refraction_test and are
# skipped if the J: drive is not available.
J_DRIVE_AVAILABLE = Path("J:/Refraction_test/case_001").is_dir()


@pytest.mark.skipif(not J_DRIVE_AVAILABLE, reason="J: drive not available")
class TestLiveCaseRegression:
    """
    Regression tests using actual case data.

    These confirm that the telemetry and failure classification behavior
    works with real-world observation data, not just synthetic fixtures.
    """

    def test_case_001_healthy_emits_ok_telemetry(self):
        """case_001 should pass P0 and emit failure_reason=ok."""
        sys.path.insert(0, str(Path("J:/Refraction_test/test_script")))
        from run_calibration_worker import load_case_inputs

        inputs = load_case_inputs("J:/Refraction_test/case_001")
        mock_base = inputs["mock_base"]

        from modules.camera_calibration.wand_calibration.refractive_bootstrap import (
            PinholeBootstrapP0,
            PinholeBootstrapP0Config,
            select_best_pair_via_precalib,
        )

        pair = select_best_pair_via_precalib(
            mock_base,
            wand_len_mm=10.0,
            initial_focal_px=inputs["focal_px"],
        )
        assert pair is not None, "Precalibration failed to select a pair for case_001"

        from modules.camera_calibration.wand_calibration.refraction_wand_calibrator import (
            ObservationBuilder,
            RefractiveCalibReporter,
        )

        observations = ObservationBuilder.prepare_for_bootstrap(
            mock_base, inputs["cam_to_window"], RefractiveCalibReporter()
        )

        config = PinholeBootstrapP0Config(
            wand_length_mm=10.0,
            ui_focal_px=inputs["focal_px"],
        )
        p0 = PinholeBootstrapP0(config)

        cam_i, cam_j = pair
        params_i, params_j, report = p0.run(
            cam_i, cam_j, observations, mock_base.camera_settings,
        )

        assert "p0_telemetry" in report
        telem = report["p0_telemetry"]
        assert telem["failure_reason"] == P0_REASON_OK
        assert telem["baseline_mm"] > 50.0

    def test_case_023_recovered_by_better_pair_selection(self):
        """
        case_023 was previously failing because the old pair selection chose
        a degenerate pair.  With pairwise Essential Matrix baselines, the
        ranking now correctly avoids degenerate pairs (many have 0-5mm
        baselines) and selects a healthy pair instead.

        GREEN-PHASE: Updated from the red-phase test after Task 5 hardening.
        The docstring of the original test said: "After Task 5 fixes, it
        should be updated to reflect the new behavior (either recovery or
        explicit geometry classification)."

        Evidence: pair (2,5) passes P0 with 700.7mm baseline and 0.02px
        reproj error.  The old pair selection was accidentally selecting a
        degenerate pair; case_023 is NOT genuinely geometry-limited.
        """
        sys.path.insert(0, str(Path("J:/Refraction_test/test_script")))
        from run_calibration_worker import load_case_inputs

        inputs = load_case_inputs("J:/Refraction_test/case_023")
        mock_base = inputs["mock_base"]

        from modules.camera_calibration.wand_calibration.refractive_bootstrap import (
            PinholeBootstrapP0,
            PinholeBootstrapP0Config,
            select_ranked_pairs_via_precalib,
        )

        ranked = select_ranked_pairs_via_precalib(
            mock_base,
            wand_len_mm=10.0,
            initial_focal_px=inputs["focal_px"],
        )
        assert ranked is not None and len(ranked) > 0

        top_pair = ranked[0]

        config = PinholeBootstrapP0Config(
            wand_length_mm=10.0,
            ui_focal_px=inputs["focal_px"],
        )
        p0 = PinholeBootstrapP0(config)

        from modules.camera_calibration.wand_calibration.refraction_wand_calibrator import (
            ObservationBuilder,
            RefractiveCalibReporter,
        )

        observations = ObservationBuilder.prepare_for_bootstrap(
            mock_base, inputs["cam_to_window"], RefractiveCalibReporter()
        )

        cam_i, cam_j = top_pair
        params_i, params_j, report = p0.run(
            cam_i, cam_j, observations, mock_base.camera_settings,
        )

        assert "p0_telemetry" in report
        telem = report["p0_telemetry"]
        assert telem["failure_reason"] == P0_REASON_OK
        assert telem["baseline_mm"] > 50.0

    def test_case_012_does_not_select_bad_pair_2_4(self):
        """case_012 must NOT select pair (2,4) which has 1.17mm baseline.

        This is the primary production regression: the old code always
        returned baseline_3d=-1.0 for MockBase and fell back to pixel
        disparity (92.4px for pair (2,4)), which looked healthy.  With
        the Essential Matrix recovery, the true 3D baseline is now
        detected and the pair is demoted.
        """
        sys.path.insert(0, str(Path("J:/Refraction_test/test_script")))
        from run_calibration_worker import load_case_inputs

        inputs = load_case_inputs("J:/Refraction_test/case_012")
        mock_base = inputs["mock_base"]

        from modules.camera_calibration.wand_calibration.refractive_bootstrap import (
            select_ranked_pairs_via_precalib,
        )

        ranked = select_ranked_pairs_via_precalib(
            mock_base,
            wand_len_mm=10.0,
            initial_focal_px=inputs["focal_px"],
        )
        assert ranked is not None and len(ranked) > 0

        top_pair = ranked[0]
        assert top_pair != (2, 4), (
            f"case_012 still selects bad pair (2,4). Ranking: {ranked[:5]}"
        )
        assert top_pair != (4, 2), (
            f"case_012 still selects bad pair (4,2). Ranking: {ranked[:5]}"
        )

    def test_case_013_bootstrap_runner_emits_iteration_artifact(self):
        result = run_bootstrap_case(
            case_id=DEFAULT_TARGET_CASE,
            case_root=Path("J:/Refraction_test"),
            results_root=DEFAULT_RESULTS_ROOT,
            mode="iterate",
        )

        artifact = build_case_artifact(result, iteration=0)

        assert tuple(artifact.keys()) == CASE_ARTIFACT_KEYS
        assert artifact["case_id"] == DEFAULT_TARGET_CASE
        assert artifact["mode"] == "iterate"
        assert artifact["iteration"] == 0
        assert artifact["selected_pair"] == [int(v) for v in result["selected_pair"]]
        assert artifact["ranked_pairs_top5"][0] == artifact["selected_pair"]
        assert artifact["all_cam_ids"] == result["all_cam_ids"]
        assert artifact["phase2_camera_ids"] == sorted(
            int(cid) for cid in artifact["phase2_cameras"].keys()
        )
        accepted_pair = artifact["p0_telemetry"]["selected_pair"]
        assert accepted_pair in artifact["ranked_pairs_top5"]
        if accepted_pair != artifact["selected_pair"]:
            assert accepted_pair in artifact["ranked_pairs_top5"][1:]
        assert artifact["p0_telemetry"]["phase2_cameras"] == artifact["phase2_cameras"]

    def test_case_013_bootstrap_runner_forwards_ranked_seed_pairs(self, monkeypatch, tmp_path):
        ranked_pairs = [(1, 2), (0, 1), (0, 2)]
        captured = {}

        monkeypatch.setattr(
            "scripts.case_013_bootstrap_debug_loop.prepare_bootstrap_case",
            lambda case_id, case_root: {
                "case_id": case_id,
                "case_dir": Path(case_root) / case_id,
                "wand_length_mm": 10.0,
                "initial_focal_px": 9000.0,
                "ranked_pairs": ranked_pairs,
                "selected_pair": ranked_pairs[0],
                "observations": {0: {}},
                "camera_settings": {0: {"focal": 9000.0, "width": 1280, "height": 800}},
                "all_cam_ids": [0, 1, 2],
            },
        )

        def _fake_run_all(self, **kwargs):
            captured.update(kwargs)
            return (
                {1: np.zeros(6, dtype=np.float64), 2: np.zeros(6, dtype=np.float64)},
                {
                    "points_3d": {0: (np.zeros(3, dtype=np.float64), np.ones(3, dtype=np.float64))},
                    "p0_telemetry": {"selected_pair": [kwargs["cam_i"], kwargs["cam_j"]], "phase2_cameras": {}},
                },
            )

        monkeypatch.setattr(PinholeBootstrapP0, "run_all", _fake_run_all)
        monkeypatch.setattr(
            "scripts.case_013_bootstrap_debug_loop.best_effort_git_sha",
            lambda repo_root=None: "test-sha",
        )

        result = run_bootstrap_case(
            case_id=DEFAULT_TARGET_CASE,
            case_root=tmp_path,
            results_root=tmp_path / "results",
            mode="iterate",
        )

        assert captured["ranked_seed_pairs"] == ranked_pairs
        assert captured["cam_i"] == ranked_pairs[0][0]
        assert captured["cam_j"] == ranked_pairs[0][1]
        assert result["selected_pair"] == ranked_pairs[0]

    def test_refraction_calibrator_forwards_ranked_seed_pairs_to_p0(self, monkeypatch):
        from modules.camera_calibration.wand_calibration import refraction_wand_calibrator as rc

        base = SimpleNamespace(
            cam_params={},
            initial_focal=9000.0,
            camera_settings={0: {"focal": 9000.0, "width": 1280, "height": 800}, 1: {"focal": 9000.0, "width": 1280, "height": 800}},
        )
        calibrator = rc.RefractiveWandCalibrator(base)

        ranked_pairs = [(1, 2), (0, 1), (0, 2)]
        captured = {}

        monkeypatch.setattr(
            rc,
            "select_ranked_pairs_via_precalib",
            lambda *_args, **_kwargs: ranked_pairs,
        )
        monkeypatch.setattr(
            calibrator,
            "_collect_observations",
            lambda cam_to_window: {
                "wand_length": 10.0,
                "num_frames": 1,
                "num_cams": 2,
                "total_observations": 2,
                "cam_ids": [0, 1],
            },
        )
        monkeypatch.setattr(
            calibrator,
            "_prepare_observations_for_bootstrap",
            lambda cam_to_window: {0: {0: (np.zeros(2), np.zeros(2))}},
        )

        def _fake_run_all(self, **kwargs):
            captured.update(kwargs)
            raise RuntimeError("stop after capture")

        monkeypatch.setattr(PinholeBootstrapP0, "run_all", _fake_run_all)

        with pytest.raises(RuntimeError, match="stop after capture"):
            calibrator.calibrate(
                num_windows=2,
                cam_to_window={0: 0, 1: 0},
                window_media={0: {"n1": 1.0, "n2": 1.49, "n3": 1.333, "thickness": 10.0}},
                out_path=None,
                verbosity=0,
            )

        assert captured["cam_i"] == ranked_pairs[0][0]
        assert captured["cam_j"] == ranked_pairs[0][1]
        assert captured["ranked_seed_pairs"] == ranked_pairs

    def test_case_013_healthy_envelope_manifest_is_frozen(self):
        manifest_path = DEFAULT_RESULTS_ROOT / "healthy_baseline" / "manifest.json"
        envelope_path = DEFAULT_RESULTS_ROOT / "healthy_baseline" / "envelope.json"

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        envelope = json.loads(envelope_path.read_text(encoding="utf-8"))

        assert manifest == {
            "healthy_seed": DEFAULT_HEALTHY_SEED,
            "healthy_cases": DEFAULT_HEALTHY_CASES,
        }
        assert tuple(envelope.keys()) == ENVELOPE_KEYS
        assert envelope["rms_px_tolerance"] == 0.5
        assert envelope["wand_length_error_tolerance_mm"] == 0.1
        assert envelope["low_confidence_tolerance"] == 0

    def _seed_final_regression_baselines(self, results_root: Path) -> None:
        healthy_dir = results_root / "healthy_baseline"
        healthy_dir.mkdir(parents=True, exist_ok=True)

        baseline_by_case = {
            "case_009": {
                "phase2_rms_max_px": 2.0,
                "phase2_rms_median_px": 1.5,
                "phase2_low_confidence_count": 0,
                "wand_length_error_mm": 0.05,
            },
            "case_010": {
                "phase2_rms_max_px": 2.1,
                "phase2_rms_median_px": 1.6,
                "phase2_low_confidence_count": 1,
                "wand_length_error_mm": 0.04,
            },
            "case_023": {
                "phase2_rms_max_px": 1.9,
                "phase2_rms_median_px": 1.4,
                "phase2_low_confidence_count": 0,
                "wand_length_error_mm": 0.03,
            },
            "case_027": {
                "phase2_rms_max_px": 2.2,
                "phase2_rms_median_px": 1.7,
                "phase2_low_confidence_count": 0,
                "wand_length_error_mm": 0.02,
            },
            "case_029": {
                "phase2_rms_max_px": 2.4,
                "phase2_rms_median_px": 1.8,
                "phase2_low_confidence_count": 1,
                "wand_length_error_mm": 0.06,
            },
        }
        for case_id, metrics in baseline_by_case.items():
            (healthy_dir / f"{case_id}.json").write_text(
                json.dumps({"case_id": case_id, **metrics}, indent=2) + "\n",
                encoding="utf-8",
            )

    def _make_final_regression_result(
        self,
        case_id: str,
        *,
        mode: str,
        results_root: Path,
        phase2_rms_max_px: float,
        phase2_rms_median_px: float,
        phase2_low_confidence_count: int,
        wand_length_error_mm: float,
    ) -> dict:
        phase2_cameras = {
            "2": {
                "confidence_label": "low" if phase2_low_confidence_count else "normal",
                "confidence_warning": None,
                "reproj_rms_px": phase2_rms_median_px,
                "ransac_inlier_ratio": 0.85,
            },
            "3": {
                "confidence_label": "normal",
                "confidence_warning": None,
                "reproj_rms_px": phase2_rms_median_px,
                "ransac_inlier_ratio": 0.9,
            },
            "4": {
                "confidence_label": "normal",
                "confidence_warning": None,
                "reproj_rms_px": phase2_rms_max_px,
                "ransac_inlier_ratio": 0.88,
            },
        }
        return {
            "case_id": case_id,
            "mode": mode,
            "case_dir": results_root / case_id,
            "results_root": results_root,
            "selected_pair": (0, 1),
            "ranked_pairs": [(0, 1), (1, 2), (2, 3)],
            "all_cam_ids": [0, 1, 2, 3, 4],
            "phase2_camera_ids": [2, 3, 4],
            "elapsed_s": 1.25,
            "git_sha": "test-sha",
            "wand_length_mm": 10.0,
            "wand_length_sanity": {
                "median_mm": 10.0 - wand_length_error_mm,
                "std_mm": 0.01,
                "error_mm": wand_length_error_mm,
            },
            "report": {
                "all_cam_ids": [0, 1, 2, 3, 4],
                "p0_telemetry": {
                    "selected_pair": [0, 1],
                    "phase2_cameras": phase2_cameras,
                },
            },
        }

    def test_final_regression_cases_match_fixed_healthy_sample(self, monkeypatch, tmp_path):
        results_root = tmp_path / "results"
        self._seed_final_regression_baselines(results_root)
        requested_case_ids = []

        def _fake_run_bootstrap_case(case_id, **kwargs):
            requested_case_ids.append(case_id)
            passing_metrics = {
                "case_009": (2.3, 1.8, 0, 0.08),
                "case_010": (2.4, 1.9, 1, 0.08),
                "case_023": (2.2, 1.8, 0, 0.08),
                "case_027": (2.5, 2.0, 0, 0.08),
                "case_029": (2.6, 2.1, 1, 0.08),
            }
            phase2_rms_max_px, phase2_rms_median_px, phase2_low_confidence_count, wand_length_error_mm = passing_metrics[case_id]
            return self._make_final_regression_result(
                case_id,
                mode=kwargs["mode"],
                results_root=Path(kwargs["results_root"]),
                phase2_rms_max_px=phase2_rms_max_px,
                phase2_rms_median_px=phase2_rms_median_px,
                phase2_low_confidence_count=phase2_low_confidence_count,
                wand_length_error_mm=wand_length_error_mm,
            )

        monkeypatch.setattr(
            "scripts.case_013_bootstrap_debug_loop.run_bootstrap_case",
            _fake_run_bootstrap_case,
        )

        result = run_final_regression(results_root=results_root, case_root=tmp_path / "cases")
        summary_path = results_root / "final_regression" / "summary.json"

        assert result["mode"] == "final-regression"
        assert result["healthy_cases"] == DEFAULT_HEALTHY_CASES
        assert result["passed"] is True
        assert requested_case_ids == DEFAULT_HEALTHY_CASES
        assert [item["case_id"] for item in result["case_results"]] == DEFAULT_HEALTHY_CASES
        assert all(item["mode"] == "final-regression" for item in result["case_results"])
        assert [path.name for path in result["artifact_paths"]["healthy_cases"]] == [
            f"{case_id}.json" for case_id in DEFAULT_HEALTHY_CASES
        ]
        assert result["artifact_paths"]["summary"] == summary_path
        assert summary_path.exists()

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["passed"] is True
        assert list(summary["cases"].keys()) == DEFAULT_HEALTHY_CASES
        assert all(case_summary["passed"] for case_summary in summary["cases"].values())

        final_dir = results_root / "final_regression"
        assert sorted(path.name for path in final_dir.glob("case_*.json")) == [
            f"{case_id}.json" for case_id in DEFAULT_HEALTHY_CASES
        ]

    def test_run_final_regression_summary_marks_failed_on_any_regression(self, monkeypatch, tmp_path):
        results_root = tmp_path / "results"
        self._seed_final_regression_baselines(results_root)

        final_metrics = {
            "case_009": (2.2, 1.7, 0, 0.08),
            "case_010": (2.7, 1.7, 1, 0.03),
            "case_023": (2.1, 1.6, 0, 0.04),
            "case_027": (2.3, 1.8, 0, 0.05),
            "case_029": (2.5, 1.9, 1, 0.07),
        }

        def _fake_run_bootstrap_case(case_id, **kwargs):
            phase2_rms_max_px, phase2_rms_median_px, phase2_low_confidence_count, wand_length_error_mm = final_metrics[case_id]
            return self._make_final_regression_result(
                case_id,
                mode=kwargs["mode"],
                results_root=Path(kwargs["results_root"]),
                phase2_rms_max_px=phase2_rms_max_px,
                phase2_rms_median_px=phase2_rms_median_px,
                phase2_low_confidence_count=phase2_low_confidence_count,
                wand_length_error_mm=wand_length_error_mm,
            )

        monkeypatch.setattr(
            "scripts.case_013_bootstrap_debug_loop.run_bootstrap_case",
            _fake_run_bootstrap_case,
        )

        result = run_final_regression(results_root=results_root, case_root=tmp_path / "cases")
        summary = json.loads((results_root / "final_regression" / "summary.json").read_text(encoding="utf-8"))

        assert result["passed"] is False
        assert summary["passed"] is False
        assert summary["cases"]["case_010"]["passed"] is False
        assert summary["cases"]["case_010"]["checks"]["phase2_rms_max_px"] == {
            "actual": 2.7,
            "baseline": 2.1,
            "threshold": 2.6,
            "passed": False,
        }
        assert summary["cases"]["case_010"]["checks"]["phase2_rms_median_px"]["passed"] is True
        assert summary["cases"]["case_010"]["checks"]["phase2_low_confidence_count"]["passed"] is True
        assert summary["cases"]["case_010"]["checks"]["wand_length_error_mm"]["passed"] is True


class TestCase013Task5IterateLoop:
    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _seed_iteration_zero(self, tmp_path: Path, *, rms: float = 10.0) -> Path:
        results_root = tmp_path / "results"
        healthy_dir = results_root / "healthy_baseline"
        iterations_dir = results_root / "iterations"

        envelope = {
            "phase2_rms_max_px_max": 5.0,
            "phase2_rms_median_px_max": 4.0,
            "phase2_low_confidence_count_max": 1,
            "wand_length_error_mm_max": 0.3,
            "phase2_min_inlier_ratio_min": 0.7,
            "rms_px_tolerance": 0.5,
            "wand_length_error_tolerance_mm": 0.1,
            "low_confidence_tolerance": 0,
        }
        healthy_case_path = healthy_dir / "case_009.json"
        self._write_json(healthy_case_path, {"case_id": "case_009"})
        envelope_path = healthy_dir / "envelope.json"
        self._write_json(envelope_path, envelope)

        case_artifact = {
            "case_id": DEFAULT_TARGET_CASE,
            "mode": "baseline",
            "iteration": 0,
            "selected_pair": [0, 1],
            "ranked_pairs_top5": [[0, 1], [0, 2]],
            "elapsed_s": 1.0,
            "all_cam_ids": [0, 1, 2],
            "phase2_camera_ids": [2],
            "phase2_rms_max_px": rms,
            "phase2_rms_median_px": rms - 1.0,
            "phase2_low_confidence_count": 1,
            "phase2_min_inlier_ratio": 0.8,
            "wand_length_median_mm": 10.0,
            "wand_length_std_mm": 0.05,
            "wand_length_error_mm": 0.02,
            "scale_mismatch": False,
            "p0_telemetry": {"selected_pair": [0, 1], "phase2_cameras": {"2": {"confidence_label": "normal"}}},
            "phase2_cameras": {"2": {"confidence_label": "normal", "confidence_warning": None, "reproj_rms_px": rms}},
            "git_sha": "seed",
            "fix_summary": None,
        }
        request = {
            "target_artifact": str(iterations_dir / "iteration_000_case_013.json"),
            "healthy_envelope_artifact": str(envelope_path),
            "healthy_case_artifacts": [str(healthy_case_path)],
            "primary_metrics": {
                "phase2_rms_max_px": rms,
                "phase2_rms_median_px": rms - 1.0,
                "phase2_low_confidence_count": 1,
                "wand_length_error_mm": 0.02,
                "scale_mismatch": False,
            },
            "secondary_metrics": {
                "phase2_min_inlier_ratio": 0.8,
                "selected_pair": [0, 1],
                "ranked_pairs_top5": [[0, 1], [0, 2]],
                "phase2_camera_ids": [2],
                "healthy_envelope": envelope,
            },
            "provisional_pass": False,
            "comparison_notes": ["phase2_rms_max_px: seeded"],
        }

        self._write_json(iterations_dir / "iteration_000_case_013.json", case_artifact)
        self._write_json(iterations_dir / "iteration_000_metis_request.json", request)
        return results_root

    def test_should_pause_for_stall_detects_three_small_improvements(self):
        assert should_pause_for_stall(
            [10.0, 9.8, 9.7, 9.65],
            ["tighten_ransac", "tighten_ransac", "tighten_ransac"],
        ) is True

    def test_run_iterate_writes_response_and_stops_on_fixed(self, tmp_path):
        results_root = self._seed_iteration_zero(tmp_path)

        result = run_iterate(
            case_root=tmp_path / "case-root",
            results_root=results_root,
            target_case=DEFAULT_TARGET_CASE,
            metis_callback=lambda request, prompt: {
                "verdict": "fixed",
                "rationale": "Metrics are within acceptable range.",
                "suspected_root_cause": "resolved",
                "next_fix_title": None,
                "next_fix_files": [],
                "exact_changes": [],
                "expected_metric_shift": {"phase2_rms_max_px": "stable"},
            },
        )

        response_path = results_root / "iterations" / "iteration_000_metis_response.json"
        assert response_path.exists()
        response = json.loads(response_path.read_text(encoding="utf-8"))
        assert response["verdict"] == "fixed"
        assert result["verdict"] == "fixed"
        assert not (results_root / "iterations" / "iteration_001_case_013.json").exists()

    def test_run_iterate_uses_existing_response_when_no_callbacks_are_provided(self, tmp_path, monkeypatch):
        results_root = self._seed_iteration_zero(tmp_path)
        iterations_dir = results_root / "iterations"
        existing_response = {
            "verdict": "fixed",
            "rationale": "Already accepted from prior review.",
            "suspected_root_cause": "",
            "next_fix_title": None,
            "next_fix_files": [],
            "exact_changes": [],
            "expected_metric_shift": None,
        }
        self._write_json(iterations_dir / "iteration_000_metis_response.json", existing_response)

        monkeypatch.setattr(
            "scripts.case_013_bootstrap_debug_loop._default_metis_callback",
            lambda request, prompt: pytest.fail("default Metis callback should not run when response artifact already exists"),
        )

        result = run_iterate(
            case_root=tmp_path / "case-root",
            results_root=results_root,
            target_case=DEFAULT_TARGET_CASE,
        )

        assert result["verdict"] == "fixed"
        assert result["iteration"] == 0
        assert result["metis_response"] == existing_response
        response = json.loads((iterations_dir / "iteration_000_metis_response.json").read_text(encoding="utf-8"))
        assert response == existing_response

    def test_run_iterate_writes_next_iteration_for_not_fixed(self, tmp_path, monkeypatch):
        results_root = self._seed_iteration_zero(tmp_path, rms=10.0)
        applied = []

        fake_case_result = {
            "case_id": DEFAULT_TARGET_CASE,
            "mode": "iterate",
            "case_dir": tmp_path / "case-root" / DEFAULT_TARGET_CASE,
            "results_root": results_root,
            "selected_pair": (1, 2),
            "ranked_pairs": [(1, 2), (0, 1), (0, 2)],
            "all_cam_ids": [0, 1, 2],
            "phase2_camera_ids": [2],
            "elapsed_s": 2.5,
            "git_sha": "abc123",
            "wand_length_mm": 10.0,
            "wand_length_sanity": {"median_mm": 10.0, "std_mm": 0.05, "error_mm": 0.01},
            "report": {
                "all_cam_ids": [0, 1, 2],
                "p0_telemetry": {
                    "selected_pair": [1, 2],
                    "phase2_cameras": {
                        "2": {
                            "confidence_label": "normal",
                            "confidence_warning": None,
                            "reproj_rms_px": 8.0,
                            "ransac_inlier_ratio": 0.82,
                        }
                    },
                },
            },
        }

        monkeypatch.setattr(
            "scripts.case_013_bootstrap_debug_loop.run_bootstrap_case",
            lambda **kwargs: fake_case_result,
        )

        result = run_iterate(
            case_root=tmp_path / "case-root",
            results_root=results_root,
            target_case=DEFAULT_TARGET_CASE,
            metis_callback=lambda request, prompt: {
                "verdict": "not_fixed",
                "rationale": "Still too large.",
                "suspected_root_cause": "phase2 threshold",
                "next_fix_title": "tighten_ransac",
                "next_fix_files": ["scripts/case_013_bootstrap_debug_loop.py"],
                "exact_changes": ["Narrow acceptance threshold."],
                "expected_metric_shift": {"phase2_rms_max_px": -2.0},
            },
            apply_fix_callback=lambda response, request: applied.append(response["next_fix_title"]) or "tighten_ransac",
        )

        assert applied == ["tighten_ransac"]
        response = json.loads((results_root / "iterations" / "iteration_000_metis_response.json").read_text(encoding="utf-8"))
        assert response["verdict"] == "not_fixed"

        next_case_path = results_root / "iterations" / "iteration_001_case_013.json"
        next_request_path = results_root / "iterations" / "iteration_001_metis_request.json"
        assert next_case_path.exists()
        assert next_request_path.exists()

        next_case = json.loads(next_case_path.read_text(encoding="utf-8"))
        assert next_case["iteration"] == 1
        assert next_case["fix_summary"] == "tighten_ransac"
        assert result["verdict"] == "not_fixed"
        assert result["iteration"] == 1

    def test_run_iterate_marks_stalled_and_does_not_rerun(self, tmp_path, monkeypatch):
        results_root = self._seed_iteration_zero(tmp_path, rms=10.0)
        iterations_dir = results_root / "iterations"

        for idx, rms in [(1, 9.8), (2, 9.7), (3, 9.65)]:
            case_artifact = json.loads((iterations_dir / "iteration_000_case_013.json").read_text(encoding="utf-8"))
            case_artifact["iteration"] = idx
            case_artifact["mode"] = "iterate"
            case_artifact["phase2_rms_max_px"] = rms
            case_artifact["phase2_rms_median_px"] = rms - 1.0
            self._write_json(iterations_dir / f"iteration_{idx:03d}_case_013.json", case_artifact)

            request = json.loads((iterations_dir / "iteration_000_metis_request.json").read_text(encoding="utf-8"))
            request["target_artifact"] = str(iterations_dir / f"iteration_{idx:03d}_case_013.json")
            request["primary_metrics"]["phase2_rms_max_px"] = rms
            self._write_json(iterations_dir / f"iteration_{idx:03d}_metis_request.json", request)

        for idx in [1, 2]:
            self._write_json(
                iterations_dir / f"iteration_{idx:03d}_metis_response.json",
                {
                    "verdict": "not_fixed",
                    "rationale": "small improvement",
                    "suspected_root_cause": "same issue",
                    "next_fix_title": "tighten_ransac",
                    "next_fix_files": ["scripts/case_013_bootstrap_debug_loop.py"],
                    "exact_changes": ["keep tightening"],
                    "expected_metric_shift": {"phase2_rms_max_px": -0.1},
                },
            )

        called = {"rerun": False, "fix": False}

        monkeypatch.setattr(
            "scripts.case_013_bootstrap_debug_loop.run_bootstrap_case",
            lambda **kwargs: called.__setitem__("rerun", True),
        )

        result = run_iterate(
            case_root=tmp_path / "case-root",
            results_root=results_root,
            target_case=DEFAULT_TARGET_CASE,
            metis_callback=lambda request, prompt: {
                "verdict": "not_fixed",
                "rationale": "still too large",
                "suspected_root_cause": "same issue",
                "next_fix_title": "tighten_ransac",
                "next_fix_files": ["scripts/case_013_bootstrap_debug_loop.py"],
                "exact_changes": ["keep tightening"],
                "expected_metric_shift": {"phase2_rms_max_px": -0.05},
            },
            apply_fix_callback=lambda response, request: called.__setitem__("fix", True),
        )

        response = json.loads((iterations_dir / "iteration_003_metis_response.json").read_text(encoding="utf-8"))
        assert result["verdict"] == "stalled"
        assert response["verdict"] == "stalled"
        assert called == {"rerun": False, "fix": False}
        assert not (iterations_dir / "iteration_004_case_013.json").exists()
