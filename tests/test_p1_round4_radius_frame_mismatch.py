# pyright: reportMissingImports=false
"""
Regression tests for P1 Round-4 radius estimation frame mismatch (H1).

Root cause: After BA optimization, cam_params moves to post-alignment frame
while caller-side X_A_scaled / X_B_scaled remain in pre-alignment (bootstrap)
frame.  The radius estimator computes X_cam = R @ X_world + tvec, where
R, tvec are post-alignment but X_world is pre-alignment → wrong Zc → inflated
R_mm = r_px * Zc / f.

DESIGN (red-phase):
    - test_radius_estimation_frame_consistent: PASSES (green baseline)
    - test_radius_estimation_frame_mismatch_inflates: FAILS before fix (red anchor)
"""

import sys
from pathlib import Path

import numpy as np
import cv2
import pytest

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.camera_calibration.wand_calibration.refraction_wand_calibrator import (
    RefractiveWandCalibrator,
)


# ═══════════════════════════════════════════════════════════════════════════
# Minimal mock for base_calibrator (only fields touched by __init__)
# ═══════════════════════════════════════════════════════════════════════════

class _MockBaseCalibrator:
    """Minimal stand-in for WandCalibrator.  RefractiveWandCalibrator.__init__
    only reads self.base and attaches logger/reporter — no heavy deps."""
    pass


def _make_calibrator():
    """Create a RefractiveWandCalibrator with a trivial mock base."""
    return RefractiveWandCalibrator(_MockBaseCalibrator())


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic data builders
# ═══════════════════════════════════════════════════════════════════════════

def _identity_cam_params(focal: float = 5000.0) -> np.ndarray:
    """Camera at world origin, looking down +Z.

    cam_params layout: [rvec(3), tvec(3), f, cx, cy, k1, k2]
    Identity rotation (rvec=[0,0,0]), zero translation → X_cam == X_world.
    """
    return np.array([
        0.0, 0.0, 0.0,   # rvec  (identity)
        0.0, 0.0, 0.0,   # tvec  (origin)
        focal,            # f
        0.0, 0.0,         # cx, cy
        0.0, 0.0,         # k1, k2
    ], dtype=np.float64)


def _build_consistent_scenario(
    n_frames: int = 20,
    z_distance: float = 500.0,
    focal: float = 5000.0,
    r_px_small: float = 15.0,
    r_px_large: float = 20.0,
    seed: int = 42,
):
    """Build fully consistent (same-frame) cam_params + 3D points + radii.

    Points are at ~z_distance mm in front of a single camera at origin.
    Expected small-sphere radius: r_px_small * z_distance / focal
    Expected large-sphere radius: r_px_large * z_distance / focal
    """
    rng = np.random.default_rng(seed)
    cam_id = 0

    cam_params = {cam_id: _identity_cam_params(focal)}

    X_A = {}  # small sphere
    X_B = {}  # large sphere
    radii_small = {}
    radii_large = {}

    for fid in range(n_frames):
        # Slight random jitter so points aren't all identical
        jitter = rng.uniform(-5.0, 5.0, size=3)
        jitter[2] = 0.0  # keep Z stable
        X_A[fid] = np.array([0.0, 0.0, z_distance]) + jitter
        X_B[fid] = np.array([5.0, 0.0, z_distance]) + jitter  # 5mm apart

        radii_small[fid] = {cam_id: r_px_small}
        radii_large[fid] = {cam_id: r_px_large}

    dataset = {
        'radii_small': radii_small,
        'radii_large': radii_large,
    }
    return cam_params, X_A, X_B, dataset


def _build_mismatched_scenario(
    n_frames: int = 20,
    z_distance: float = 500.0,
    focal: float = 5000.0,
    r_px_small: float = 15.0,
    r_px_large: float = 20.0,
    seed: int = 42,
):
    """Simulate the H1 bug: points in bootstrap frame, cam_params in post-BA frame.

    - X_A_scaled, X_B_scaled are in the bootstrap frame (camera at origin).
    - cam_params is in post-alignment frame: camera has been rotated 15° and
      translated by [200, 50, 300], simulating a BA alignment shift.

    With mismatched frames, X_cam = R_post @ X_bootstrap + tvec_post yields
    a *very* different Zc, producing inflated radius estimates.

    The rotation + translation are chosen so that Zc remains positive (not
    filtered out by the Zc > 10 guard) but is far from the true ~500mm,
    producing clearly inflated radii.
    """
    rng = np.random.default_rng(seed)
    cam_id = 0

    # --- Post-BA camera extrinsics (mismatched with bootstrap points) ---
    # 15° rotation around Y axis (moderate — keeps Zc positive but shifted)
    angle_y = np.pi / 12  # 15 degrees
    rvec_post = np.array([0.0, angle_y, 0.0], dtype=np.float64)
    # Translation adding 300mm along Z → inflates Zc significantly
    tvec_post = np.array([200.0, 50.0, 300.0], dtype=np.float64)

    cam_params_mismatched = {
        cam_id: np.array([
            rvec_post[0], rvec_post[1], rvec_post[2],
            tvec_post[0], tvec_post[1], tvec_post[2],
            focal,
            0.0, 0.0,     # cx, cy
            0.0, 0.0,     # k1, k2
        ], dtype=np.float64)
    }

    # --- Points stay in bootstrap frame (same as consistent scenario) ---
    X_A = {}
    X_B = {}
    radii_small = {}
    radii_large = {}

    for fid in range(n_frames):
        jitter = rng.uniform(-5.0, 5.0, size=3)
        jitter[2] = 0.0
        X_A[fid] = np.array([0.0, 0.0, z_distance]) + jitter
        X_B[fid] = np.array([5.0, 0.0, z_distance]) + jitter

        radii_small[fid] = {cam_id: r_px_small}
        radii_large[fid] = {cam_id: r_px_large}

    dataset = {
        'radii_small': radii_small,
        'radii_large': radii_large,
    }
    return cam_params_mismatched, X_A, X_B, dataset


# ═══════════════════════════════════════════════════════════════════════════
# Regression tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRadiusEstimationFrameMismatch:
    """
    P1 Round-4 regression: radius estimation must use frame-consistent
    cam_params and 3D points.  When frames mismatch (post-BA cam_params +
    pre-BA points), radius estimates are physically implausible.
    """

    # --- GREEN: frame-consistent scenario always passes ---

    def test_radius_estimation_frame_consistent(self):
        """Frame-consistent cam_params + points → physically plausible radii.

        Expected: R_mm_small ≈ 15 * 500 / 5000 = 1.5 mm
                  R_mm_large ≈ 20 * 500 / 5000 = 2.0 mm
        Tolerance: 1.0 – 2.5 mm for small sphere (generous for jitter).
        """
        calibrator = _make_calibrator()
        cam_params, X_A, X_B, dataset = _build_consistent_scenario()

        r_small, r_large = calibrator._estimate_and_log_sphere_radii(
            dataset, cam_params, X_A, X_B, tag="TEST_CONSISTENT", cams_cpp=None,
        )

        # Plausibility: small sphere should be ~1.5mm
        assert 1.0 <= r_small <= 2.5, (
            f"Frame-consistent small radius {r_small:.2f}mm outside plausible "
            f"range [1.0, 2.5]mm (expected ~1.5mm)"
        )

        # Plausibility: large sphere should be ~2.0mm
        assert 1.5 <= r_large <= 3.0, (
            f"Frame-consistent large radius {r_large:.2f}mm outside plausible "
            f"range [1.5, 3.0]mm (expected ~2.0mm)"
        )

    # --- RED-PHASE: frame-mismatch scenario must FAIL before fix ---

    def test_radius_estimation_frame_mismatch_inflates(self):
        """Frame-mismatched cam_params + points inflate radii beyond plausibility.

        This test FAILS before the fix is applied because the current code
        does not guard against the caller passing stale (bootstrap-frame)
        X_A_scaled with post-BA cam_params.

        Mechanism:
            X_cam = R_post @ X_bootstrap + tvec_post
            Zc = X_cam[2]  → very different from the true ~500mm
            R_mm = r_px * Zc / f  → inflated or collapsed

        The assertion checks that mismatched frames produce radii close
        to the consistent-frame result.  Before the fix, they don't —
        proving the bug.
        """
        calibrator = _make_calibrator()

        # Consistent baseline
        cam_params_ok, X_A, X_B, dataset = _build_consistent_scenario()
        r_small_ok, r_large_ok = calibrator._estimate_and_log_sphere_radii(
            dataset, cam_params_ok, X_A, X_B, tag="TEST_BASELINE", cams_cpp=None,
        )

        # Mismatched scenario (same points, wrong cam_params frame)
        cam_params_bad, X_A_m, X_B_m, dataset_m = _build_mismatched_scenario()
        r_small_bad, r_large_bad = calibrator._estimate_and_log_sphere_radii(
            dataset_m, cam_params_bad, X_A_m, X_B_m, tag="TEST_MISMATCH", cams_cpp=None,
        )

        # This assertion SHOULD fail before the fix: mismatched frames
        # inflate radii far beyond the consistent result.
        diff_small = abs(r_small_bad - r_small_ok)
        assert diff_small < 0.5, (
            f"Frame mismatch inflated small-sphere radius from "
            f"{r_small_ok:.2f}mm to {r_small_bad:.2f}mm "
            f"(difference={diff_small:.2f}mm > 0.5mm). "
            f"Root cause H1 confirmed: stale caller-side points after BA alignment."
        )

        diff_large = abs(r_large_bad - r_large_ok)
        assert diff_large < 0.5, (
            f"Frame mismatch inflated large-sphere radius from "
            f"{r_large_ok:.2f}mm to {r_large_bad:.2f}mm "
            f"(difference={diff_large:.2f}mm > 0.5mm). "
            f"Root cause H1 confirmed: stale caller-side points after BA alignment."
        )

    # --- GREEN-AFTER-FIX: point-sync restores frame consistency ---

    def test_point_sync_fixes_frame_mismatch(self):
        """After syncing points to post-BA frame, radius estimation is plausible.

        This test simulates the H1 fix applied in calibrate(): caller-side
        X_A_scaled is replaced with _bundle_points coordinates (post-alignment
        frame) before passing to _estimate_and_log_sphere_radii().

        The key insight: once points and cam_params share the same coordinate
        frame, X_cam = R @ X_world + tvec yields the correct Zc regardless of
        which frame that is.

        PASSES after the fix.  Complements test_radius_estimation_frame_mismatch_inflates
        which stays as a permanent mechanism-demonstration (red) test.
        """
        calibrator = _make_calibrator()

        n_frames = 20
        focal = 5000.0
        r_px_small = 15.0
        r_px_large = 20.0
        z_distance = 500.0

        # --- Post-BA camera extrinsics (same as _build_mismatched_scenario) ---
        angle_y = np.pi / 12   # 15°
        rvec_post = np.array([0.0, angle_y, 0.0])
        tvec_post = np.array([200.0, 50.0, 300.0])
        R_post, _ = cv2.Rodrigues(rvec_post)

        cam_id = 0
        cam_params_post = {
            cam_id: np.array([
                rvec_post[0], rvec_post[1], rvec_post[2],
                tvec_post[0], tvec_post[1], tvec_post[2],
                focal, 0.0, 0.0, 0.0, 0.0,
            ], dtype=np.float64)
        }

        # Camera center in world: C = -R^T @ tvec
        C_world = -R_post.T @ tvec_post
        # Camera look-direction in world: R^T @ [0,0,1]
        look_dir = R_post.T @ np.array([0.0, 0.0, 1.0])

        # --- Build points in the POST-alignment world frame ---
        # Each point is ~500mm in front of the camera (in camera space).
        rng = np.random.default_rng(42)
        X_A_synced = {}
        X_B_synced = {}
        radii_small = {}
        radii_large = {}

        for fid in range(n_frames):
            jitter = rng.uniform(-5.0, 5.0, size=3)
            jitter[2] = 0.0  # keep Z_cam stable
            pt_a = C_world + z_distance * look_dir + jitter
            pt_b = pt_a + np.array([5.0, 0.0, 0.0])
            X_A_synced[fid] = pt_a
            X_B_synced[fid] = pt_b
            radii_small[fid] = {cam_id: r_px_small}
            radii_large[fid] = {cam_id: r_px_large}

        dataset_synced = {
            'radii_small': radii_small,
            'radii_large': radii_large,
        }

        r_small, r_large = calibrator._estimate_and_log_sphere_radii(
            dataset_synced, cam_params_post, X_A_synced, X_B_synced,
            tag="TEST_SYNCED", cams_cpp=None,
        )

        # Expected: r_px * z_distance / focal = 15 * 500 / 5000 = 1.5mm
        assert 1.0 <= r_small <= 2.5, (
            f"Post-sync small-sphere radius {r_small:.2f}mm outside plausible "
            f"range [1.0, 2.5]mm (expected ~1.5mm)"
        )
        assert 1.5 <= r_large <= 3.0, (
            f"Post-sync large-sphere radius {r_large:.2f}mm outside plausible "
            f"range [1.5, 3.0]mm (expected ~2.0mm)"
        )
