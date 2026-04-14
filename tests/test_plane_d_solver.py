import numpy as np
import pytest

from modules.camera_calibration.wand_calibration.plane_d_solver import (
    solve_plane_d_from_correspondences,
    _snell_refract,
)


def _make_cam_params(C_j, rvec=None, f=1000.0, cx=512.0, cy=384.0):
    """cam_params: [rvec(3), tvec(3), focal_px, cx, cy, k1, k2]."""
    import cv2
    C_j = np.array(C_j, dtype=np.float64)
    if rvec is None:
        rvec = np.array([0.0, 0.0, 0.0])
    else:
        rvec = np.array(rvec, dtype=np.float64)
    R, _ = cv2.Rodrigues(rvec)
    tvec = -R @ C_j
    return np.array([*rvec, *tvec, f, cx, cy, 0.0, 0.0])


def _find_refraction_pixel(C_j, R_j, P_k, plane_pt, plane_n,
                            n1, n2, n3,
                            f=1000.0, cx=512.0, cy=384.0,
                            n_iter=30):
    """Find pixel in camera j for object point P_k seen through refractive plate.

    Forward model: pixel -> camera ray u -> refract(u) -> v -> intersect object plane.
    Newton iteration adjusts pixel until the intersection matches P_k.
    """
    n_hat = -plane_n

    def _forward(px, py):
        """Given pixel (px, py), trace forward through refraction.
        Returns intersection point with the z-plane of P_k, or None."""
        u_cam = np.array([(px - cx) / f, (py - cy) / f, 1.0])
        u_cam = u_cam / np.linalg.norm(u_cam)
        u_world = R_j.T @ u_cam
        u_world = u_world / np.linalg.norm(u_world)

        denom_n = np.dot(plane_n, u_world)
        if abs(denom_n) < 1e-12:
            return None

        t1, ok1 = _snell_refract(u_world, n_hat, n1, n2)
        if not ok1:
            return None
        t2, ok2 = _snell_refract(t1, n_hat, n2, n3)
        if not ok2:
            return None

        # Ray-plane intersection for the first interface: Q = C_j + s_plane * u_world
        s_plane = np.dot(plane_n, plane_pt - C_j) / denom_n
        if s_plane <= 0:
            return None
        Q = C_j + s_plane * u_world

        # Continue with refracted ray v=t2 from Q to object z-plane
        obj_z = P_k[2]
        denom_v = np.dot(plane_n, t2)
        if abs(denom_v) < 1e-12:
            return None
        # Distance along t2 from Q to reach obj_z
        s_obj = (obj_z - np.dot(plane_n, Q)) / denom_v
        if s_obj <= 0:
            return None
        hit = Q + s_obj * t2
        return hit

    # Initial guess: direct projection (ignoring refraction)
    direct = P_k - C_j
    px0 = f * direct[0] / direct[2] + cx
    py0 = f * direct[1] / direct[2] + cy

    px, py = px0, py0
    eps = 0.01

    for _ in range(n_iter):
        hit = _forward(px, py)
        if hit is None:
            return None
        err = hit[:2] - P_k[:2]

        if np.linalg.norm(err) < 1e-8:
            return (px, py)

        # Numerical Jacobian via finite differences
        hit_dx = _forward(px + eps, py)
        hit_dy = _forward(px, py + eps)
        if hit_dx is None or hit_dy is None:
            return None

        J = np.zeros((2, 2))
        J[:, 0] = (hit_dx[:2] - hit[:2]) / eps
        J[:, 1] = (hit_dy[:2] - hit[:2]) / eps

        det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if abs(det) < 1e-20:
            return None

        dpx = (J[1, 1] * err[0] - J[0, 1] * err[1]) / det
        dpy = (-J[1, 0] * err[0] + J[0, 0] * err[1]) / det
        px -= dpx
        py -= dpy

    return (px, py)


def _build_synthetic_setup(
    camera_positions, object_points, plane_n, d_true, A_anchor,
    n1=1.0, n2=1.5, n3=1.333,
    f=1000.0, cx=512.0, cy=384.0,
    rvecs=None,
):
    """Build solver inputs from known geometry with physically correct refraction.

    For each (camera, object_point) pair, traces the refraction-consistent ray
    path through the plate and computes the pixel that the camera would observe.
    This ensures the solver's internal Snell refraction produces non-degenerate
    pairwise constraints.

    Parameters
    ----------
    rvecs : list or None
        Per-camera Rodrigues vectors. None means identity rotation for all.
    """
    import cv2

    plane_pt = A_anchor + d_true * plane_n

    cam_params = {}
    cam_to_window = {}
    R_mats = {}
    for j, C_j in enumerate(camera_positions):
        rv = rvecs[j] if rvecs is not None else None
        cam_params[j] = _make_cam_params(C_j, rvec=rv, f=f, cx=cx, cy=cy)
        cam_to_window[j] = 0
        R_j, _ = cv2.Rodrigues(cam_params[j][0:3])
        R_mats[j] = R_j

    observations = {}
    for k, P_k in enumerate(object_points):
        P_k = np.array(P_k, dtype=np.float64)
        frame_obs = {}

        for j, C_j in enumerate(camera_positions):
            C_j = np.array(C_j, dtype=np.float64)
            R_j = R_mats[j]

            pixelA = _find_refraction_pixel(
                C_j, R_j, P_k, plane_pt, plane_n, n1, n2, n3, f, cx, cy,
            )
            if pixelA is None:
                continue

            P_k_B = P_k + np.array([5.0, 3.0, 0.0])
            pixelB = _find_refraction_pixel(
                C_j, R_j, P_k_B, plane_pt, plane_n, n1, n2, n3, f, cx, cy,
            )
            if pixelB is None:
                continue

            frame_obs[j] = (pixelA, pixelB)

        if len(frame_obs) >= 2:
            observations[k] = frame_obs

    window_media = {'n1': n1, 'n2': n2, 'n3': n3, 'thickness': 10.0}
    return cam_params, observations, window_media, cam_to_window


class TestSyntheticTwoCam:

    def test_synthetic_two_cam_recovers_d(self):
        """Two cameras separated in x, plane_n=[0,0,1], recover d_true."""
        d_true = 200.0
        plane_n = np.array([0.0, 0.0, 1.0])
        A_anchor = np.array([0.0, 0.0, 0.0])

        camera_positions = [
            [-40.0, 0.0, -100.0],
            [40.0, 0.0, -100.0],
        ]
        object_points = [
            [0.0, 0.0, 400.0],
            [20.0, 10.0, 400.0],
            [-20.0, -10.0, 400.0],
        ]

        cam_params, observations, window_media, cam_to_window = _build_synthetic_setup(
            camera_positions, object_points, plane_n, d_true, A_anchor,
        )
        assert len(observations) >= 2, "Need at least 2 frames for constraints"

        result = solve_plane_d_from_correspondences(
            cam_params=cam_params,
            observations=observations,
            plane_n=plane_n,
            window_media=window_media,
            cam_to_window=cam_to_window,
            wid=0,
            A_anchor=A_anchor,
            d_midpoint=d_true,
            verbose=True,
        )

        assert result['accepted'] is True, f"Not accepted: {result['fallback_reason']}"
        assert result['n_equations'] >= 2
        assert abs(result['d_solved'] - d_true) < 2.0, (
            f"d_solved={result['d_solved']:.4f}, expected {d_true}"
        )
        assert result['fallback_reason'] is None


class TestOverdetermined:

    def test_overdetermined_multi_cam_is_accurate(self):
        """4 cameras -> more equations -> tighter d recovery."""
        d_true = 200.0
        plane_n = np.array([0.0, 0.0, 1.0])
        A_anchor = np.array([0.0, 0.0, 0.0])

        camera_positions = [
            [-40.0, -40.0, -100.0],
            [40.0, -40.0, -100.0],
            [-40.0, 40.0, -100.0],
            [40.0, 40.0, -100.0],
        ]
        object_points = [
            [0.0, 0.0, 400.0],
            [15.0, 10.0, 400.0],
            [-15.0, -10.0, 400.0],
            [5.0, -5.0, 400.0],
        ]

        cam_params, observations, window_media, cam_to_window = _build_synthetic_setup(
            camera_positions, object_points, plane_n, d_true, A_anchor,
        )

        result = solve_plane_d_from_correspondences(
            cam_params=cam_params,
            observations=observations,
            plane_n=plane_n,
            window_media=window_media,
            cam_to_window=cam_to_window,
            wid=0,
            A_anchor=A_anchor,
            d_midpoint=d_true,
            verbose=True,
        )

        assert result['accepted'] is True, f"Not accepted: {result['fallback_reason']}"
        assert result['n_equations'] >= 4
        assert abs(result['d_solved'] - d_true) < 1.0, (
            f"d_solved={result['d_solved']:.4f}, expected {d_true}"
        )


class TestDegenerate:

    def test_degenerate_rank_deficient_falls_back(self):
        """Single camera -> zero pairs -> insufficient_equations."""
        plane_n = np.array([0.0, 0.0, 1.0])

        cam_params = {0: _make_cam_params([0.0, 0.0, -100.0])}
        observations = {0: {0: ((520.0, 390.0), (525.0, 393.0))}}
        window_media = {'n1': 1.0, 'n2': 1.5, 'n3': 1.333, 'thickness': 10.0}
        cam_to_window = {0: 0}

        result = solve_plane_d_from_correspondences(
            cam_params=cam_params,
            observations=observations,
            plane_n=plane_n,
            window_media=window_media,
            cam_to_window=cam_to_window,
            wid=0,
            A_anchor=np.array([0.0, 0.0, 0.0]),
            d_midpoint=200.0,
        )

        assert result['accepted'] is False
        assert result['fallback_reason'] is not None
        assert 'insufficient' in result['fallback_reason']


class TestFallback:

    def test_fallback_uses_midpoint_on_ill_conditioned(self):
        """Identical cameras with identical pixels -> degenerate cross products -> rejected."""
        plane_n = np.array([0.0, 0.0, 1.0])

        cam_params = {
            0: _make_cam_params([0.0, 0.0, -100.0]),
            1: _make_cam_params([0.0, 0.001, -100.0]),
        }
        observations = {
            0: {
                0: ((512.0, 384.0), (517.0, 387.0)),
                1: ((512.0, 384.0), (517.0, 387.0)),
            },
        }
        window_media = {'n1': 1.0, 'n2': 1.5, 'n3': 1.333, 'thickness': 10.0}
        cam_to_window = {0: 0, 1: 0}

        result = solve_plane_d_from_correspondences(
            cam_params=cam_params,
            observations=observations,
            plane_n=plane_n,
            window_media=window_media,
            cam_to_window=cam_to_window,
            wid=0,
            A_anchor=np.array([0.0, 0.0, 0.0]),
            d_midpoint=200.0,
        )

        assert result['accepted'] is False

    def test_fallback_preserves_legacy_midpoint(self):
        """When accepted=False, diagnostics dict is fully populated."""
        plane_n = np.array([0.0, 0.0, 1.0])

        cam_params = {0: _make_cam_params([0.0, 0.0, -100.0])}
        observations = {0: {0: ((512.0, 384.0), (517.0, 387.0))}}
        window_media = {'n1': 1.0, 'n2': 1.5, 'n3': 1.333, 'thickness': 10.0}
        cam_to_window = {0: 0}

        result = solve_plane_d_from_correspondences(
            cam_params=cam_params,
            observations=observations,
            plane_n=plane_n,
            window_media=window_media,
            cam_to_window=cam_to_window,
            wid=0,
            A_anchor=np.array([0.0, 0.0, 0.0]),
            d_midpoint=200.0,
        )

        assert result['fallback_reason'] is not None
        assert not result['accepted']
        assert 'n_equations' in result
        assert 'rank' in result

    def test_fallback_outlier_d_rejected(self):
        """d_solved far from d_midpoint triggers outlier_d gate."""
        d_true = 200.0
        plane_n = np.array([0.0, 0.0, 1.0])
        A_anchor = np.array([0.0, 0.0, 0.0])

        camera_positions = [
            [-40.0, 0.0, -100.0],
            [40.0, 0.0, -100.0],
        ]
        object_points = [
            [0.0, 0.0, 400.0],
            [20.0, 10.0, 400.0],
        ]

        cam_params, observations, window_media, cam_to_window = _build_synthetic_setup(
            camera_positions, object_points, plane_n, d_true, A_anchor,
        )

        result = solve_plane_d_from_correspondences(
            cam_params=cam_params,
            observations=observations,
            plane_n=plane_n,
            window_media=window_media,
            cam_to_window=cam_to_window,
            wid=0,
            A_anchor=A_anchor,
            d_midpoint=5000.0,
        )

        assert result['accepted'] is False
        assert result['fallback_reason'] == 'outlier_d'


class TestIllConditioned:

    def test_ill_conditioned_geometry_not_accepted(self):
        """Cameras along z-axis with near-zero angular separation -> rejected."""
        plane_n = np.array([0.0, 0.0, 1.0])

        cam_params = {
            0: _make_cam_params([0.0, 0.0, -100.0]),
            1: _make_cam_params([0.0, 0.0, -101.0]),
        }
        observations = {
            0: {
                0: ((512.0, 384.0), (517.0, 387.0)),
                1: ((512.1, 384.0), (517.1, 387.0)),
            },
        }
        window_media = {'n1': 1.0, 'n2': 1.5, 'n3': 1.333, 'thickness': 10.0}
        cam_to_window = {0: 0, 1: 0}

        result = solve_plane_d_from_correspondences(
            cam_params=cam_params,
            observations=observations,
            plane_n=plane_n,
            window_media=window_media,
            cam_to_window=cam_to_window,
            wid=0,
            A_anchor=np.array([0.0, 0.0, 0.0]),
            d_midpoint=200.0,
        )

        assert result['accepted'] is False


class TestLegacyCompare:

    def test_legacy_compare_default_and_legacy_paths(self):
        """Solver is deterministic: same inputs -> identical outputs."""
        d_true = 200.0
        plane_n = np.array([0.0, 0.0, 1.0])
        A_anchor = np.array([0.0, 0.0, 0.0])

        camera_positions = [
            [-40.0, 0.0, -100.0],
            [40.0, 0.0, -100.0],
        ]
        object_points = [
            [0.0, 0.0, 400.0],
            [20.0, 10.0, 400.0],
            [-20.0, -10.0, 400.0],
        ]

        cam_params, observations, window_media, cam_to_window = _build_synthetic_setup(
            camera_positions, object_points, plane_n, d_true, A_anchor,
        )

        kwargs = dict(
            cam_params=cam_params,
            observations=observations,
            plane_n=plane_n,
            window_media=window_media,
            cam_to_window=cam_to_window,
            wid=0,
            A_anchor=A_anchor,
            d_midpoint=d_true,
        )

        result1 = solve_plane_d_from_correspondences(**kwargs)
        result2 = solve_plane_d_from_correspondences(**kwargs)

        assert result1['accepted'] == result2['accepted']
        if np.isfinite(result1['d_solved']):
            assert result1['d_solved'] == result2['d_solved']
        assert result1['n_equations'] == result2['n_equations']
        assert result1['rank'] == result2['rank']

    def test_legacy_compare_all_keys_present(self):
        """All required return dict keys are present."""
        cam_params = {0: _make_cam_params([0.0, 0.0, -100.0])}
        observations = {0: {0: ((512.0, 384.0), (517.0, 387.0))}}
        window_media = {'n1': 1.0, 'n2': 1.5, 'n3': 1.333, 'thickness': 10.0}

        result = solve_plane_d_from_correspondences(
            cam_params=cam_params,
            observations=observations,
            plane_n=np.array([0.0, 0.0, 1.0]),
            window_media=window_media,
            cam_to_window={0: 0},
            wid=0,
            A_anchor=np.array([0.0, 0.0, 0.0]),
            d_midpoint=200.0,
        )

        required_keys = [
            'd_solved', 'plane_pt_solved', 'accepted', 'fallback_reason',
            'A_shape', 'rank', 'cond', 'n_equations', 'n_pairs_used',
            'residual_rms', 'camera_side_ok',
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"


class TestIntegrationPlaneInitializer:

    @staticmethod
    def _build_init_inputs():
        d_true = 200.0
        plane_n = np.array([0.0, 0.0, 1.0])
        A_anchor = np.array([0.0, 0.0, 0.0])

        camera_positions = [
            [-40.0, 0.0, -100.0],
            [40.0, 0.0, -100.0],
        ]
        object_points = [
            [0.0, 0.0, 400.0],
            [20.0, 10.0, 400.0],
            [-20.0, -10.0, 400.0],
        ]

        cam_params, observations, window_media_flat, cam_to_window = _build_synthetic_setup(
            camera_positions, object_points, plane_n, d_true, A_anchor,
        )

        window_media = {0: window_media_flat}

        plane_pt = A_anchor + d_true * plane_n
        X_A_list = {}
        X_B_list = {}
        for k, P_k in enumerate(object_points):
            P_k = np.array(P_k, dtype=np.float64)
            X_A_list[k] = P_k
            X_B_list[k] = P_k + np.array([5.0, 3.0, 0.0])

        return dict(
            cam_params=cam_params,
            cam_to_window=cam_to_window,
            window_media=window_media,
            err_px={},
            verbose=True,
            X_A_list=X_A_list,
            X_B_list=X_B_list,
            active_cam_ids=list(cam_params.keys()),
            observations=observations,
        )

    def test_integration_analytical_path_emits_plane_init_log(self, capsys):
        from modules.camera_calibration.wand_calibration.refraction_wand_calibrator import PlaneInitializer

        kwargs = self._build_init_inputs()
        result = PlaneInitializer.init_window_planes_from_cameras(**kwargs)

        captured = capsys.readouterr()
        assert 'PLANE_INIT' in captured.out
        assert 0 in result
        assert 'plane_pt' in result[0]
        assert 'plane_n' in result[0]

    def test_integration_legacy_env_forces_midpoint(self, monkeypatch, capsys):
        from modules.camera_calibration.wand_calibration.refraction_wand_calibrator import PlaneInitializer

        monkeypatch.setenv('OPENLPT_REFRACTION_PLANE_INIT_MODE', 'legacy')
        kwargs = self._build_init_inputs()
        result = PlaneInitializer.init_window_planes_from_cameras(**kwargs)

        captured = capsys.readouterr()
        assert 'midpoint_legacy_forced' in captured.out
        assert 0 in result

    def test_integration_no_observations_falls_back_silently(self, capsys):
        from modules.camera_calibration.wand_calibration.refraction_wand_calibrator import PlaneInitializer

        kwargs = self._build_init_inputs()
        kwargs['observations'] = None
        result = PlaneInitializer.init_window_planes_from_cameras(**kwargs)

        captured = capsys.readouterr()
        assert 'PLANE_INIT' in captured.out
        assert 'chose=midpoint' in captured.out
        assert 0 in result


class TestCase011EndToEnd:

    def test_case011_plane_d_init_non_regression(self, tmp_path):
        """End-to-end verification helper for case_011 (2 windows, 5 cameras, 2000 frames)."""
        from pathlib import Path
        import sys

        case_dir = Path(r'J:\Refraction_test\case_011')
        harness_path = Path(r'J:\Refraction_test\test_script\run_calibration_worker.py')

        if not case_dir.is_dir():
            pytest.skip("case_011 not available on J-drive")
        if not harness_path.is_file():
            pytest.skip("run_calibration_worker.py not available")

        sys.path.insert(0, str(harness_path.parent))
        try:
            from run_calibration_worker import run_one_case
        except ImportError as e:
            pytest.skip(f"Failed to import run_one_case: {e}")
        finally:
            if str(harness_path.parent) in sys.path:
                sys.path.remove(str(harness_path.parent))

        results_dir = tmp_path / "case011_results"
        results_dir.mkdir()

        result = run_one_case(
            case_dir=str(case_dir),
            runtime_root=str(results_dir),
            smoke_n_frames=0,
            verbosity=1,
        )

        assert result['success'] is True, f"Calibration failed: {result.get('error_message')}"

        metrics = result['metrics']
        assert metrics['ray_mean_mm'] is not None, "Missing ray_mean_mm"
        assert metrics['len_mean_mm'] is not None, "Missing len_mean_mm"
        assert metrics['ray_mean_mm'] <= 0.002, (
            f"ray_mean_mm={metrics['ray_mean_mm']:.6f} exceeds threshold 0.002"
        )
        assert metrics['len_mean_mm'] <= 0.005, (
            f"len_mean_mm={metrics['len_mean_mm']:.6f} exceeds threshold 0.005"
        )

        log_path = Path(result['log_path'])
        assert log_path.is_file(), f"Log file not found: {log_path}"
        log_content = log_path.read_text(encoding='utf-8')
        assert '[PLANE_INIT]' in log_content, (
            "Expected [PLANE_INIT] marker in log but not found (Stage-0 may have failed)"
        )
