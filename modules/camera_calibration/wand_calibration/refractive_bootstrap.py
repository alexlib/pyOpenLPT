# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportArgumentType=false
"""
Refractive Bootstrap (P0 Stage)
===============================
Frozen-Intrinsics Pinhole Bootstrap for Refractive Calibration.

This provides a physically reasonable extrinsic initialization for later stages.
It is NOT calibration - it ONLY initializes extrinsics.

KEY RULES:
- Intrinsics (fx, fy, cx, cy) are FROZEN to UI values
- NO distortion parameters
- NO camFile output (in-memory only)
- Uses 8-Point Algorithm for initialization (same as pinhole Phase 1)
- Only optimizes extrinsics (Phase 1 BA with frozen intrinsics)
- Wand length is the ONLY scale constraint
- Pair selection is handled externally via precalibrate
"""

import numpy as np
from scipy.optimize import least_squares
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
import json
import re


# =========================================================================
# P0 Failure Reason Constants (machine-readable)
# =========================================================================
P0_REASON_OK = "ok"
P0_REASON_INSUFFICIENT_GEOMETRY = "insufficient_geometry"
P0_REASON_ESSENTIAL_MATRIX_FAILED = "essential_matrix_failed"
P0_REASON_TOO_FEW_E_INLIERS = "too_few_e_inliers"
P0_REASON_UNSTABLE_SCALE_RECOVERY = "unstable_scale_recovery"
P0_REASON_CATASTROPHIC_REPROJECTION = "catastrophic_reprojection"
P0_REASON_PHASE1_BA_FAILURE = "phase1_ba_failure"


@dataclass
class Phase2CameraTelemetry:
    """Phase-2 per-camera diagnostics for added cameras."""
    camera_id: int
    total_correspondences: int = 0
    prefilter_candidate_removed: int = 0
    prefilter_removed: int = 0
    prefilter_removed_nonfinite: int = 0
    prefilter_removed_depth: int = 0
    prefilter_removed_wand_length: int = 0
    prefilter_removed_angle: int = 0
    prefilter_kept: int = 0
    prefilter_min_correspondences: int = 6
    prefilter_starvation_fallback: bool = False
    pnp_correspondences: int = 0
    ransac_inliers: Optional[int] = None
    ransac_inlier_ratio: Optional[float] = None
    reproj_rms_px: Optional[float] = None
    confidence_label: str = "normal"
    confidence_warning: Optional[str] = None

    def to_dict(self) -> dict:
        d = {}
        for k, v in asdict(self).items():
            if isinstance(v, (np.floating, np.integer)):
                d[k] = float(v)
            elif isinstance(v, np.ndarray):
                d[k] = v.tolist()
            else:
                d[k] = v
        return d


@dataclass
class P0Telemetry:
    """Structured P0 diagnostics for debugging and downstream classification."""
    selected_pair: Optional[Tuple[int, int]] = None
    baseline_mm: Optional[float] = None

    # Essential matrix stage
    e_inliers: Optional[int] = None
    e_total: Optional[int] = None

    # Pose recovery stage
    pose_inliers: Optional[int] = None
    pose_total: Optional[int] = None
    cheirality_ratio: Optional[float] = None

    # Triangulation / scale stage
    valid_inlier_wand_pairs: Optional[int] = None
    median_triangulation_length: Optional[float] = None
    scale_factor: Optional[float] = None
    scale_factor_finite: Optional[bool] = None

    # BA stage
    ba_initial_cost: Optional[float] = None
    ba_final_cost: Optional[float] = None
    ba_converged: Optional[bool] = None
    ba_message: Optional[str] = None

    # Post-validation
    reproj_err_mean: Optional[float] = None
    reproj_err_max: Optional[float] = None
    wand_length_median: Optional[float] = None
    wand_length_error: Optional[float] = None
    valid_frames: Optional[int] = None

    # Failure tracking
    failure_reason: str = P0_REASON_OK
    failure_detail: Optional[str] = None
    
    # Phase-2 per-camera telemetry
    phase2_cameras: Dict[int, Phase2CameraTelemetry] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict."""
        d = {}
        for k, v in asdict(self).items():
            if k == "phase2_cameras":
                phase2_dict = {}
                for cid, cam_dict in v.items():
                    sanitized = {}
                    for ck, cv in cam_dict.items():
                        if isinstance(cv, (np.floating, np.integer)):
                            sanitized[ck] = float(cv)
                        elif isinstance(cv, np.ndarray):
                            sanitized[ck] = cv.tolist()
                        else:
                            sanitized[ck] = cv
                    phase2_dict[cid] = sanitized
                d[k] = phase2_dict
            elif isinstance(v, (np.floating, np.integer)):
                d[k] = float(v)
            elif isinstance(v, np.ndarray):
                d[k] = v.tolist()
            elif isinstance(v, tuple):
                d[k] = list(v)
            else:
                d[k] = v
        return d

    def emit(self):
        """Print a structured [P0_TELEMETRY] line for machine parsing."""
        print(f"\n[P0_TELEMETRY] {json.dumps(self.to_dict(), default=str)}")


class P0FailureError(RuntimeError):
    """P0 failure with structured reason and telemetry."""
    def __init__(self, message: str, reason: str, telemetry: 'P0Telemetry'):
        telemetry.failure_reason = reason
        telemetry.failure_detail = message
        self.reason = reason
        self.telemetry = telemetry
        super().__init__(message)


def _compute_pair_disparity(pair, points) -> float:
    """Median pixel disparity for a camera pair across shared frames.

    Higher disparity correlates with larger baseline separation and is
    used as a cheap geometry-quality proxy that does not require 3-D
    reconstruction.
    """
    c1, c2 = pair

    def _extract_uv(pts):
        if pts is None:
            return None
        arr = np.asarray(pts)
        if arr.ndim == 1 and arr.shape[0] >= 2:
            return np.array(arr[:2], dtype=np.float64)
        if arr.ndim >= 2 and arr.shape[0] >= 1 and arr.shape[1] >= 2:
            n = min(2, arr.shape[0])
            return np.mean(arr[:n, :2].astype(np.float64), axis=0)
        return None

    d: List[float] = []
    for _, cams in points.items():
        if c1 not in cams or c2 not in cams:
            continue
        uv1 = _extract_uv(cams[c1])
        uv2 = _extract_uv(cams[c2])
        if uv1 is None or uv2 is None:
            continue
        disp = np.linalg.norm(uv1 - uv2)
        if np.isfinite(disp):
            d.append(float(disp))
    if not d:
        return -1.0
    return float(np.median(d))


# Minimum median pixel disparity to consider a pair as having adequate
# geometry.  Used as a fallback when precalib camera positions are
# unavailable.  Degenerate pairs have disparity < 1 px, healthy > 20 px.
_MIN_GEOMETRY_DISPARITY_PX = 5.0

# Minimum 3D baseline (mm) between camera centres from precalibration.
#   degenerate baselines: 0.32 mm (case_023), 0.94 mm (case_028), 1.17 mm (case_012)
#   healthy baselines:    241-2024 mm
_MIN_GEOMETRY_BASELINE_MM = 10.0


def _camera_center(R, T) -> Optional[np.ndarray]:
    """Compute camera centre C = -R^T @ T.  Returns 3-element array or None."""
    try:
        T_col = np.asarray(T, dtype=np.float64).reshape(3, 1)
        R_arr = np.asarray(R, dtype=np.float64)
        C = -R_arr.T @ T_col
        if np.all(np.isfinite(C)):
            return C.flatten()
    except Exception:
        pass
    return None


def _extract_camera_centers(calibrator, cam_ids) -> Dict[int, np.ndarray]:
    """Extract camera centres from whatever the calibrator provides.

    Tries these sources in order (first success wins per camera):
      1. ``calibrator.cameras[cid]``  — WandCalibrator after ``_parse_results``
      2. ``calibrator.final_params[cid]``  — WandCalibrator alternate structure

    Returns ``{cid: centre_3}`` for every camera whose position could be
    determined.
    """
    centers: Dict[int, np.ndarray] = {}

    # --- Source 1: calibrator.cameras ---
    cameras = getattr(calibrator, 'cameras', None)
    if cameras and isinstance(cameras, dict):
        for cid in cam_ids:
            if cid in centers:
                continue
            cam = cameras.get(cid)
            if cam is None or not isinstance(cam, dict):
                continue
            R = cam.get('R')
            T = cam.get('T')
            if R is not None and T is not None:
                c = _camera_center(R, T)
                if c is not None:
                    centers[cid] = c

    # --- Source 2: calibrator.final_params ---
    final_params = getattr(calibrator, 'final_params', None)
    if final_params and isinstance(final_params, dict):
        for cid in cam_ids:
            if cid in centers:
                continue
            fp = final_params.get(cid)
            if fp is None or not isinstance(fp, dict):
                continue
            R = fp.get('R')
            T = fp.get('T')
            if R is not None and T is not None:
                c = _camera_center(R, T)
                if c is not None:
                    centers[cid] = c

    return centers


def _compute_pairwise_essential_baseline(
    pair, calibrator, wand_len_mm: float, focal_px: float,
) -> float:
    """Pairwise baseline (mm) via Essential Matrix for a single camera pair.

    Mirrors the P0 bootstrap's own 8-point decomposition: for cameras
    looking through refractive media the *pairwise* Essential Matrix may
    be degenerate even when global camera positions are well-separated.
    Returns -1.0 on failure.
    """
    c1, c2 = pair
    wand_data = (
        getattr(calibrator, 'wand_points_filtered', None)
        or getattr(calibrator, 'wand_points', None)
        or {}
    )
    if not wand_data:
        return -1.0

    cam_settings = getattr(calibrator, 'camera_settings', None) or {}
    image_size = getattr(calibrator, 'image_size', None) or (800, 1280)

    def _get_K(cid):
        cs = cam_settings.get(cid)
        if cs:
            f = float(cs.get('focal', 0) or 0)
            w = int(cs.get('width', 0) or 0)
            h = int(cs.get('height', 0) or 0)
            if f > 0 and w > 0 and h > 0:
                return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])
        h_def, w_def = image_size
        return np.array([
            [focal_px, 0, w_def / 2.0],
            [0, focal_px, h_def / 2.0],
            [0, 0, 1.0],
        ])

    pts1, pts2 = [], []
    for fid in sorted(wand_data):
        obs = wand_data[fid]
        if c1 in obs and c2 in obs:
            p1, p2 = obs[c1], obs[c2]
            pts1.append(np.asarray(p1[0][:2], dtype=np.float64))
            pts1.append(np.asarray(p1[1][:2], dtype=np.float64))
            pts2.append(np.asarray(p2[0][:2], dtype=np.float64))
            pts2.append(np.asarray(p2[1][:2], dtype=np.float64))
    pts1 = np.array(pts1, dtype=np.float64) if pts1 else np.empty((0, 2))
    pts2 = np.array(pts2, dtype=np.float64) if pts2 else np.empty((0, 2))
    if len(pts1) < 8:
        return -1.0

    try:
        K1 = _get_K(c1)
        K2 = _get_K(c2)
        pts1_n = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K1, None).reshape(-1, 2)
        pts2_n = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K2, None).reshape(-1, 2)

        E, _ = cv2.findEssentialMat(
            pts1_n, pts2_n, focal=1.0, pp=(0.0, 0.0),
            method=cv2.RANSAC, prob=0.999, threshold=1e-3,
        )
        if E is None or E.shape != (3, 3):
            return -1.0

        _, R_rel, t_rel, _ = cv2.recoverPose(
            E, pts1_n, pts2_n, focal=1.0, pp=(0.0, 0.0),
        )

        P1_norm = np.hstack((np.eye(3), np.zeros((3, 1))))
        P2_norm = np.hstack((R_rel, t_rel))
        pts4d = cv2.triangulatePoints(P1_norm, P2_norm, pts1_n.T, pts2_n.T)
        pts3d_raw = (pts4d[:3] / pts4d[3]).T

        dists = np.array([
            np.linalg.norm(pts3d_raw[i] - pts3d_raw[i + 1])
            for i in range(0, len(pts3d_raw), 2)
        ])
        valid = dists[(dists > 1e-3) & (dists < 1e3)]
        if len(valid) < 5:
            return -1.0
        scale = wand_len_mm / float(np.median(valid))
        t_scaled = t_rel * scale

        baseline = float(np.linalg.norm(t_scaled))
        return baseline if np.isfinite(baseline) else -1.0
    except Exception:
        return -1.0


def _compute_precalib_baseline(pair, calibrator, camera_centers=None) -> float:
    """3D baseline (mm) between two camera centres using precalib R/T.

    Camera centre: ``C = -R^T @ T``.  Returns -1.0 if unavailable.

    Parameters
    ----------
    pair : tuple of int
        Camera IDs.
    calibrator : object
        Calibrator instance (used only when *camera_centers* is None).
    camera_centers : dict, optional
        Pre-extracted ``{cid: centre_3}`` dict.  When supplied, the
        calibrator is not queried — this avoids repeated attribute
        look-ups and supports calibrators that don't expose extrinsics.
    """
    c1, c2 = pair

    # Fast path: pre-extracted centres
    if camera_centers is not None:
        C1 = camera_centers.get(c1)
        C2 = camera_centers.get(c2)
        if C1 is not None and C2 is not None:
            dist = float(np.linalg.norm(np.asarray(C1) - np.asarray(C2)))
            return dist if np.isfinite(dist) else -1.0
        return -1.0

    # Legacy path: read directly from calibrator (WandCalibrator compat)
    cameras = getattr(calibrator, 'cameras', None)
    if not cameras:
        return -1.0
    cam1 = cameras.get(c1)
    cam2 = cameras.get(c2)
    if cam1 is None or cam2 is None:
        return -1.0
    R1 = cam1.get('R')
    T1 = cam1.get('T')
    R2 = cam2.get('R')
    T2 = cam2.get('T')
    if R1 is None or T1 is None or R2 is None or T2 is None:
        return -1.0
    C1 = _camera_center(R1, T1)
    C2 = _camera_center(R2, T2)
    if C1 is None or C2 is None:
        return -1.0
    dist = float(np.linalg.norm(C1 - C2))
    return dist if np.isfinite(dist) else -1.0


def select_ranked_pairs_via_precalib(
    base_calibrator,
    wand_len_mm: float,
    initial_focal_px: float,
) -> Optional[List[Tuple[int, int]]]:
    """
    Return camera pairs ranked by combined reprojection quality and
    geometry quality (pixel disparity as baseline proxy).

    Pairs with low disparity (nearly co-located cameras) are demoted
    regardless of per-camera reprojection error, which prevents the
    pathological case where two cameras with the lowest individual error
    happen to be nearly co-located.

    Returns:
        Ranked list of (cam_i, cam_j) pairs (best first), or None on failure.
    """
    print("\n[BOOT] Running Precalibration Check to select best pair...")

    points = (
        getattr(base_calibrator, 'wand_points_filtered', None)
        or getattr(base_calibrator, 'wand_points', {})
    )

    try:
        if not hasattr(base_calibrator, 'run_precalibration_check'):
            raise AttributeError("run_precalibration_check is unavailable on base calibrator")
        ret, msg, precalib_result = base_calibrator.run_precalibration_check(
            wand_length=wand_len_mm,
            init_focal_length=initial_focal_px,
        )
    except Exception as e:
        print(f"  [WARN] Precalibration failed: {e}. Falling back to shared count.")
        return _ranked_pairs_fallback(points)

    if not ret:
        print(f"  [WARN] Precalibration returned False: {msg}")

    wand_data = base_calibrator.wand_points_filtered or base_calibrator.wand_points
    all_cam_ids = sorted(list(set(cid for f in wand_data.values() for cid in f)))

    per_cam_error: Dict[int, float] = {}
    if hasattr(base_calibrator, 'per_frame_errors') and base_calibrator.per_frame_errors:
        cam_errors_list: Dict[int, List[float]] = {cid: [] for cid in all_cam_ids}
        for fid, frame_data in base_calibrator.per_frame_errors.items():
            if 'cam_errors' in frame_data:
                for cid, err in frame_data['cam_errors'].items():
                    if cid in cam_errors_list:
                        cam_errors_list[cid].append(err)
        for cid in all_cam_ids:
            if cam_errors_list[cid]:
                per_cam_error[cid] = np.sqrt(np.mean(np.array(cam_errors_list[cid])**2))

    if not per_cam_error:
        for line in msg.split('\n'):
            match = re.search(r'Cam\s*(\d+):\s*([\d.]+)\s*px', line)
            if match:
                per_cam_error[int(match.group(1))] = float(match.group(2))

    if not per_cam_error:
        print("  [WARN] Could not determine per-camera errors.")
        return None

    print("\n[BOOT] Per-camera reprojection errors (Pinhole approx):")
    for cid in sorted(per_cam_error.keys()):
        print(f"  Cam {cid}: {per_cam_error[cid]:.2f}px")

    cam_ids = sorted(per_cam_error.keys())
    if len(cam_ids) < 2:
        return None

    # --- Extract camera centres for 3D baseline checks ---
    camera_centers = _extract_camera_centers(base_calibrator, cam_ids)
    if camera_centers:
        print(f"  [BOOT] Camera centres available for {len(camera_centers)}/{len(cam_ids)} cameras (precalib)")
    else:
        print("  [BOOT] Calibrator did not expose extrinsics; pairwise Essential Matrix fallback may be used")

    candidate_pairs: List[Tuple[int, int]] = []
    for i in range(len(cam_ids)):
        for j in range(i + 1, len(cam_ids)):
            candidate_pairs.append((cam_ids[i], cam_ids[j]))

    if not candidate_pairs:
        return None

    essential_baseline_cache: Dict[Tuple[int, int], float] = {}
    pair_metrics: Dict[Tuple[int, int], dict] = {}

    max_err = max(per_cam_error.values(), default=1.0)
    if max_err <= 0:
        max_err = 1.0

    def _pair_score(pair):
        if pair in pair_metrics:
            return pair_metrics[pair]["score"]

        c1, c2 = pair
        reproj_avg = (per_cam_error[c1] + per_cam_error[c2]) / 2.0
        reproj_norm = reproj_avg / max_err

        baseline_3d = _compute_precalib_baseline(pair, base_calibrator, camera_centers=camera_centers)
        baseline_source = "precalib"
        disparity = _compute_pair_disparity(pair, points)

        if baseline_3d < 0:
            if pair not in essential_baseline_cache:
                essential_baseline_cache[pair] = _compute_pairwise_essential_baseline(
                    pair, base_calibrator, wand_len_mm, initial_focal_px,
                )
            baseline_3d = essential_baseline_cache[pair]
            baseline_source = "essential"

        if baseline_3d >= 0:
            geometry_ok = baseline_3d >= _MIN_GEOMETRY_BASELINE_MM
        else:
            geometry_ok = disparity >= _MIN_GEOMETRY_DISPARITY_PX
            baseline_source = "disparity"

        penalty = 0.0 if geometry_ok else 1000.0
        score = penalty + reproj_norm
        pair_metrics[pair] = {
            "score": score,
            "reproj_avg": reproj_avg,
            "baseline_3d": baseline_3d,
            "baseline_source": baseline_source,
            "disparity": disparity,
            "geometry_ok": geometry_ok,
        }
        return score

    ranked = sorted(candidate_pairs, key=_pair_score)

    if not any(pair_metrics[pair]["geometry_ok"] for pair in ranked):
        print("[BOOT] No geometry-valid camera pair found; aborting pair selection.")
        return None

    print("\n[BOOT] Pair ranking (geometry-aware):")
    for idx, pair in enumerate(ranked[:5]):
        metrics = pair_metrics[pair]
        bl = metrics["baseline_3d"]
        disp = metrics["disparity"]
        reproj = metrics["reproj_avg"]
        source = metrics["baseline_source"]
        invalid = " [geometry-invalid]" if not metrics["geometry_ok"] else ""
        tag = " <-- selected" if idx == 0 else ""
        print(
            f"  #{idx+1} {pair}: reproj_avg={reproj:.2f}px, baseline_3d={bl:.1f}mm, "
            f"disparity={disp:.1f}px, source={source}{invalid}{tag}"
        )

    print(f"[BOOT] Selected best pair: {ranked[0]}")
    return ranked


def _ranked_pairs_fallback(points) -> Optional[List[Tuple[int, int]]]:
    """Fallback ranking when precalibration is unavailable.

    Uses shared-frame count and median pixel disparity (same signals as
    the original fallback path, but returns a ranked list).
    """
    if not points:
        return None

    counts: Dict[Tuple[int, int], int] = {}
    all_cams: set = set()
    for fid, cams in points.items():
        cam_list = list(cams.keys())
        for i in range(len(cam_list)):
            for j in range(i + 1, len(cam_list)):
                c1, c2 = sorted((cam_list[i], cam_list[j]))
                counts[(c1, c2)] = counts.get((c1, c2), 0) + 1
                all_cams.add(c1)
                all_cams.add(c2)

    if not counts:
        cams_sorted = sorted(list(all_cams))
        if len(cams_sorted) >= 2:
            return [(cams_sorted[0], cams_sorted[1])]
        return None

    def _fallback_score(item):
        pair, cnt = item
        disparity = _compute_pair_disparity(pair, points)
        geometry_ok = disparity >= _MIN_GEOMETRY_DISPARITY_PX
        penalty = 0.0 if geometry_ok else 1000.0
        return (-cnt + penalty, -disparity)

    ranked = [p for p, _ in sorted(counts.items(), key=_fallback_score)]

    if ranked:
        top = ranked[0]
        disp = _compute_pair_disparity(top, points)
        cnt = counts[top]
        print(
            f"[BOOT] Fallback: Selected pair {top} with {cnt} shared frames "
            f"(median disparity={disp:.2f}px)."
        )
    return ranked if ranked else None


def select_best_pair_via_precalib(
    base_calibrator,
    wand_len_mm: float,
    initial_focal_px: float,
) -> Optional[Tuple[int, int]]:
    """
    Backward-compatible wrapper: returns the single best pair from the
    geometry-aware ranked list.
    """
    ranked = select_ranked_pairs_via_precalib(
        base_calibrator, wand_len_mm, initial_focal_px,
    )
    if ranked and len(ranked) > 0:
        return ranked[0]
    return None


@dataclass
class PinholeBootstrapP0Config:
    """Configuration for P0 bootstrap."""
    wand_length_mm: float = 10.0
    ui_focal_px: float = 9000.0  # UI-provided focal length (FROZEN)
    ftol: float = 1e-6
    xtol: float = 1e-6
    
    # Phase-2 prefiltering thresholds
    phase2_prefilter_wand_length_error_mm: float = 2.0  # Max wand length error before removing a 3D point
    phase2_prefilter_depth_min_mm: float = 100.0        # Min depth (Z-coord in cam frame) to prevent behind-camera
    phase2_prefilter_depth_max_mm: float = 10000.0      # Max depth to prevent extreme-depth outliers
    phase2_prefilter_min_triangulation_angle_deg: float = 0.5
    
    # Phase-2 RANSAC PnP settings
    phase2_ransac_enabled: bool = False                  # Enable RANSAC for added-camera PnP
    phase2_ransac_reproj_threshold_px: float = 3.0       # Inlier threshold (px) for RANSAC PnP
    phase2_ransac_confidence: float = 0.999              # RANSAC confidence level
    phase2_ransac_min_inlier_ratio: float = 0.5          # Min inlier ratio to accept RANSAC result
    
    # Phase-2 confidence labeling thresholds
    phase2_confidence_min_correspondences: int = 12      # Min 2D-3D pairs for normal confidence
    phase2_confidence_max_reproj_rms_px: float = 5.0     # Max RMS for normal confidence
    phase2_confidence_min_inlier_ratio: float = 0.6      # Min inlier ratio for normal confidence (when RANSAC enabled)

    phase2_pair_fallback_rms_threshold_px: float = 35.0
    phase2_pair_fallback_max_retries: int = 2


class PinholeBootstrapP0:
    """
    Stage P0: Two-camera pinhole initialization with frozen intrinsics.
    
    Uses 8-Point Algorithm (same as original pinhole Phase 1) but with frozen intrinsics.
    Optimizes only extrinsics.
    
    Pair selection is handled externally (via precalibrate).
    """
    
    def __init__(self, config: PinholeBootstrapP0Config):
        self.config = config
        self.last_phase2_telemetry: Dict[int, dict] = {}

    @staticmethod
    def _get_camera_intrinsics(cam_id: int, camera_settings: Dict[int, dict]) -> Tuple[np.ndarray, float, float, float]:
        if cam_id not in camera_settings:
            raise ValueError(f"[P0] Missing camera_settings for cam {cam_id}")
        cfg = camera_settings[cam_id]
        f = float(cfg.get('focal', 0.0))
        w = float(cfg.get('width', 0.0))
        h = float(cfg.get('height', 0.0))
        if f <= 0 or w <= 0 or h <= 0:
            raise ValueError(
                f"[P0] Invalid camera_settings for cam {cam_id}: focal={f}, width={w}, height={h}"
            )
        cx, cy = w / 2.0, h / 2.0
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        return K, f, cx, cy
        
    def run(
        self,
        cam_i: int,
        cam_j: int,
        observations: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]],
        camera_settings: Dict[int, dict],
        progress_callback=None
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Run P0 pinhole bootstrap for camera pair using 8-Point Algorithm.
        
        This is identical to pinhole Phase 1, but with frozen intrinsics.
        
        Args:
            cam_i, cam_j: Camera IDs (cam_i fixed at origin)
            observations: {fid: {cid: (uvA, uvB)}}
            camera_settings: per-camera intrinsics source from UI table
            progress_callback: Optional callback(phase, ray, len, cost)
            
        Returns:
            params_i: [rvec(3), tvec(3)] for cam_i (zeros)
            params_j: [rvec(3), tvec(3)] for cam_j (from 8-Point + refinement)
            report: diagnostics dict
        """
        telemetry = P0Telemetry(selected_pair=(cam_i, cam_j))

        K_i, f_i, cx_i, cy_i = self._get_camera_intrinsics(cam_i, camera_settings)
        K_j, f_j, cx_j, cy_j = self._get_camera_intrinsics(cam_j, camera_settings)
        
        print(f"\n{'='*60}")
        print(f"[P0] Pinhole Bootstrap - Frozen Intrinsics (8-Point)")
        print(f"{'='*60}")
        print(f"  Camera pair: ({cam_i}, {cam_j})")
        
        if progress_callback:
            try:
                progress_callback("P0 Pair Init", -1, 0, 0, 0)
            except:
                pass
        
        valid_frames = self._collect_valid_frames(observations, cam_i, cam_j)
        telemetry.valid_frames = len(valid_frames)
        print(f"  Valid frames: {len(valid_frames)}")
        
        if len(valid_frames) < 10:
            raise P0FailureError(
                f"[P0 FAIL] Insufficient frames: {len(valid_frames)} < 10",
                P0_REASON_INSUFFICIENT_GEOMETRY,
                telemetry,
            )
        
        pts_i = []
        pts_j = []
        
        for fid in valid_frames:
            uvA_i, uvB_i = observations[fid][cam_i]
            uvA_j, uvB_j = observations[fid][cam_j]
            
            pts_i.append(uvA_i)
            pts_i.append(uvB_i)
            pts_j.append(uvA_j)
            pts_j.append(uvB_j)
        
        pts_i = np.array(pts_i, dtype=np.float64)
        pts_j = np.array(pts_j, dtype=np.float64)
        
        print(f"\n[P0] Step 1: Essential Matrix (8-Point Algorithm)...")
        if progress_callback:
            try:
                progress_callback("Use PinHole model to initialize camera parameters...", -1, 0, 0, 0)
            except:
                pass

        pts_i_norm = cv2.undistortPoints(pts_i.reshape(-1, 1, 2), K_i, None).reshape(-1, 2)
        pts_j_norm = cv2.undistortPoints(pts_j.reshape(-1, 1, 2), K_j, None).reshape(-1, 2)
        E, mask = cv2.findEssentialMat(
            pts_i_norm,
            pts_j_norm,
            focal=1.0,
            pp=(0.0, 0.0),
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1e-3,
        )
        
        telemetry.e_total = len(pts_i)

        if E is None or E.shape != (3, 3):
            telemetry.e_inliers = 0
            raise P0FailureError(
                "[P0 FAIL] Essential Matrix computation failed",
                P0_REASON_ESSENTIAL_MATRIX_FAILED,
                telemetry,
            )
        
        inlier_idx = np.where(mask.ravel() > 0)[0]
        n_E_inliers = len(inlier_idx)
        telemetry.e_inliers = n_E_inliers

        if n_E_inliers < 8:
            raise P0FailureError(
                f"[P0 FAIL] Too few Essential inliers: {n_E_inliers}",
                P0_REASON_TOO_FEW_E_INLIERS,
                telemetry,
            )

        n_inliers, R_rel, t_rel, mask_pose = cv2.recoverPose(
            E, pts_i_norm[inlier_idx], pts_j_norm[inlier_idx], focal=1.0, pp=(0.0, 0.0)
        )

        telemetry.pose_inliers = int(n_inliers)
        telemetry.pose_total = n_E_inliers
        telemetry.cheirality_ratio = float(n_inliers) / max(1, n_E_inliers)

        print(f"  E-Matrix Inliers: {n_E_inliers} / {len(pts_i)}")
        print(f"  Pose Inliers: {n_inliers} / {n_E_inliers}")
        print(f"  Cheirality ratio: {telemetry.cheirality_ratio:.3f}")
        
        print(f"\n[P0] Step 2: Triangulation & Scale Recovery...")
        if progress_callback:
            try:
                progress_callback("Use PinHole model to initialize camera parameters...", -1, 0, 0, 0)
            except:
                pass
        
        P_i = np.hstack([np.eye(3), np.zeros((3, 1))])
        P_j = np.hstack([R_rel, t_rel])
        
        pts_4d_hom = cv2.triangulatePoints(P_i, P_j, pts_i_norm[inlier_idx].T, pts_j_norm[inlier_idx].T)
        pts_3d_inlier = (pts_4d_hom[:3] / pts_4d_hom[3]).T

        pose_inlier_idx_local = np.where(mask_pose.ravel() > 0)[0]
        pose_inlier_idx_global = inlier_idx[pose_inlier_idx_local]

        pts_4d_hom_all = cv2.triangulatePoints(P_i, P_j, pts_i_norm.T, pts_j_norm.T)
        pts_3d = (pts_4d_hom_all[:3] / pts_4d_hom_all[3]).T

        wand_lengths_inlier = []
        for i_frame in range(0, len(pts_3d) - 1, 2):
            if i_frame in inlier_idx and (i_frame + 1) in inlier_idx:
                idx_A_in_inliers = np.where(inlier_idx == i_frame)[0][0]
                idx_B_in_inliers = np.where(inlier_idx == (i_frame + 1))[0][0]
                ptA = pts_3d_inlier[idx_A_in_inliers]
                ptB = pts_3d_inlier[idx_B_in_inliers]
                wand_lengths_inlier.append(np.linalg.norm(ptB - ptA))

        telemetry.valid_inlier_wand_pairs = len(wand_lengths_inlier)

        if len(wand_lengths_inlier) < 5:
            raise P0FailureError(
                f"[P0 FAIL] Triangulation produced only {len(wand_lengths_inlier)} inlier wand pairs (need >= 5)",
                P0_REASON_UNSTABLE_SCALE_RECOVERY,
                telemetry,
            )

        wand_lengths_inlier = np.array(wand_lengths_inlier)
        valid_lengths_inlier = wand_lengths_inlier[(wand_lengths_inlier > 0.001) & (wand_lengths_inlier < 1000)]

        if len(valid_lengths_inlier) < 3:
            wand_lengths = []
            for i in range(0, len(pts_3d), 2):
                ptA = pts_3d[i]
                ptB = pts_3d[i + 1]
                wand_lengths.append(np.linalg.norm(ptB - ptA))
            wand_lengths = np.array(wand_lengths)
            valid_lengths = wand_lengths[(wand_lengths > 0.001) & (wand_lengths < 1000)]
            print(f"  [WARN] Insufficient inlier pairs ({len(valid_lengths_inlier)}); using all correspondences for scale.")
            if len(valid_lengths) == 0:
                raise P0FailureError(
                    "[P0 FAIL] No valid triangulation lengths for scale recovery",
                    P0_REASON_UNSTABLE_SCALE_RECOVERY,
                    telemetry,
                )
            median_length = np.median(valid_lengths)
        else:
            median_length = np.median(valid_lengths_inlier)
            print(f"  Scale anchor: {len(valid_lengths_inlier)} valid inlier wand pairs, median={median_length:.4f} mm")

        telemetry.median_triangulation_length = float(median_length)

        scale_factor = self.config.wand_length_mm / median_length
        telemetry.scale_factor = float(scale_factor)
        telemetry.scale_factor_finite = bool(np.isfinite(scale_factor))

        if not np.isfinite(scale_factor) or scale_factor <= 0:
            raise P0FailureError(
                f"[P0 FAIL] Scale factor is invalid: {scale_factor}",
                P0_REASON_UNSTABLE_SCALE_RECOVERY,
                telemetry,
            )

        print(f"  Scale factor: {scale_factor:.6f} (finite={telemetry.scale_factor_finite})")
        
        t_scaled = t_rel * scale_factor
        
        rvec_j, _ = cv2.Rodrigues(R_rel)
        
        params_i = np.zeros(6)
        params_j = np.concatenate([rvec_j.flatten(), t_scaled.flatten()])
        
        print(f"\n[P0] Step 3: Extrinsic Refinement (frozen intrinsics)...")
        if progress_callback:
            try:
                progress_callback("Use PinHole model to initialize camera parameters...", -1, 0, 0, 0)
            except:
                pass
        
        pts_3d_scaled = pts_3d * scale_factor
        n_pts = len(pts_3d_scaled)
        
        x0 = np.concatenate([params_i, params_j, pts_3d_scaled.flatten()])
        
        from scipy.sparse import lil_matrix
        
        n_frames = len(valid_frames)
        n_res = n_frames * 9
        n_cams = 2
        n_cam_params = 6
        pt_start = n_cams * n_cam_params
        n_params = pt_start + n_pts * 3
        
        A_sparsity = lil_matrix((n_res, n_params), dtype=int)
        
        for i, fid in enumerate(valid_frames):
            idx_ptA = pt_start + i * 6
            idx_ptB = pt_start + i * 6 + 3
            base_res = i * 9
            
            A_sparsity[base_res, idx_ptA:idx_ptA+3] = 1
            A_sparsity[base_res, idx_ptB:idx_ptB+3] = 1
            
            A_sparsity[base_res+1:base_res+3, 0:6] = 1
            A_sparsity[base_res+1:base_res+3, idx_ptA:idx_ptA+3] = 1
            A_sparsity[base_res+3:base_res+5, 0:6] = 1
            A_sparsity[base_res+3:base_res+5, idx_ptB:idx_ptB+3] = 1
            
            A_sparsity[base_res+5:base_res+7, 6:12] = 1
            A_sparsity[base_res+5:base_res+7, idx_ptA:idx_ptA+3] = 1
            A_sparsity[base_res+7:base_res+9, 6:12] = 1
            A_sparsity[base_res+7:base_res+9, idx_ptB:idx_ptB+3] = 1
        
        self._res_call_count = 0 
        initial_cost_captured = [None]

        def residuals_func(x):
            p_i = x[:6]
            p_j = x[6:12]
            pts = x[12:].reshape(-1, 3)
            
            R_i, _ = cv2.Rodrigues(p_i[:3])
            t_i = p_i[3:6].reshape(3, 1)
            R_j, _ = cv2.Rodrigues(p_j[:3])
            t_j = p_j[3:6].reshape(3, 1)
            
            res = []
            sq_err_len = 0.0
            n_len = 0
            sq_err_proj = 0.0
            n_proj = 0
            for idx, fid in enumerate(valid_frames):
                uvA_i, uvB_i = observations[fid][cam_i]
                uvA_j, uvB_j = observations[fid][cam_j]
                
                ptA = pts[idx * 2]
                ptB = pts[idx * 2 + 1]
                
                wand_len = np.linalg.norm(ptB - ptA)
                d_len = wand_len - self.config.wand_length_mm
                res.append(d_len)
                sq_err_len += d_len * d_len
                n_len += 1
                
                proj_Ai = self._project(ptA, R_i, t_i, K_i)
                proj_Bi = self._project(ptB, R_i, t_i, K_i)
                diff_Ai = (proj_Ai - uvA_i)
                diff_Bi = (proj_Bi - uvB_i)
                res.extend(diff_Ai.tolist())
                res.extend(diff_Bi.tolist())
                sq_err_proj += float(np.sum(diff_Ai**2) + np.sum(diff_Bi**2))
                n_proj += 4

                proj_Aj = self._project(ptA, R_j, t_j, K_j)
                proj_Bj = self._project(ptB, R_j, t_j, K_j)
                diff_Aj = (proj_Aj - uvA_j)
                diff_Bj = (proj_Bj - uvB_j)
                res.extend(diff_Aj.tolist())
                res.extend(diff_Bj.tolist())
                sq_err_proj += float(np.sum(diff_Aj**2) + np.sum(diff_Bj**2))
                n_proj += 4

            res_arr = np.array(res)

            self._res_call_count += 1
            if initial_cost_captured[0] is None:
                initial_cost_captured[0] = float(0.5 * np.sum(res_arr**2))

            if progress_callback and self._res_call_count % 5 == 0:
                try:
                    rmse_len = np.sqrt(sq_err_len / max(1, n_len))
                    rmse_proj = np.sqrt(sq_err_proj / max(1, n_proj))
                    rmse_ray = -1.0
                    cost = 0.5 * np.sum(res_arr**2)

                    progress_callback(
                        "Use PinHole model to initialize camera parameters...",
                        rmse_ray,
                        rmse_len,
                        rmse_proj,
                        cost,
                    )
                except:
                    pass
            
            return res_arr

        
        result = least_squares(
            residuals_func, x0,
            jac_sparsity=A_sparsity,
            method='trf',
            x_scale='jac',
            f_scale=1.0,
            verbose=1,
            ftol=self.config.ftol,
            xtol=self.config.xtol,
            max_nfev=100,
        )

        telemetry.ba_initial_cost = initial_cost_captured[0]
        telemetry.ba_final_cost = float(result.cost)
        telemetry.ba_converged = bool(result.success)
        telemetry.ba_message = str(result.message)

        if not result.success and result.cost > 1e8:
            raise P0FailureError(
                f"[P0 FAIL] Phase 1 BA failed to converge: cost={result.cost:.2e}, "
                f"message='{result.message}'",
                P0_REASON_PHASE1_BA_FAILURE,
                telemetry,
            )

        params_i_raw = result.x[:6]
        params_j_raw = result.x[6:12]
        pts_3d_raw = result.x[12:].reshape(-1, 3)

        R_i_raw, _ = cv2.Rodrigues(params_i_raw[:3])
        t_i_raw = params_i_raw[3:6].reshape(3, 1)
        R_j_raw, _ = cv2.Rodrigues(params_j_raw[:3])
        t_j_raw = params_j_raw[3:6].reshape(3, 1)

        pts_3d_opt = (R_i_raw @ pts_3d_raw.T + t_i_raw).T

        R_j_anchored = R_j_raw @ R_i_raw.T
        t_j_anchored = t_j_raw - R_j_anchored @ t_i_raw
        rvec_j_anchored, _ = cv2.Rodrigues(R_j_anchored)

        params_i_opt = np.zeros(6)
        params_j_opt = np.concatenate([rvec_j_anchored.flatten(), t_j_anchored.flatten()])

        print(f"  [ANCHORING] Phase 1 BA: cam_{cam_i} fixed at origin (post-processing re-expression)")
        print(f"  cam_{cam_j} rvec: [{params_j_opt[0]:.4f}, {params_j_opt[1]:.4f}, {params_j_opt[2]:.4f}]")
        print(f"  cam_{cam_j} tvec: [{params_j_opt[3]:.2f}, {params_j_opt[4]:.2f}, {params_j_opt[5]:.2f}]")
        print(f"  BA cost: initial={telemetry.ba_initial_cost:.2e}, final={result.cost:.2e}")
        
        report = self._compute_diagnostics(
            cam_i, cam_j, params_i_opt, params_j_opt,
            observations, valid_frames, K_i, K_j
        )
        report['scale_factor'] = scale_factor
        report['n_inliers'] = n_inliers

        telemetry.baseline_mm = report['baseline_mm']
        telemetry.reproj_err_mean = report['reproj_err_mean']
        telemetry.reproj_err_max = report['reproj_err_max']
        telemetry.wand_length_median = report['wand_length_median']
        telemetry.wand_length_error = report['wand_length_error']
        
        self._validate(report, telemetry)
        
        print(f"\n[P0] Phase 1 Complete:")
        print(f"  cam_{cam_i} rvec: [{params_i_opt[0]:.4f}, {params_i_opt[1]:.4f}, {params_i_opt[2]:.4f}]")
        print(f"  cam_{cam_i} tvec: [{params_i_opt[3]:.2f}, {params_i_opt[4]:.2f}, {params_i_opt[5]:.2f}]")
        print(f"  cam_{cam_j} rvec: [{params_j_opt[0]:.4f}, {params_j_opt[1]:.4f}, {params_j_opt[2]:.4f}]")
        print(f"  cam_{cam_j} tvec: [{params_j_opt[3]:.2f}, {params_j_opt[4]:.2f}, {params_j_opt[5]:.2f}]")
        print(f"  Baseline: {report['baseline_mm']:.2f} mm")
        print(f"  Wand length: {report['wand_length_median']:.4f} mm")

        telemetry.emit()

        report['p0_telemetry'] = telemetry.to_dict()
        
        return params_i_opt, params_j_opt, report
    

    
    def _project(self, pt3d: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Project 3D point to 2D using pinhole model."""
        pt_cam = R @ pt3d.reshape(3, 1) + t
        pt_cam = pt_cam.flatten()
        if pt_cam[2] <= 0:
            return np.array([1e6, 1e6])  # Behind camera
        pt_norm = pt_cam[:2] / pt_cam[2]
        pt_px = K[:2, :2] @ pt_norm + K[:2, 2]
        return pt_px

    def _ray_dir_world(self, uv: np.ndarray, K: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Build world-space pinhole ray direction from pixel coordinate."""
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])
        x = (float(uv[0]) - cx) / max(abs(fx), 1e-12)
        y = (float(uv[1]) - cy) / max(abs(fy), 1e-12)
        d_cam = np.array([x, y, 1.0], dtype=np.float64)
        d_cam /= (np.linalg.norm(d_cam) + 1e-12)
        d_world = R.T @ d_cam
        d_world /= (np.linalg.norm(d_world) + 1e-12)
        return d_world

    def _point_to_ray_dist(self, X: np.ndarray, C: np.ndarray, d: np.ndarray) -> float:
        """Distance from 3D point to 3D ray (half-line), in mm."""
        v = X - C
        t = float(np.dot(v, d))
        if t < 0.0:
            return float(np.linalg.norm(v))
        return float(np.linalg.norm(v - t * d))

    def _compute_triangulation_angle_deg(
        self,
        X: np.ndarray,
        cam_center_a: np.ndarray,
        cam_center_b: np.ndarray,
    ) -> float:
        v1 = np.asarray(cam_center_a, dtype=np.float64) - np.asarray(X, dtype=np.float64)
        v2 = np.asarray(cam_center_b, dtype=np.float64) - np.asarray(X, dtype=np.float64)
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 <= 1e-12 or n2 <= 1e-12:
            return 0.0
        cosang = float(np.dot(v1, v2) / (n1 * n2))
        cosang = float(np.clip(cosang, -1.0, 1.0))
        return float(np.degrees(np.arccos(cosang)))

    def _classify_phase2_seed_frame(
        self,
        XA: np.ndarray,
        XB: np.ndarray,
        R_i: np.ndarray,
        t_i: np.ndarray,
        R_j: np.ndarray,
        t_j: np.ndarray,
        cam_center_i: np.ndarray,
        cam_center_j: np.ndarray,
    ) -> Optional[str]:
        pts = [np.asarray(XA, dtype=np.float64), np.asarray(XB, dtype=np.float64)]
        if any((pt.shape != (3,)) or (not np.all(np.isfinite(pt))) for pt in pts):
            return "nonfinite"

        depth_min = float(self.config.phase2_prefilter_depth_min_mm)
        depth_max = float(self.config.phase2_prefilter_depth_max_mm)
        for pt in pts:
            zi = float((R_i @ pt.reshape(3, 1) + t_i)[2, 0])
            zj = float((R_j @ pt.reshape(3, 1) + t_j)[2, 0])
            if not np.isfinite(zi) or not np.isfinite(zj):
                return "nonfinite"
            if zi < depth_min or zi > depth_max or zj < depth_min or zj > depth_max:
                return "depth"

        wand_len = float(np.linalg.norm(pts[1] - pts[0]))
        if (not np.isfinite(wand_len)) or (
            abs(wand_len - float(self.config.wand_length_mm)) > float(self.config.phase2_prefilter_wand_length_error_mm)
        ):
            return "wand_length"

        min_angle = float(self.config.phase2_prefilter_min_triangulation_angle_deg)
        if min_angle > 0.0:
            angles = [
                self._compute_triangulation_angle_deg(pt, cam_center_i, cam_center_j)
                for pt in pts
            ]
            if min(angles) < min_angle:
                return "angle"

        return None

    def _build_phase2_correspondences(
        self,
        cid: int,
        observations: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]],
        points_3d: Dict[int, Tuple[np.ndarray, np.ndarray]],
        params_i: np.ndarray,
        params_j: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Phase2CameraTelemetry]:
        telemetry = Phase2CameraTelemetry(camera_id=cid)
        telemetry.prefilter_min_correspondences = 6

        R_i, _ = cv2.Rodrigues(params_i[:3])
        t_i = params_i[3:6].reshape(3, 1)
        R_j, _ = cv2.Rodrigues(params_j[:3])
        t_j = params_j[3:6].reshape(3, 1)
        cam_center_i = _camera_center(R_i, t_i)
        cam_center_j = _camera_center(R_j, t_j)

        kept_2d: List[np.ndarray] = []
        kept_3d: List[np.ndarray] = []
        raw_2d: List[np.ndarray] = []
        raw_3d: List[np.ndarray] = []

        for fid, (XA, XB) in points_3d.items():
            if fid not in observations or cid not in observations[fid]:
                continue
            uvA, uvB = observations[fid][cid]

            frame_pairs = []
            if uvA is not None:
                frame_pairs.append((np.asarray(uvA, dtype=np.float64), np.asarray(XA, dtype=np.float64)))
            if uvB is not None:
                frame_pairs.append((np.asarray(uvB, dtype=np.float64), np.asarray(XB, dtype=np.float64)))
            if not frame_pairs:
                continue

            telemetry.total_correspondences += len(frame_pairs)
            for uv, X in frame_pairs:
                raw_2d.append(uv)
                raw_3d.append(X)

            reason = self._classify_phase2_seed_frame(
                np.asarray(XA, dtype=np.float64),
                np.asarray(XB, dtype=np.float64),
                R_i,
                t_i,
                R_j,
                t_j,
                cam_center_i,
                cam_center_j,
            ) if cam_center_i is not None and cam_center_j is not None else None

            if reason is None:
                for uv, X in frame_pairs:
                    kept_2d.append(uv)
                    kept_3d.append(X)
                continue

            removed = len(frame_pairs)
            telemetry.prefilter_candidate_removed += removed
            if reason == "nonfinite":
                telemetry.prefilter_removed_nonfinite += removed
            elif reason == "depth":
                telemetry.prefilter_removed_depth += removed
            elif reason == "wand_length":
                telemetry.prefilter_removed_wand_length += removed
            elif reason == "angle":
                telemetry.prefilter_removed_angle += removed

        telemetry.prefilter_removed = telemetry.prefilter_candidate_removed
        telemetry.prefilter_kept = len(kept_2d)

        if telemetry.prefilter_candidate_removed > 0 and telemetry.prefilter_kept < telemetry.prefilter_min_correspondences:
            telemetry.prefilter_starvation_fallback = True
            telemetry.prefilter_removed = 0
            telemetry.prefilter_kept = len(raw_2d)
            kept_2d = raw_2d
            kept_3d = raw_3d
            print(
                f"    [PREFILTER] cam_{cid}: candidate filter would keep only {telemetry.prefilter_candidate_removed} removed / "
                f"{telemetry.total_correspondences} total below floor {telemetry.prefilter_min_correspondences}; restoring all correspondences."
            )
        elif telemetry.prefilter_candidate_removed > 0:
            print(
                f"    [PREFILTER] cam_{cid}: removed {telemetry.prefilter_removed}/{telemetry.total_correspondences} correspondences "
                f"(nonfinite={telemetry.prefilter_removed_nonfinite}, depth={telemetry.prefilter_removed_depth}, "
                f"wand={telemetry.prefilter_removed_wand_length}, angle={telemetry.prefilter_removed_angle})."
            )

        return (
            np.asarray(kept_2d, dtype=np.float64),
            np.asarray(kept_3d, dtype=np.float64),
            telemetry,
        )

    def _label_phase2_confidence(self, telemetry: Phase2CameraTelemetry):
        warnings: List[str] = []

        if telemetry.pnp_correspondences < int(self.config.phase2_confidence_min_correspondences):
            warnings.append("insufficient_correspondences")

        min_inlier_ratio = float(self.config.phase2_confidence_min_inlier_ratio)
        if telemetry.ransac_inlier_ratio is not None and telemetry.ransac_inlier_ratio < min_inlier_ratio:
            warnings.append("low_inlier_ratio")

        if (
            telemetry.reproj_rms_px is not None
            and telemetry.reproj_rms_px > float(self.config.phase2_confidence_max_reproj_rms_px)
        ):
            warnings.append("high_rms")

        telemetry.confidence_label = "low" if warnings else "normal"
        telemetry.confidence_warning = ",".join(warnings) if warnings else None

    def _compute_phase2_reproj_rms(
        self,
        pts_3d_arr: np.ndarray,
        pts_2d: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        K: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> float:
        pts_reproj, _ = cv2.projectPoints(pts_3d_arr, rvec, tvec, K, dist_coeffs)
        pts_reproj = pts_reproj.reshape(-1, 2)
        return float(np.sqrt(np.mean(np.sum((pts_2d - pts_reproj) ** 2, axis=1))))

    @staticmethod
    def _normalize_seed_pair(pair: Tuple[int, int]) -> Tuple[int, int]:
        return (int(pair[0]), int(pair[1]))

    def _build_seed_pair_attempts(
        self,
        cam_i: int,
        cam_j: int,
        ranked_seed_pairs: Optional[List[Tuple[int, int]]],
    ) -> List[Tuple[int, int]]:
        ordered_pairs: List[Tuple[int, int]] = [self._normalize_seed_pair((cam_i, cam_j))]
        if ranked_seed_pairs:
            for pair in ranked_seed_pairs:
                norm_pair = self._normalize_seed_pair(pair)
                if norm_pair not in ordered_pairs:
                    ordered_pairs.append(norm_pair)
        return ordered_pairs

    @staticmethod
    def _summarize_phase2_rms(phase2_telemetry: Dict[int, dict]) -> Tuple[Optional[float], Optional[float]]:
        finite_rms_values: List[float] = []
        for cam_telem in phase2_telemetry.values():
            rms = cam_telem.get('reproj_rms_px') if isinstance(cam_telem, dict) else None
            if rms is None:
                continue
            rms_float = float(rms)
            if np.isfinite(rms_float):
                finite_rms_values.append(rms_float)

        if not finite_rms_values:
            return None, None

        return float(np.max(finite_rms_values)), float(np.median(finite_rms_values))

    def _should_retry_seed_pair_from_phase2_rms(
        self,
        phase2_rms_max_px: Optional[float],
        attempt_index: int,
        total_attempts: int,
    ) -> bool:
        threshold = float(self.config.phase2_pair_fallback_rms_threshold_px)
        if threshold <= 0.0 or phase2_rms_max_px is None:
            return False
        if phase2_rms_max_px <= threshold:
            return False
        return attempt_index < total_attempts - 1

    def _solve_phase2_initial_pose(
        self,
        cid: int,
        pts_2d: np.ndarray,
        pts_3d_arr: np.ndarray,
        K: np.ndarray,
        dist_coeffs: np.ndarray,
        telemetry: Phase2CameraTelemetry,
    ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], np.ndarray, np.ndarray]:
        if not bool(self.config.phase2_ransac_enabled):
            success, rvec, tvec = cv2.solvePnP(
                pts_3d_arr, pts_2d, K, dist_coeffs,
                flags=cv2.SOLVEPNP_EPNP,
            )
            if not success:
                return False, None, None, pts_2d, pts_3d_arr
            return True, rvec, tvec, pts_2d, pts_3d_arr

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d_arr,
            pts_2d,
            K,
            dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP,
            reprojectionError=float(self.config.phase2_ransac_reproj_threshold_px),
            confidence=float(self.config.phase2_ransac_confidence),
        )
        if not success:
            return False, None, None, pts_2d, pts_3d_arr

        if inliers is None or len(inliers) == 0:
            inlier_idx = np.arange(len(pts_2d), dtype=np.int32)
        else:
            inlier_idx = np.asarray(inliers, dtype=np.int32).reshape(-1)

        telemetry.ransac_inliers = int(len(inlier_idx))
        telemetry.ransac_inlier_ratio = float(len(inlier_idx) / max(1, len(pts_2d)))

        refine_pts_2d = pts_2d[inlier_idx]
        refine_pts_3d = pts_3d_arr[inlier_idx]

        min_ratio = float(self.config.phase2_ransac_min_inlier_ratio)
        if telemetry.ransac_inlier_ratio < min_ratio:
            print(
                f"    [RANSAC] cam_{cid}: low inlier ratio "
                f"{telemetry.ransac_inlier_ratio:.3f} < {min_ratio:.3f}; retaining pose with inlier-only refinement."
            )

        if len(refine_pts_2d) < telemetry.prefilter_min_correspondences:
            print(
                f"    [RANSAC] cam_{cid}: only {len(refine_pts_2d)} inliers; "
                f"falling back to all correspondences for refinement floor {telemetry.prefilter_min_correspondences}."
            )
            return True, rvec, tvec, pts_2d, pts_3d_arr

        return True, rvec, tvec, refine_pts_2d, refine_pts_3d
    
    def _collect_valid_frames(
        self,
        observations: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]],
        cam_i: int,
        cam_j: int
    ) -> List[int]:
        """Collect frames where both cameras see both A and B."""
        valid = []
        for fid, cam_obs in observations.items():
            if cam_i in cam_obs and cam_j in cam_obs:
                uvA_i, uvB_i = cam_obs[cam_i]
                uvA_j, uvB_j = cam_obs[cam_j]
                if all(x is not None for x in [uvA_i, uvB_i, uvA_j, uvB_j]):
                    valid.append(fid)
        return valid
    
    def _compute_diagnostics(
        self,
        cam_i: int,
        cam_j: int,
        params_i: np.ndarray,
        params_j: np.ndarray,
        observations: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]],
        valid_frames: List[int],
        K_i: np.ndarray,
        K_j: np.ndarray,
    ) -> dict:
        """Compute diagnostics after optimization."""
        # Build projection matrices
        R_i, _ = cv2.Rodrigues(params_i[:3])
        t_i = params_i[3:6].reshape(3, 1)
        R_j, _ = cv2.Rodrigues(params_j[:3])
        t_j = params_j[3:6].reshape(3, 1)
        
        P_i = K_i @ np.hstack([R_i, t_i])
        P_j = K_j @ np.hstack([R_j, t_j])
        
        wand_lengths = []
        reproj_errors = []
        
        for fid in valid_frames[:200]:
            uvA_i, uvB_i = observations[fid][cam_i]
            uvA_j, uvB_j = observations[fid][cam_j]
            
            # Triangulate
            pts_4d_A = cv2.triangulatePoints(P_i, P_j, 
                                             uvA_i.reshape(2, 1), uvA_j.reshape(2, 1))
            pts_4d_B = cv2.triangulatePoints(P_i, P_j, 
                                             uvB_i.reshape(2, 1), uvB_j.reshape(2, 1))
            
            ptA = (pts_4d_A[:3] / pts_4d_A[3]).flatten()
            ptB = (pts_4d_B[:3] / pts_4d_B[3]).flatten()
            
            wand_lengths.append(np.linalg.norm(ptB - ptA))
            
            # Reprojection error
            proj_Ai = self._project(ptA, R_i, t_i, K_i)
            proj_Aj = self._project(ptA, R_j, t_j, K_j)
            
            # Include both ptA and ptB
            proj_Bi = self._project(ptB, R_i, t_i, K_i)
            proj_Bj = self._project(ptB, R_j, t_j, K_j)
            reproj_errors.append(np.linalg.norm(proj_Ai - uvA_i))
            reproj_errors.append(np.linalg.norm(proj_Bi - uvB_i))
            reproj_errors.append(np.linalg.norm(proj_Aj - uvA_j))
            reproj_errors.append(np.linalg.norm(proj_Bj - uvB_j))
        
        return {
            'baseline_mm': np.linalg.norm(params_j[3:6]),
            'wand_length_median': np.median(wand_lengths) if wand_lengths else 0,
            'wand_length_std': np.std(wand_lengths) if wand_lengths else 0,
            'wand_length_error': abs(np.median(wand_lengths) - self.config.wand_length_mm) if wand_lengths else float('inf'),
            'reproj_err_mean': np.mean(reproj_errors) if reproj_errors else 0,
            'reproj_err_max': np.max(reproj_errors) if reproj_errors else 0,
            'valid_frames': len(valid_frames),
        }
    
    def _validate(self, report: dict, telemetry: P0Telemetry):
        """Validate P0 results. FAIL if constraints violated."""
        print(f"\n{'-'*60}")
        print("[P0 VALIDATION]")
        print(f"{'-'*60}")
        
        b = report['baseline_mm']
        print(f"  Baseline: {b:.2f} mm (recommended min: 50 mm)")
        if b < 50.0:
            print(f"  [WARN] Baseline is below recommended minimum: {b:.2f} mm < 50 mm")
        
        reproj = report.get('reproj_err_mean', 0)
        print(f"  Reproj error mean: {reproj:.2f} px")
        if reproj > 50.0:
            raise P0FailureError(
                f"[P0 FAIL] Reprojection error too high: {reproj:.2f} px",
                P0_REASON_CATASTROPHIC_REPROJECTION,
                telemetry,
            )
        
        wand_err = report.get('wand_length_error', float('inf'))
        print(f"  Wand length error: {wand_err:.4f} mm")
        
        print("[P0 VALIDATION] PASSED")
        print(f"{'-'*60}")
    
    # =========================================================================
    # Phase 2: Calibrate remaining cameras using 3D points from Phase 1
    # =========================================================================
    
    def run_phase2(
        self,
        cam_params: Dict[int, np.ndarray],
        observations: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]],
        points_3d: Dict[int, Tuple[np.ndarray, np.ndarray]],
        camera_settings: Dict[int, dict],
        all_cam_ids: List[int],
    ) -> Dict[int, np.ndarray]:
        """
        Phase 2: Calibrate remaining cameras using 3D points from Phase 1.
        
        For each camera not in cam_params:
        - Collect 2D-3D correspondences from points_3d
        - Solve PnP with frozen intrinsics
        """
        dist_coeffs = np.zeros(5)
        self.last_phase2_telemetry = {}
        
        seed_cam_ids = list(cam_params.keys())
        if len(seed_cam_ids) < 2:
            print("[P0 Phase 2] Need at least two seed cameras. Skipping.")
            return cam_params
        seed_cam_i, seed_cam_j = seed_cam_ids[:2]

        calibrated_cams = set(cam_params.keys())
        remaining_cams = [c for c in all_cam_ids if c not in calibrated_cams]
        
        if not remaining_cams:
            print("[P0 Phase 2] No remaining cameras to calibrate.")
            return cam_params
        
        print(f"\n{'='*60}")
        print(f"[P0 Phase 2] Calibrating {len(remaining_cams)} remaining cameras")
        print(f"{'='*60}")
        print(f"  Already calibrated: {sorted(calibrated_cams)}")
        print(f"  To calibrate: {remaining_cams}")
        
        for cid in remaining_cams:
            print(f"\n  --- Calibrating cam_{cid} ---")
            pts_2d, pts_3d_arr, telemetry = self._build_phase2_correspondences(
                cid,
                observations,
                points_3d,
                cam_params[seed_cam_i],
                cam_params[seed_cam_j],
            )

            if len(pts_2d) < telemetry.prefilter_min_correspondences:
                telemetry.pnp_correspondences = len(pts_2d)
                self._label_phase2_confidence(telemetry)
                self.last_phase2_telemetry[cid] = telemetry.to_dict()
                print(
                    f"    [WARN] Insufficient correspondences after prefilter: "
                    f"{len(pts_2d)} < {telemetry.prefilter_min_correspondences}. Skipping."
                )
                continue

            telemetry.pnp_correspondences = int(len(pts_2d))
            print(f"    Correspondences: {len(pts_2d)}")
            
            K, _, _, _ = self._get_camera_intrinsics(cid, camera_settings)
            success, rvec, tvec, pts_2d_refine, pts_3d_refine = self._solve_phase2_initial_pose(
                cid,
                pts_2d,
                pts_3d_arr,
                K,
                dist_coeffs,
                telemetry,
            )

            if not success:
                mode = "RANSAC PnP" if self.config.phase2_ransac_enabled else "PnP (EPNP)"
                self._label_phase2_confidence(telemetry)
                self.last_phase2_telemetry[cid] = telemetry.to_dict()
                print(f"    [WARN] {mode} failed for cam_{cid}. Skipping.")
                continue

            if self.config.phase2_ransac_enabled:
                print(
                    f"    RANSAC inliers: {telemetry.ransac_inliers}/{telemetry.pnp_correspondences} "
                    f"({telemetry.ransac_inlier_ratio:.3f})"
                )
            
            # Refine with ITERATIVE
            success, rvec, tvec = cv2.solvePnP(
                pts_3d_refine, pts_2d_refine, K, dist_coeffs,
                rvec, tvec, useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success:
                print(f"    [WARN] PnP (ITERATIVE) failed for cam_{cid}. Falling back to initial pose result.")
            
            rvec = rvec.flatten()
            tvec = tvec.flatten()
            
            reproj_err_init = self._compute_phase2_reproj_rms(
                pts_3d_refine, pts_2d_refine, rvec, tvec, K, dist_coeffs,
            )
            
            print(f"    PnP result: RMS = {reproj_err_init:.2f}px")
            
            # Per-camera extrinsic-only optimization (like original Phase 2, but frozen f)
            print(f"    Optimizing extrinsics (frozen intrinsics)...")
            
            x0_cam = np.concatenate([rvec, tvec])  # [rvec(3), tvec(3)]
            
            def residuals_cam(x):
                r = x[:3].reshape(3, 1)
                t = x[3:6].reshape(3, 1)
                pts_proj, _ = cv2.projectPoints(pts_3d_refine, r, t, K, dist_coeffs)
                pts_proj = pts_proj.reshape(-1, 2)
                return (pts_2d_refine - pts_proj).flatten()
            
            result = least_squares(
                residuals_cam, x0_cam,
                method='lm',
                ftol=self.config.ftol,
                xtol=self.config.xtol,
                max_nfev=100,
            )
            
            rvec_opt = result.x[:3]
            tvec_opt = result.x[3:6]
            
            # Compute final reprojection error
            reproj_err_final = self._compute_phase2_reproj_rms(
                pts_3d_refine, pts_2d_refine, rvec_opt, tvec_opt, K, dist_coeffs,
            )
            telemetry.reproj_rms_px = float(reproj_err_final)
            self._label_phase2_confidence(telemetry)
            self.last_phase2_telemetry[cid] = telemetry.to_dict()
            
            print(f"    rvec: [{rvec_opt[0]:.4f}, {rvec_opt[1]:.4f}, {rvec_opt[2]:.4f}]")
            print(f"    tvec: [{tvec_opt[0]:.2f}, {tvec_opt[1]:.2f}, {tvec_opt[2]:.2f}]")
            print(f"    Reproj RMS: {reproj_err_init:.2f} -> {reproj_err_final:.2f}px")
            if telemetry.prefilter_removed > 0 or telemetry.prefilter_starvation_fallback:
                print(
                    f"    Telemetry: kept={telemetry.prefilter_kept}, removed={telemetry.prefilter_removed}, "
                    f"fallback={telemetry.prefilter_starvation_fallback}, confidence={telemetry.confidence_label}"
                )
            
            cam_params[cid] = np.concatenate([rvec_opt, tvec_opt])
        
        print(f"\n[P0 Phase 2] Calibrated {len(cam_params)} cameras total.")
        return cam_params
    
    def triangulate_all_points(
        self,
        cam_i: int,
        cam_j: int,
        params_i: np.ndarray,
        params_j: np.ndarray,
        observations: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]],
        camera_settings: Dict[int, dict],
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Triangulate all 3D wand points using Phase 1 cameras."""
        K_i, _, _, _ = self._get_camera_intrinsics(cam_i, camera_settings)
        K_j, _, _, _ = self._get_camera_intrinsics(cam_j, camera_settings)
        
        R_i, _ = cv2.Rodrigues(params_i[:3])
        t_i = params_i[3:6].reshape(3, 1)
        R_j, _ = cv2.Rodrigues(params_j[:3])
        t_j = params_j[3:6].reshape(3, 1)
        
        P_i = K_i @ np.hstack([R_i, t_i])
        P_j = K_j @ np.hstack([R_j, t_j])
        
        points_3d = {}
        valid_frames = self._collect_valid_frames(observations, cam_i, cam_j)
        
        for fid in valid_frames:
            uvA_i, uvB_i = observations[fid][cam_i]
            uvA_j, uvB_j = observations[fid][cam_j]
            
            pts_4d_A = cv2.triangulatePoints(P_i, P_j, 
                                             uvA_i.reshape(2, 1), uvA_j.reshape(2, 1))
            pts_4d_B = cv2.triangulatePoints(P_i, P_j, 
                                             uvB_i.reshape(2, 1), uvB_j.reshape(2, 1))
            
            XA = (pts_4d_A[:3] / pts_4d_A[3]).flatten()
            XB = (pts_4d_B[:3] / pts_4d_B[3]).flatten()
            
            points_3d[fid] = (XA, XB)
        
        return points_3d
    
    def run_phase3(
        self,
        cam_params: Dict[int, np.ndarray],
        observations: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]],
        camera_settings: Dict[int, dict],
        cam_anchor_id: int = None,  # Camera to anchor (cam_i from Phase 1)
        progress_callback=None
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, Tuple[np.ndarray, np.ndarray]]]:
        """
        Phase 3: Global BA with all cameras and frozen intrinsics.
        
        Joint optimization of all camera extrinsics + 3D points.
        """
        K_by_cam = {}
        for cid in cam_params.keys():
            K_by_cam[cid], _, _, _ = self._get_camera_intrinsics(cid, camera_settings)
        
        all_cam_ids = sorted(cam_params.keys())
        n_cams = len(all_cam_ids)
        cam_id_to_idx = {cid: i for i, cid in enumerate(all_cam_ids)}
        
        print(f"\n{'='*60}")
        print(f"[P0 Phase 3] Global BA with frozen intrinsics")
        print(f"{'='*60}")
        print(f"  Cameras: {all_cam_ids}")
        print("  Frozen intrinsics: per-camera table values")
        
        # Collect valid frames (seen by at least 2 calibrated cameras)
        valid_frames = []
        for fid, cams in observations.items():
            calibrated_in_frame = [c for c in cams.keys() if c in cam_params]
            if len(calibrated_in_frame) >= 2:
                valid_frames.append(fid)
        
        print(f"  Valid frames: {len(valid_frames)}")
        
        if len(valid_frames) < 10:
            print("  [WARN] Not enough frames for Phase 3, skipping.")
            return cam_params, {}
        
        # Triangulate initial 3D points using first available pair per frame
        print("  Triangulating initial points for global BA...")
        pts_3d_init = []
        frame_cams = []  # [(fid, [cams that see this frame])]
        
        for fid in valid_frames:
            cams_in_frame = [c for c in observations[fid].keys() if c in cam_params]
            if len(cams_in_frame) < 2:
                continue
                
            # Use first two cameras for triangulation
            c1, c2 = cams_in_frame[0], cams_in_frame[1]
            p1, p2 = cam_params[c1], cam_params[c2]
            
            R1, _ = cv2.Rodrigues(p1[:3])
            t1 = p1[3:6].reshape(3, 1)
            R2, _ = cv2.Rodrigues(p2[:3])
            t2 = p2[3:6].reshape(3, 1)
            
            P1 = K_by_cam[c1] @ np.hstack([R1, t1])
            P2 = K_by_cam[c2] @ np.hstack([R2, t2])
            
            uvA_1, uvB_1 = observations[fid][c1]
            uvA_2, uvB_2 = observations[fid][c2]
            
            pts_4d_A = cv2.triangulatePoints(P1, P2, uvA_1.reshape(2, 1), uvA_2.reshape(2, 1))
            pts_4d_B = cv2.triangulatePoints(P1, P2, uvB_1.reshape(2, 1), uvB_2.reshape(2, 1))
            
            ptA = (pts_4d_A[:3] / pts_4d_A[3]).flatten()
            ptB = (pts_4d_B[:3] / pts_4d_B[3]).flatten()
            
            pts_3d_init.append(ptA)
            pts_3d_init.append(ptB)
            frame_cams.append((fid, cams_in_frame))
        
        pts_3d_init = np.array(pts_3d_init)
        n_pts = len(pts_3d_init)
        n_frames = len(frame_cams)
        
        print(f"  Initial points: {n_pts}")
        
        # ANCHORED: cam_anchor_id fixed to Phase 2 pose; only remaining cameras are free
        n_cam_params = 6  # Only extrinsics
        if cam_anchor_id is not None and cam_anchor_id in all_cam_ids:
            cam_anchor_pose = cam_params[cam_anchor_id].copy()
            free_cam_ids = [cid for cid in all_cam_ids if cid != cam_anchor_id]
            print(f"  [ANCHORING] Phase 3 BA: cam_{cam_anchor_id} fixed to Phase 2 pose")
            print(f"    cam_{cam_anchor_id} rvec: [{cam_anchor_pose[0]:.4f}, {cam_anchor_pose[1]:.4f}, {cam_anchor_pose[2]:.4f}]")
            print(f"    cam_{cam_anchor_id} tvec: [{cam_anchor_pose[3]:.2f}, {cam_anchor_pose[4]:.2f}, {cam_anchor_pose[5]:.2f}]")
        else:
            cam_anchor_pose = None
            free_cam_ids = all_cam_ids

        n_free_cams = len(free_cam_ids)
        free_cam_id_to_idx = {cid: i for i, cid in enumerate(free_cam_ids)}
        pt_start = n_free_cams * n_cam_params
        
        x0 = np.zeros(pt_start + n_pts * 3)
        for i, cid in enumerate(free_cam_ids):
            x0[i * n_cam_params:(i + 1) * n_cam_params] = cam_params[cid][:6]
        x0[pt_start:] = pts_3d_init.flatten()
        
        # Build sparse Jacobian
        from scipy.sparse import lil_matrix
        
        # Count residuals
        n_res = 0
        for fid, cams_in_frame in frame_cams:
            n_res += 1  # wand length
            n_res += len(cams_in_frame) * 4  # 2 points × 2 coords per camera
        
        n_params = len(x0)
        A_sparsity = lil_matrix((n_res, n_params), dtype=int)
        
        ridx = 0
        for frame_idx, (fid, cams_in_frame) in enumerate(frame_cams):
            idx_ptA = pt_start + frame_idx * 6
            idx_ptB = pt_start + frame_idx * 6 + 3
            
            # Wand length
            A_sparsity[ridx, idx_ptA:idx_ptA+3] = 1
            A_sparsity[ridx, idx_ptB:idx_ptB+3] = 1
            ridx += 1
            
            # Reprojection for each camera
            for cid in cams_in_frame:
                if cid == cam_anchor_id:
                    # Anchor camera: only points (no cam params in state)
                    A_sparsity[ridx:ridx+2, idx_ptA:idx_ptA+3] = 1
                    ridx += 2
                    A_sparsity[ridx:ridx+2, idx_ptB:idx_ptB+3] = 1
                    ridx += 2
                else:
                    # Free camera
                    cam_idx = free_cam_id_to_idx[cid]
                    cam_start = cam_idx * n_cam_params
                    A_sparsity[ridx:ridx+2, cam_start:cam_start+6] = 1
                    A_sparsity[ridx:ridx+2, idx_ptA:idx_ptA+3] = 1
                    ridx += 2
                    A_sparsity[ridx:ridx+2, cam_start:cam_start+6] = 1
                    A_sparsity[ridx:ridx+2, idx_ptB:idx_ptB+3] = 1
                    ridx += 2
        
        print(f"  Residuals: {n_res}, Params: {n_params}")
        
        # Residuals function
        self._phase3_res_count = 0
        def residuals_phase3(x):
            # Extract camera params
            cams = {}
            for cid in all_cam_ids:
                if cid == cam_anchor_id and cam_anchor_pose is not None:
                    # Use fixed Phase 2 pose
                    R, _ = cv2.Rodrigues(cam_anchor_pose[:3])
                    t = cam_anchor_pose[3:6].reshape(3, 1)
                else:
                    cam_idx = free_cam_id_to_idx[cid]
                    p = x[cam_idx * n_cam_params:(cam_idx + 1) * n_cam_params]
                    R, _ = cv2.Rodrigues(p[:3])
                    t = p[3:6].reshape(3, 1)
                cams[cid] = (R, t)
            
            pts = x[pt_start:].reshape(-1, 3)
            
            res = []
            
            # Track stats for progress reporting
            sq_err_len = 0.0
            n_len = 0
            sq_err_proj = 0.0
            n_proj = 0
            
            for frame_idx, (fid, cams_in_frame) in enumerate(frame_cams):
                ptA = pts[frame_idx * 2]
                ptB = pts[frame_idx * 2 + 1]
                
                # Wand length
                wand_len = np.linalg.norm(ptB - ptA)
                d_len = wand_len - self.config.wand_length_mm
                res.append(d_len)
                
                sq_err_len += d_len**2
                n_len += 1
                
                # Reprojection for each camera
                for cid in cams_in_frame:
                    R, t = cams[cid]
                    uvA, uvB = observations[fid][cid]
                    
                    # Project ptA
                    proj_A = self._project(ptA, R, t, K_by_cam[cid])
                    diffA = proj_A - uvA
                    res.extend(diffA.tolist())
                    sq_err_proj += float(np.sum(diffA**2))
                    n_proj += 2
                    
                    # Project ptB
                    proj_B = self._project(ptB, R, t, K_by_cam[cid])
                    diffB = proj_B - uvB
                    res.extend(diffB.tolist())
                    sq_err_proj += float(np.sum(diffB**2))
                    n_proj += 2

            res_arr = np.array(res)

            # Report progress
            self._phase3_res_count += 1
            if progress_callback and self._phase3_res_count % 5 == 0:
                try:
                    rmse_len = np.sqrt(sq_err_len / max(1, n_len))
                    rmse_ray = -1.0
                    rmse_proj = np.sqrt(sq_err_proj / max(1, n_proj))
                    cost = 0.5 * float(np.sum(res_arr**2))
                    progress_callback(
                        "Use PinHole model to initialize camera parameters...",
                        rmse_ray,
                        rmse_len,
                        rmse_proj,
                        cost,
                    )
                except:
                    pass
            
            return res_arr

        
        # Run global BA
        print("  Running global BA...")
        result = least_squares(
            residuals_phase3, x0,
            jac_sparsity=A_sparsity,
            method='trf',
            x_scale='jac',
            f_scale=1.0,
            verbose=1,
            ftol=self.config.ftol,
            xtol=self.config.xtol,
            max_nfev=100,
        )
        
        if not result.success and result.cost > 1e10:
            print(f"  [WARN] Phase 3 BA did not converge (cost={result.cost:.2e}). Returning Phase 2 params.")
            return cam_params, {}  # Return Phase 2 params unchanged, empty points dict
        
        print(f"  Phase 3 cost: {result.cost:.2e}")
        
        # Extract optimized params
        cam_params_opt = {}
        for cid in free_cam_ids:
            cam_idx = free_cam_id_to_idx[cid]
            cam_params_opt[cid] = result.x[cam_idx * n_cam_params:(cam_idx + 1) * n_cam_params]

        if cam_anchor_id is not None and cam_anchor_pose is not None:
            cam_params_opt[cam_anchor_id] = cam_anchor_pose  # Keep Phase 2 pose
        
        # Compute final reprojection error (recompute from optimized result, skipping wand residuals properly)
        all_reproj_errs = []
        pts_final = result.x[pt_start:].reshape(-1, 3)
        for frame_idx, (fid, cams_in_frame) in enumerate(frame_cams):
            ptA = pts_final[frame_idx * 2]
            ptB = pts_final[frame_idx * 2 + 1]
            for cid in cams_in_frame:
                if cid == cam_anchor_id and cam_anchor_pose is not None:
                    R_c, _ = cv2.Rodrigues(cam_anchor_pose[:3])
                    t_c = cam_anchor_pose[3:6].reshape(3, 1)
                else:
                    cam_idx = free_cam_id_to_idx[cid]
                    p = result.x[cam_idx * n_cam_params:(cam_idx + 1) * n_cam_params]
                    R_c, _ = cv2.Rodrigues(p[:3])
                    t_c = p[3:6].reshape(3, 1)
                proj_A = self._project(ptA, R_c, t_c, K_by_cam[cid])
                proj_B = self._project(ptB, R_c, t_c, K_by_cam[cid])
                uvA, uvB = observations[fid][cid]
                all_reproj_errs.append(np.linalg.norm(proj_A - uvA))
                all_reproj_errs.append(np.linalg.norm(proj_B - uvB))
        rms = np.sqrt(np.mean(np.array(all_reproj_errs)**2)) if all_reproj_errs else float('nan')
        print(f"  Final RMS: {rms:.2f}px")
        
        # Re-triangulate final 3D points using Phase 3 optimized cameras
        pts_3d_opt = result.x[pt_start:].reshape(-1, 3)
        points_3d_final = {}
        for frame_idx, (fid, cams_in_frame) in enumerate(frame_cams):
            ptA = pts_3d_opt[frame_idx * 2]
            ptB = pts_3d_opt[frame_idx * 2 + 1]
            points_3d_final[fid] = (ptA.copy(), ptB.copy())

        return cam_params_opt, points_3d_final
    
    def run_all(
        self,
        cam_i: int,
        cam_j: int,
        observations: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]],
        camera_settings: Dict[int, dict],
        all_cam_ids: List[int],
        progress_callback=None,
        ranked_seed_pairs: Optional[List[Tuple[int, int]]] = None,
    ) -> Tuple[Dict[int, np.ndarray], dict]:
        """
        Run full P0 bootstrap: Phase 1 (8-Point + BA) + Phase 2 (PnP) + Phase 3 (Global BA).
        
        All phases use frozen intrinsics (fixed focal length).
        """
        seed_pair_attempts = self._build_seed_pair_attempts(cam_i, cam_j, ranked_seed_pairs)
        total_attempts = 1
        if len(seed_pair_attempts) > 1 and int(self.config.phase2_pair_fallback_max_retries) > 0:
            total_attempts = min(
                len(seed_pair_attempts),
                int(self.config.phase2_pair_fallback_max_retries) + 1,
            )

        best_attempt = None
        chosen_attempt = None

        for attempt_index, seed_pair in enumerate(seed_pair_attempts[:total_attempts]):
            attempt_cam_i, attempt_cam_j = seed_pair
            if attempt_index > 0:
                print(
                    f"\n[P0] Retrying bootstrap with ranked seed pair #{attempt_index + 1}: "
                    f"({attempt_cam_i}, {attempt_cam_j})"
                )

            params_i, params_j, report = self.run(
                attempt_cam_i,
                attempt_cam_j,
                observations,
                camera_settings,
                progress_callback=progress_callback,
            )

            cam_params = {
                attempt_cam_i: params_i,
                attempt_cam_j: params_j,
            }

            print(f"\n[P0] Triangulating 3D points for Phase 2...")
            if progress_callback:
                try:
                    progress_callback("Use PinHole model to initialize camera parameters...", -1, 0, 0, 0)
                except:
                    pass

            points_3d = self.triangulate_all_points(
                attempt_cam_i, attempt_cam_j, params_i, params_j, observations, camera_settings
            )
            report['points_3d'] = points_3d
            print(f"  Triangulated {len(points_3d)} frames")

            if progress_callback:
                try:
                    progress_callback("Use PinHole model to initialize camera parameters...", -1, 0, 0, 0)
                except:
                    pass

            cam_params = self.run_phase2(
                cam_params, observations, points_3d, camera_settings, all_cam_ids
            )
            phase2_telemetry = dict(self.last_phase2_telemetry)
            if 'p0_telemetry' in report:
                report['p0_telemetry']['phase2_cameras'] = phase2_telemetry

            phase2_rms_max_px, phase2_rms_median_px = self._summarize_phase2_rms(phase2_telemetry)
            report['phase2_rms_max_px'] = phase2_rms_max_px
            report['phase2_rms_median_px'] = phase2_rms_median_px

            attempt_state = {
                'cam_i': attempt_cam_i,
                'cam_j': attempt_cam_j,
                'cam_params': cam_params,
                'report': report,
                'points_3d': points_3d,
                'phase2_telemetry': phase2_telemetry,
                'phase2_rms_max_px': phase2_rms_max_px,
            }

            candidate_rms = phase2_rms_max_px if phase2_rms_max_px is not None else float('inf')
            best_rms = (
                best_attempt['phase2_rms_max_px']
                if best_attempt is not None and best_attempt['phase2_rms_max_px'] is not None
                else float('inf')
            )
            if best_attempt is None or candidate_rms < best_rms:
                best_attempt = attempt_state

            if self._should_retry_seed_pair_from_phase2_rms(phase2_rms_max_px, attempt_index, total_attempts):
                print(
                    f"[P0] Phase-2 RMS gate rejected seed pair ({attempt_cam_i}, {attempt_cam_j}): "
                    f"max RMS {phase2_rms_max_px:.2f}px > "
                    f"threshold {float(self.config.phase2_pair_fallback_rms_threshold_px):.2f}px"
                )
                continue

            chosen_attempt = attempt_state
            break

        if chosen_attempt is None:
            chosen_attempt = best_attempt

        if chosen_attempt is None:
            raise RuntimeError("[P0] Bootstrap attempt selection failed unexpectedly")

        cam_i = chosen_attempt['cam_i']
        cam_j = chosen_attempt['cam_j']
        cam_params = chosen_attempt['cam_params']
        report = chosen_attempt['report']
        points_3d = chosen_attempt['points_3d']
        self.last_phase2_telemetry = dict(chosen_attempt['phase2_telemetry'])

        # Phase 3: Global BA with all cameras
        if progress_callback:
            try:
                progress_callback("Use PinHole model to initialize camera parameters...", -1, 0, 0, 0)
            except:
                pass

        cam_params, points_3d_phase3 = self.run_phase3(
            cam_params, observations, camera_settings,
            cam_anchor_id=cam_i,  # Anchor cam_i to Phase 2 pose
            progress_callback=progress_callback
        )
        report['points_3d'] = points_3d_phase3  # Update with Phase 3 points (consistent with final poses)

        
        report['all_cam_ids'] = list(cam_params.keys())
        
        return cam_params, report

