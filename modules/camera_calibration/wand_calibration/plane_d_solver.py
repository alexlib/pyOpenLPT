"""
Analytical solver for fixed-normal plane offset `d` from multi-view 2D correspondences.

The plane is parameterized as:  plane_pt = A_anchor + d * plane_n
where A_anchor is the mean camera center and plane_n is the fixed unit normal.

The solver builds pairwise ray-intersection constraints from observed 2D wand
endpoints across cameras sharing a window, refracts them through the window plate,
and solves a linear least-squares system for d.

Fallback gating rejects the solution when the system is ill-conditioned,
under-determined, or geometrically inconsistent.
"""

import numpy as np
import cv2


def _snell_refract(u, n_hat, n_a, n_b):
    """Refract unit vector u through interface with normal n_hat.

    n_hat points toward the incoming ray side.

    Returns
    -------
    t : np.ndarray or None
        Refracted unit direction, or None on TIR / wrong-way ray.
    ok : bool
    """
    eta = n_a / n_b
    c = -np.dot(n_hat, u)
    if c <= 0:
        return None, False  # ray going wrong way
    k = 1.0 - eta ** 2 * (1.0 - c ** 2)
    if k < 0:
        return None, False  # total internal reflection
    t = eta * u + (eta * c - np.sqrt(k)) * n_hat
    t = t / np.linalg.norm(t)
    return t, True


def solve_plane_d_from_correspondences(
    cam_params,
    observations,
    plane_n,
    window_media,
    cam_to_window,
    wid,
    A_anchor,
    d_midpoint,
    active_cam_ids=None,
    verbose=False,
):
    """Solve for the scalar plane offset *d* from multi-view 2D correspondences.

    Parameters
    ----------
    cam_params : dict
        cid -> np.array(11) = [rvec(3), tvec(3), focal_px, cx, cy, k1, k2]
    observations : dict
        fid -> {cid: (uvA, uvB)} where uvA/uvB are (u, v) pixel coords.
    plane_n : np.ndarray, shape (3,)
        Fixed unit normal (camera-side -> object-side).
    window_media : dict
        Keys: 'n1', 'n2', 'n3' (or 'n_object'), 'thickness'.
    cam_to_window : dict
        cid -> wid mapping.
    wid : int or str
        Window id to filter cameras for.
    A_anchor : np.ndarray, shape (3,)
        Mean camera center = anchor for parameterization.
    d_midpoint : float
        Legacy midpoint-depth seed (used for fallback gating).
    active_cam_ids : list/set or None
        Active camera ids; None uses all in cam_params.
    verbose : bool

    Returns
    -------
    dict with keys: d_solved, plane_pt_solved, accepted, fallback_reason,
        A_shape, rank, cond, n_equations, n_pairs_used, residual_rms,
        camera_side_ok.
    """
    n1 = window_media.get('n1', 1.0)
    n2 = window_media.get('n2', 1.5)
    n3 = window_media.get('n3', window_media.get('n_object', 1.333))

    n_hat = -plane_n

    active_set = set(active_cam_ids) if active_cam_ids is not None else None

    A_vals = []
    b_vals = []
    n_pairs_used = 0

    for fid, frame_obs in observations.items():
        window_cams = [
            cid for cid in frame_obs
            if cam_to_window.get(cid) == wid
            and (active_set is None or cid in active_set)
            and cid in cam_params
        ]
        if len(window_cams) < 2:
            continue

        # Process endpoint A (idx=0) and endpoint B (idx=1) independently
        for uv_idx in (0, 1):
            cam_data = {}

            for cid in window_cams:
                p = cam_params[cid]
                uv_pair = frame_obs[cid]
                uv = np.asarray(uv_pair[uv_idx], dtype=np.float64)
                u_px, v_px = uv[0], uv[1]

                f, cx, cy = p[6], p[7], p[8]

                R, _ = cv2.Rodrigues(p[0:3])
                tvec = p[3:6]
                C_j = -R.T @ tvec

                # K_inv @ [u, v, 1] -> camera-frame ray -> world-frame ray
                u_cam = np.array([(u_px - cx) / f, (v_px - cy) / f, 1.0])
                u_cam = u_cam / np.linalg.norm(u_cam)
                u_ij = R.T @ u_cam
                u_ij = u_ij / np.linalg.norm(u_ij)

                # Anchor-relative first-interface intersection:
                # Q_ij(d) = Q_ij0 + d * q_ij  where  q_ij = u_ij / dot(n, u_ij)
                denom = np.dot(plane_n, u_ij)
                if abs(denom) < 1e-9:
                    continue

                Q_ij0 = C_j + (np.dot(plane_n, A_anchor) - np.dot(plane_n, C_j)) / denom * u_ij
                q_ij = u_ij / denom

                # Snell refraction: air(n1)->glass(n2)->water(n3), constant w.r.t. d
                t1, ok1 = _snell_refract(u_ij, n_hat, n1, n2)
                if not ok1:
                    continue

                t2, ok2 = _snell_refract(t1, n_hat, n2, n3)
                if not ok2:
                    continue

                v_ij = t2
                cam_data[cid] = (Q_ij0, q_ij, v_ij)

            # Pairwise constraint: dot(q_j - q_k, cross(v_j, v_k)) * d = -dot(Q_j0 - Q_k0, cross(v_j, v_k))
            valid_cids = list(cam_data.keys())
            for j_idx in range(len(valid_cids)):
                for k_idx in range(j_idx + 1, len(valid_cids)):
                    cid_j = valid_cids[j_idx]
                    cid_k = valid_cids[k_idx]

                    Q_j0, q_j, v_j = cam_data[cid_j]
                    Q_k0, q_k, v_k = cam_data[cid_k]

                    c_i = np.cross(v_j, v_k)
                    sin_angle = np.linalg.norm(c_i)
                    if sin_angle < np.sin(np.radians(1.0)):
                        continue

                    A_i = np.dot(q_j - q_k, c_i)
                    b_i = -np.dot(Q_j0 - Q_k0, c_i)

                    A_vals.append(A_i)
                    b_vals.append(b_i)
                    n_pairs_used += 1

    # ---- Solve Ax = b via least-squares ----
    n_equations = len(A_vals)

    if n_equations == 0:
        return _make_result(
            d_solved=np.nan,
            plane_pt_solved=None,
            accepted=False,
            fallback_reason="insufficient_equations",
            A_shape=(0, 1),
            rank=0,
            cond=float('inf'),
            n_equations=0,
            n_pairs_used=0,
            residual_rms=0.0,
            camera_side_ok=False,
        )

    A_mat = np.array(A_vals, dtype=np.float64).reshape(-1, 1)
    b_vec = np.array(b_vals, dtype=np.float64)

    result = np.linalg.lstsq(A_mat, b_vec, rcond=None)
    d_solved = float(result[0][0])
    rank = int(result[2])

    if A_mat.shape[0] >= 2:
        cond = float(np.linalg.cond(A_mat))
    else:
        cond = float('inf')

    residuals = A_mat.flatten() * d_solved - b_vec
    residual_rms = float(np.sqrt(np.mean(residuals ** 2)))

    if verbose:
        print(f"[plane_d_solver] n_eq={n_equations}, rank={rank}, "
              f"cond={cond:.2e}, d_solved={d_solved:.4f}, rms={residual_rms:.6f}")

    # ---- Fallback gating ----
    fallback_reason = None
    camera_side_ok = True

    if n_equations < 2:
        fallback_reason = "insufficient_equations"

    if fallback_reason is None and (rank == 0 or cond > 1e8):
        fallback_reason = "ill_conditioned"

    if fallback_reason is None and not np.isfinite(d_solved):
        fallback_reason = "non_finite"

    if fallback_reason is None:
        if abs(d_solved - d_midpoint) > 0.5 * max(abs(d_midpoint), 1.0):
            fallback_reason = "outlier_d"

    # Camera-side check: dot(plane_n, C_j - plane_pt) must be < 0 for all active cams
    plane_pt_solved = A_anchor + d_solved * plane_n if np.isfinite(d_solved) else None

    if fallback_reason is None and plane_pt_solved is not None:
        check_cids = active_cam_ids if active_cam_ids is not None else list(cam_params.keys())
        for cid in check_cids:
            if cid not in cam_params:
                continue
            p = cam_params[cid]
            R, _ = cv2.Rodrigues(p[0:3])
            C_j = -R.T @ p[3:6]
            if np.dot(plane_n, C_j - plane_pt_solved) >= 0:
                camera_side_ok = False
                fallback_reason = "camera_side_violation"
                break

    accepted = fallback_reason is None

    return _make_result(
        d_solved=d_solved,
        plane_pt_solved=plane_pt_solved,
        accepted=accepted,
        fallback_reason=fallback_reason,
        A_shape=A_mat.shape,
        rank=rank,
        cond=cond,
        n_equations=n_equations,
        n_pairs_used=n_pairs_used,
        residual_rms=residual_rms,
        camera_side_ok=camera_side_ok,
    )


def _make_result(
    d_solved, plane_pt_solved, accepted, fallback_reason,
    A_shape, rank, cond, n_equations, n_pairs_used,
    residual_rms, camera_side_ok,
):
    """Construct the standard return dict."""
    return {
        'd_solved': d_solved,
        'plane_pt_solved': plane_pt_solved,
        'accepted': accepted,
        'fallback_reason': fallback_reason,
        'A_shape': A_shape,
        'rank': rank,
        'cond': cond,
        'n_equations': n_equations,
        'n_pairs_used': n_pairs_used,
        'residual_rms': residual_rms,
        'camera_side_ok': camera_side_ok,
    }
