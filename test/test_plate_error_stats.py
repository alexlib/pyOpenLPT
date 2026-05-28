import numpy as np
import cv2

from modules.camera_calibration.view import (
    format_calibration_error_stats,
    compute_plate_triangulation_error_stats,
)


class _KP:
    def __init__(self, x, y):
        self.pt = (float(x), float(y))


def _project(K, R, T, point):
    rvec, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64))
    uv, _ = cv2.projectPoints(
        np.asarray([point], dtype=np.float64),
        rvec,
        np.asarray(T, dtype=np.float64).reshape(3, 1),
        K,
        np.zeros(5),
    )
    return uv.reshape(2)


def test_format_calibration_error_stats_writes_mean_std_or_none():
    assert format_calibration_error_stats((1.25, 0.5)) == "1.25,0.5"
    assert format_calibration_error_stats(None) == "None"


def test_compute_plate_triangulation_error_stats_uses_shared_world_points():
    K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    R0 = np.eye(3)
    T0 = np.zeros((3, 1))
    R1 = np.eye(3)
    T1 = np.array([[-100.0], [0.0], [0.0]])
    points = [np.array([0.0, 0.0, 1000.0]), np.array([50.0, 20.0, 1100.0])]
    saved = {}
    for cid, R, T in [(0, R0, T0), (1, R1, T1)]:
        saved[(cid, "frame.csv")] = {
            "world_coords": points,
            "keypoints": [_KP(*_project(K, R, T, p)) for p in points],
        }
    cams = {
        0: {"K": K, "dist": np.zeros(5), "R": R0, "T": T0},
        1: {"K": K, "dist": np.zeros(5), "R": R1, "T": T1},
    }

    stats = compute_plate_triangulation_error_stats(cams, saved)

    assert stats is not None
    mean, std = stats
    assert mean < 1e-6
    assert std < 1e-6


def test_compute_plate_triangulation_error_stats_returns_none_for_single_camera():
    stats = compute_plate_triangulation_error_stats({0: {}}, {})
    assert stats is None
