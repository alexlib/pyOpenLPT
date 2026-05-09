# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


DEFAULT_THRESHOLD_MM = 0.2
TRACK_FILE_NAMES = ("LongTrackActive", "LongTrackInactive", "ExitTrack")
REQUIRED_DETECTION_COLUMNS = ["TrackID", "FrameID", "WorldX", "WorldY", "WorldZ", "R3D"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate OpenLPT test_STB_Bubble tracking outputs against the bubble manifest. "
            "By default this evaluates runtime frames 50..99, i.e. the last 50 frames "
            "of a 100-frame fixed fixture."
        )
    )
    parser.add_argument("--manifest", type=Path, default=script_dir / "manifest.csv")
    parser.add_argument("--result-dir", type=Path, default=script_dir.parents[1] / "results" / "test_STB_Bubble" / "ConvergeTrack")
    parser.add_argument("--frame-end", type=int, default=99, help="Final runtime frame ID used in ConvergeTrack filenames.")
    parser.add_argument("--eval-start", type=int, default=50, help="First runtime frame included in metrics.")
    parser.add_argument("--eval-end", type=int, default=99, help="Last runtime frame included in metrics.")
    parser.add_argument("--threshold-mm", type=float, default=DEFAULT_THRESHOLD_MM)
    parser.add_argument("--output", type=Path, default=script_dir / "e2e_metrics.json")
    return parser.parse_args(argv)


def load_tracking_outputs(result_dir: Path, frame_end: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for stem in TRACK_FILE_NAMES:
        path = result_dir / f"{stem}_{frame_end}.csv"
        if not path.exists():
            parts.append(pd.DataFrame(columns=[*REQUIRED_DETECTION_COLUMNS, "source_file"]))
            continue
        data = pd.read_csv(path)
        missing = [column for column in REQUIRED_DETECTION_COLUMNS if column not in data.columns]
        if missing:
            raise ValueError(f"{path} is missing required bubble columns: {missing}")
        if len(data):
            data["source_file"] = path.name
            parts.append(data)
    if not parts:
        return pd.DataFrame(columns=[*REQUIRED_DETECTION_COLUMNS, "source_file"])
    return pd.concat(parts, ignore_index=True)


def filter_frames(data: pd.DataFrame, frame_col: str, start: int, end: int) -> pd.DataFrame:
    return data[(data[frame_col] >= start) & (data[frame_col] <= end)].copy()


def match_detections_to_ground_truth(gt: pd.DataFrame, detections: pd.DataFrame, threshold_mm: float) -> pd.DataFrame:
    gt_by_frame: dict[int, tuple[pd.DataFrame, cKDTree]] = {}
    for frame, frame_gt in gt.groupby("runtime_frame_0based"):
        points = frame_gt[["x", "y", "z"]].to_numpy(float)
        gt_by_frame[int(frame)] = (frame_gt.reset_index(drop=True), cKDTree(points))

    matched: list[pd.DataFrame] = []
    for frame, frame_det in detections.groupby("FrameID"):
        frame_id = int(frame)
        if frame_id not in gt_by_frame:
            continue
        frame_gt, tree = gt_by_frame[frame_id]
        distances, indices = tree.query(frame_det[["WorldX", "WorldY", "WorldZ"]].to_numpy(float), k=1)
        current = frame_det[["TrackID", "FrameID", "WorldX", "WorldY", "WorldZ", "R3D", "source_file"]].copy().reset_index(drop=True)
        current["match_dist_mm"] = distances
        current["gt_track_id"] = frame_gt.loc[indices, "track_id"].to_numpy(int)
        current["gt_radius_mm"] = frame_gt.loc[indices, "radius_mm"].to_numpy(float)
        current["detected_radius_mm"] = current["R3D"].to_numpy(float)
        current["radius_error_mm"] = (current["detected_radius_mm"] - current["gt_radius_mm"]).abs()
        current["correct"] = current["match_dist_mm"] <= threshold_mm
        matched.append(current)

    if not matched:
        return pd.DataFrame(
            columns=[
                "TrackID",
                "FrameID",
                "WorldX",
                "WorldY",
                "WorldZ",
                "R3D",
                "source_file",
                "match_dist_mm",
                "gt_track_id",
                "gt_radius_mm",
                "detected_radius_mm",
                "radius_error_mm",
                "correct",
            ]
        )
    return pd.concat(matched, ignore_index=True)


def compute_correct_connection(matched: pd.DataFrame) -> tuple[float | None, float | None]:
    if len(matched) == 0:
        return None, None
    values: list[float] = []
    for _, group in matched.groupby("TrackID"):
        correct = group[group["correct"]]
        if len(correct) == 0:
            values.append(0.0)
        else:
            modal_correct_count = int(correct["gt_track_id"].value_counts().iloc[0])
            values.append(modal_correct_count / len(group))
    return float(np.mean(values)), float(np.median(values))


def _radius_stats(correct: pd.DataFrame) -> dict[str, float | None]:
    if len(correct) == 0:
        return {
            "radius_error_mean_mm": None,
            "radius_error_median_mm": None,
            "radius_error_p95_mm": None,
            "radius_error_max_mm": None,
        }
    errors = correct["radius_error_mm"].to_numpy(float)
    return {
        "radius_error_mean_mm": float(np.mean(errors)),
        "radius_error_median_mm": float(np.median(errors)),
        "radius_error_p95_mm": float(np.percentile(errors, 95)),
        "radius_error_max_mm": float(np.max(errors)),
    }


def evaluate_subset(gt: pd.DataFrame, detections: pd.DataFrame, label: str, threshold_mm: float) -> dict[str, object]:
    matched = match_detections_to_ground_truth(gt, detections, threshold_mm)
    correct = matched[matched["correct"]].copy()

    gt_tracks = int(gt["track_id"].nunique())
    covered_gt_tracks = int(correct["gt_track_id"].nunique()) if len(correct) else 0
    fragmentation = None
    if len(correct):
        fragmentation = float(correct.groupby("gt_track_id")["TrackID"].nunique().mean())
    cr_mean, cr_median = compute_correct_connection(matched)

    result: dict[str, object] = {
        "label": label,
        "threshold_mm": threshold_mm,
        "gt_rows": int(len(gt)),
        "gt_tracks": gt_tracks,
        "detected_rows": int(len(detections)),
        "detected_tracks": int(detections["TrackID"].nunique()) if len(detections) else 0,
        "matched_rows": int(len(matched)),
        "correct_rows": int(len(correct)),
        "coverage_C_track": covered_gt_tracks / gt_tracks if gt_tracks else 0.0,
        "covered_gt_tracks": covered_gt_tracks,
        "position_error_mean_mm": None if len(correct) == 0 else float(correct["match_dist_mm"].mean()),
        "position_error_median_mm": None if len(correct) == 0 else float(correct["match_dist_mm"].median()),
        "fragmentation_F_mean_detected_tracks_per_covered_gt": fragmentation,
        "correct_connection_Cr_mean_per_detected_track": cr_mean,
        "correct_connection_Cr_median_per_detected_track": cr_median,
    }
    result.update(_radius_stats(correct))
    return result


def evaluate(
    manifest_path: Path,
    result_dir: Path,
    frame_end: int,
    eval_start: int,
    eval_end: int,
    threshold_mm: float,
) -> dict[str, object]:
    manifest = pd.read_csv(manifest_path)
    detections = load_tracking_outputs(result_dir, frame_end)
    detections = filter_frames(detections, "FrameID", eval_start, eval_end)
    manifest_window = filter_frames(manifest, "runtime_frame_0based", eval_start, eval_end)

    track_files = {}
    for stem in TRACK_FILE_NAMES:
        path = result_dir / f"{stem}_{frame_end}.csv"
        track_files[str(path)] = {"exists": path.exists(), "rows": int(len(pd.read_csv(path))) if path.exists() else 0}

    return {
        "manifest_path": str(manifest_path),
        "result_dir": str(result_dir),
        "frame_end": frame_end,
        "eval_frame_range": {"start": eval_start, "end": eval_end},
        "track_files": track_files,
        "metrics": [evaluate_subset(manifest_window, detections, "all_bubble_manifest_points", threshold_mm)],
    }


def main() -> None:
    args = parse_args()
    summary = evaluate(
        manifest_path=args.manifest.resolve(),
        result_dir=args.result_dir.resolve(),
        frame_end=args.frame_end,
        eval_start=args.eval_start,
        eval_end=args.eval_end,
        threshold_mm=args.threshold_mm,
    )
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
