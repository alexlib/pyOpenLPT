from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Build a bubble tracking manifest from test_STB_Bubble/data.mat.")
    parser.add_argument("--data", type=Path, default=script_dir / "data.mat")
    parser.add_argument("--manifest", type=Path, default=script_dir / "manifest.csv")
    parser.add_argument("--summary", type=Path, default=script_dir / "reference_summary.json")
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest_dataframe(tracks: np.ndarray, r_mm: np.ndarray) -> pd.DataFrame:
    if tracks.ndim != 2 or tracks.shape[1] < 5:
        raise ValueError("tracks must have columns x,y,z,frame,trackID")
    if r_mm.ndim != 2 or r_mm.shape[1] < 2:
        raise ValueError("r_mm must have columns trackID,radius_mm")

    radius_by_track_id = {int(row[0]): float(row[1]) for row in r_mm}
    manifest = pd.DataFrame(
        {
            "runtime_frame_0based": tracks[:, 3].astype(int) - 1,
            "source_frame_1based": tracks[:, 3].astype(int),
            "track_id": tracks[:, 4].astype(int),
            "x": tracks[:, 0].astype(float),
            "y": tracks[:, 1].astype(float),
            "z": tracks[:, 2].astype(float),
        }
    )
    manifest["radius_mm"] = manifest["track_id"].map(radius_by_track_id).astype(float)
    if manifest["radius_mm"].isna().any():
        missing = sorted(manifest.loc[manifest["radius_mm"].isna(), "track_id"].unique().tolist())
        raise ValueError(f"missing radius for track IDs: {missing[:10]}")
    return manifest


def build_manifest(data_path: Path, manifest_path: Path, summary_path: Path) -> dict[str, object]:
    data = sio.loadmat(data_path)
    manifest = build_manifest_dataframe(data["tracks"], data["r_mm"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)

    summary = {
        "data_path": str(data_path),
        "data_sha256": _sha256(data_path),
        "manifest_path": str(manifest_path),
        "manifest_rows": int(len(manifest)),
        "runtime_frame_range": {
            "start": int(manifest["runtime_frame_0based"].min()),
            "end": int(manifest["runtime_frame_0based"].max()),
        },
        "source_frame_range": {
            "start": int(manifest["source_frame_1based"].min()),
            "end": int(manifest["source_frame_1based"].max()),
        },
        "track_count": int(manifest["track_id"].nunique()),
        "radius_mm": {
            "min": float(manifest["radius_mm"].min()),
            "max": float(manifest["radius_mm"].max()),
            "mean": float(manifest["radius_mm"].mean()),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = build_manifest(args.data.resolve(), args.manifest.resolve(), args.summary.resolve())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
