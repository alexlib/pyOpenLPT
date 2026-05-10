from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_THRESHOLDS = {
    "all_bubble_manifest_points": {
        "coverage_C_track": {"min": 0.86},
        "position_error_mean_mm": {"max": 0.027},
        "fragmentation_F_mean_detected_tracks_per_covered_gt": {"max": 1.1},
        "correct_connection_Cr_mean_per_detected_track": {"min": 0.955},
        "radius_error_mean_mm": {"max": 0.016},
        "radius_error_p95_mm": {"max": 0.05},
    }
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed test_STB_Bubble tracking regression gate. This does not regenerate images; "
            "it uses the committed bubble image fixture, runs OpenLPT, evaluates frames 50..99, and checks thresholds."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--config", type=Path, default=script_dir / "config.txt")
    parser.add_argument("--manifest", type=Path, default=script_dir / "manifest.csv")
    parser.add_argument("--result-root", type=Path, default=repo_root / "test" / "results" / "test_STB_Bubble")
    parser.add_argument("--metrics-output", type=Path, default=script_dir / "e2e_metrics.json")
    parser.add_argument("--frame-end", type=int, default=99)
    parser.add_argument("--eval-start", type=int, default=50)
    parser.add_argument("--eval-end", type=int, default=99)
    parser.add_argument("--skip-tracking", action="store_true", help="Only evaluate existing tracking outputs; useful for local debugging.")
    return parser.parse_args(argv)


def _metric_by_label(metrics_summary: dict[str, Any], label: str) -> dict[str, Any] | None:
    for entry in metrics_summary.get("metrics", []):
        if entry.get("label") == label:
            return entry
    return None


def check_thresholds(metrics_summary: dict[str, Any], thresholds: dict[str, dict[str, dict[str, float]]]) -> list[str]:
    failures: list[str] = []
    for label, metric_thresholds in thresholds.items():
        entry = _metric_by_label(metrics_summary, label)
        if entry is None:
            failures.append(f"missing metrics label: {label}")
            continue
        for metric_name, bounds in metric_thresholds.items():
            value = entry.get(metric_name)
            if value is None:
                failures.append(f"{label}.{metric_name} is missing/null")
                continue
            if "min" in bounds and value < bounds["min"]:
                failures.append(f"{label}.{metric_name}={value:.6g} below min {bounds['min']:.6g}")
            if "max" in bounds and value > bounds["max"]:
                failures.append(f"{label}.{metric_name}={value:.6g} above max {bounds['max']:.6g}")
    return failures


def run_command(command: list[str], cwd: Path) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    config = args.config.resolve()
    manifest = args.manifest.resolve()
    result_root = args.result_root.resolve()
    metrics_output = args.metrics_output.resolve()

    if not manifest.exists():
        builder = config.parent / "build_bubble_manifest.py"
        run_command([sys.executable, str(builder), "--manifest", str(manifest)], cwd=repo_root)

    if not args.skip_tracking:
        if result_root.exists():
            shutil.rmtree(result_root)
        run_command([sys.executable, "openlpt.py", str(config)], cwd=repo_root)

    evaluator = config.parent / "evaluate_tracking_metrics.py"
    run_command(
        [
            sys.executable,
            str(evaluator),
            "--manifest",
            str(manifest),
            "--frame-end",
            str(args.frame_end),
            "--eval-start",
            str(args.eval_start),
            "--eval-end",
            str(args.eval_end),
            "--output",
            str(metrics_output),
        ],
        cwd=repo_root,
    )

    metrics_summary = json.loads(metrics_output.read_text(encoding="utf-8"))
    failures = check_thresholds(metrics_summary, DEFAULT_THRESHOLDS)
    if failures:
        print("Bubble tracking regression failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        raise SystemExit(1)

    print("Bubble tracking regression passed.")
    return metrics_summary


def main() -> None:
    run_gate(parse_args())


if __name__ == "__main__":
    main()
