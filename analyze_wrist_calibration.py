"""Summarize signed wrist RGB-D errors from one or more benchmark runs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def load_samples(run_dir: Path) -> list[dict[str, Any]]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    samples: list[dict[str, Any]] = []
    for scenario in manifest["scenarios"]:
        trial_dir = run_dir / f"trial_{int(scenario['trial_index']):04d}"
        for color in scenario["targets"]:
            pose_path = trial_dir / f"wrist_{color}_pregrasp_poses.json"
            if not pose_path.exists():
                continue
            payload = json.loads(pose_path.read_text())
            estimate = payload.get("poses", {}).get(color)
            if not isinstance(estimate, list) or len(estimate) != 3:
                continue
            truth = scenario["spawn_positions"][color]
            error = [(float(estimate[index]) - float(truth[index])) * 1000.0 for index in range(3)]
            quality = payload.get("quality", {}).get(color, {})
            samples.append(
                {
                    "key": (int(scenario["seed"]), color),
                    "dx_mm": error[0],
                    "dy_mm": error[1],
                    "dz_mm": error[2],
                    "xy_mm": math.hypot(error[0], error[1]),
                    "coverage": quality.get("mask_fraction"),
                    "confidence": quality.get("confidence"),
                }
            )
    return samples


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * fraction
    low, high = math.floor(rank), math.ceil(rank)
    return ordered[low] if low == high else ordered[low] * (high - rank) + ordered[high] * (rank - low)


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {"samples": 0}

    def mean(key: str) -> float | None:
        values = [float(row[key]) for row in samples if row.get(key) is not None]
        return statistics.fmean(values) if values else None

    xy = [float(row["xy_mm"]) for row in samples]
    return {
        "samples": len(samples),
        "mean_dx_mm": mean("dx_mm"),
        "mean_dy_mm": mean("dy_mm"),
        "mean_dz_mm": mean("dz_mm"),
        "mean_xy_mm": statistics.fmean(xy),
        "p95_xy_mm": percentile(xy, 0.95),
        "max_xy_mm": max(xy),
        "mean_coverage": mean("coverage"),
        "mean_confidence": mean("confidence"),
    }


def markdown(runs: list[Path], labels: list[str]) -> str:
    loaded = [load_samples(run) for run in runs]
    summaries = [summarize_samples(samples) for samples in loaded]
    lines = [
        "# Wrist RGB-D calibration analysis",
        "",
        "| Metric | " + " | ".join(labels) + " |",
        "|---|" + "---:|" * len(labels),
    ]
    metrics = (
        ("samples", "Samples", lambda value: str(value)),
        ("mean_dx_mm", "Mean signed X error", lambda value: f"{value:.2f} mm"),
        ("mean_dy_mm", "Mean signed Y error", lambda value: f"{value:.2f} mm"),
        ("mean_dz_mm", "Mean signed Z error", lambda value: f"{value:.2f} mm"),
        ("mean_xy_mm", "Mean XY error", lambda value: f"{value:.2f} mm"),
        ("p95_xy_mm", "P95 XY error", lambda value: f"{value:.2f} mm"),
        ("max_xy_mm", "Maximum XY error", lambda value: f"{value:.2f} mm"),
        ("mean_coverage", "Mean image coverage", lambda value: f"{value * 100:.1f}%"),
        ("mean_confidence", "Mean confidence", lambda value: f"{value:.3f}"),
    )
    for key, title, formatter in metrics:
        values = ["—" if summary.get(key) is None else formatter(summary[key]) for summary in summaries]
        lines.append("| " + title + " | " + " | ".join(values) + " |")
    lines.extend(["", "Signed errors are estimate minus authored benchmark position; physics truth is used only for evaluation."])
    if len(loaded) == 2:
        first = {row["key"]: row for row in loaded[0]}
        second = {row["key"]: row for row in loaded[1]}
        common = sorted(set(first) & set(second))
        improved = sum(second[key]["xy_mm"] < first[key]["xy_mm"] for key in common)
        worsened = sum(second[key]["xy_mm"] > first[key]["xy_mm"] for key in common)
        non_ties = improved + worsened
        if non_ties:
            tail = sum(math.comb(non_ties, value) for value in range(min(improved, worsened) + 1)) / (2**non_ties)
            sign_p = min(1.0, 2.0 * tail)
            lines.extend(
                [
                    "",
                    "## Matched before/after analysis",
                    "",
                    f"Common observations: {len(common)}; improved: {improved}; worsened: {worsened}; "
                    f"two-sided exact sign-test `p={sign_p:.3g}`.",
                ]
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    labels = args.labels or [run.name for run in args.runs]
    if len(labels) != len(args.runs):
        raise SystemExit("--labels must contain exactly one label per run")
    report = markdown([run.resolve() for run in args.runs], labels)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
        print(args.output.resolve())
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
