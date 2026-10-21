#!/usr/bin/env python3
"""Build an auditable held-out-object report from an existing benchmark run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark import generic_perception_outcome


def build_report(run_dir: Path) -> dict[str, Any]:
    results = json.loads((run_dir / "results.json").read_text())
    rows: list[dict[str, Any]] = []
    for result in results:
        trial_dir = Path(result["artifacts"])
        targets = list(result["targets"])
        perception = generic_perception_outcome(trial_dir, targets)
        target_rejections = {
            target: perception["geometry_rejections"][target]
            for target in targets
            if target in perception["geometry_rejections"]
        }
        outcome = (
            "success" if result["success"]
            else "safe_rejection" if perception["safe_rejection"]
            else "failure"
        )
        rows.append({
            "trial_index": result["trial_index"],
            "seed": result["seed"],
            "targets": targets,
            "outcome": outcome,
            "semantic_labels": perception["semantic_labels"],
            "feasible_labels": perception["feasible_labels"],
            "geometry_rejections": target_rejections,
            "placements_verified": result.get("placements_verified", 0),
        })
    completed = [row for row in rows if row["outcome"] in {"success", "safe_rejection", "failure"}]
    motion = [row for row in completed if row["outcome"] != "safe_rejection"]
    successes = [row for row in completed if row["outcome"] == "success"]
    motion_successes = [row for row in motion if row["outcome"] == "success"]
    target_count = sum(len(row["targets"]) for row in completed)
    detected_targets = sum(
        target in row["semantic_labels"] for row in completed for target in row["targets"]
    )
    return {
        "schema_version": 1,
        "source_run": str(run_dir.resolve()),
        "trials": len(completed),
        "task_successes": len(successes),
        "task_success_rate": len(successes) / len(completed) if completed else None,
        "target_semantic_detections": detected_targets,
        "target_semantic_detection_rate": detected_targets / target_count if target_count else None,
        "safe_rejections": sum(row["outcome"] == "safe_rejection" for row in completed),
        "motion_trials": len(motion),
        "motion_successes": len(motion_successes),
        "motion_success_rate": len(motion_successes) / len(motion) if motion else None,
        "claim_boundary": (
            "Category-agnostic grasping of held-out rigid objects within a constrained "
            "top-down parallel-jaw envelope; not unrestricted arbitrary-object grasping."
        ),
        "trials_detail": rows,
    }


def markdown(report: dict[str, Any]) -> str:
    def percent(value: float | None) -> str:
        return "n/a" if value is None else f"{100.0 * value:.1f}%"

    lines = [
        "# Held-out object benchmark",
        "",
        f"- Task success: {report['task_successes']}/{report['trials']} "
        f"({percent(report['task_success_rate'])})",
        f"- Target semantic detection: {report['target_semantic_detections']}/"
        f"{sum(len(row['targets']) for row in report['trials_detail'])} "
        f"({percent(report['target_semantic_detection_rate'])})",
        f"- Executed-grasp success: {report['motion_successes']}/{report['motion_trials']} "
        f"({percent(report['motion_success_rate'])})",
        f"- Geometry-based safe rejections: {report['safe_rejections']}",
        "",
        "| Trial | Target | Outcome | Geometry rejection |",
        "|---:|---|---|---|",
    ]
    for row in report["trials_detail"]:
        rejection = "; ".join(
            f"{name}: {reason}" for name, reason in row["geometry_rejections"].items()
        ) or "—"
        lines.append(
            f"| {row['trial_index']} | {', '.join(row['targets'])} | {row['outcome']} | {rejection} |"
        )
    lines.extend(["", f"Claim boundary: {report['claim_boundary']}", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(args.run_dir)
    output = args.output or args.run_dir / "unseen_object_report"
    output.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    output.with_suffix(".md").write_text(markdown(report))
    print(markdown(report))


if __name__ == "__main__":
    main()
