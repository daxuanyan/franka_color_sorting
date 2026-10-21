"""Compare matched randomized benchmark runs and emit a Markdown table."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


METRICS = (
    ("success_rate", "Success rate", "%"),
    ("success_rate_ci95", "Success rate 95% CI", "% CI"),
    ("detection_rate", "Detection rate", "%"),
    ("mean_duration_s", "Mean duration (all trials)", "s"),
    ("mean_success_duration_s", "Mean duration (successful trials)", "s"),
    ("mean_xy_error_mm", "Mean overhead XY error", "mm"),
    ("p95_xy_error_mm", "P95 overhead XY error", "mm"),
    ("mean_wrist_xy_error_mm", "Mean wrist XY error", "mm"),
    ("p95_wrist_xy_error_mm", "P95 wrist XY error", "mm"),
    ("recovery_events", "Recovery events", ""),
    ("recovery_attempts", "Recovery attempts", ""),
    ("recovery_successes", "Successful recoveries", ""),
    ("wrist_refinements", "Wrist corrections accepted", ""),
    ("wrist_rejections", "Wrist corrections rejected", ""),
    ("visual_servo_updates", "Visual-servo updates", ""),
    ("mean_visual_servo_final_xy_error_mm", "Final visual-servo XY error", "mm"),
    ("mean_visual_servo_xy_improvement_mm", "Visual-servo XY improvement", "mm"),
)


def load_run(path: Path) -> tuple[dict[str, Any], list[tuple[int, tuple[str, ...]]], list[bool]]:
    summary = json.loads((path / "summary.json").read_text())
    manifest = json.loads((path / "manifest.json").read_text())
    results = json.loads((path / "results.json").read_text())
    scenarios = [(int(item["seed"]), tuple(item["targets"])) for item in manifest["scenarios"]]
    outcomes = [bool(item["success"]) for item in results if item["status"] != "generated"]
    return summary, scenarios, outcomes


def exact_mcnemar_p(first_only: int, second_only: int) -> float:
    """Two-sided exact McNemar/binomial test for matched binary outcomes."""
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, value) for value in range(min(first_only, second_only) + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def format_value(value: Any, unit: str) -> str:
    if value is None:
        return "—"
    if unit == "%":
        return f"{float(value) * 100:.1f}%"
    if unit == "% CI":
        return f"{float(value[0]) * 100:.1f}%–{float(value[1]) * 100:.1f}%"
    if unit in {"s", "mm"}:
        return f"{float(value):.2f} {unit}"
    return str(value)


def comparison_markdown(paths: list[Path], labels: list[str]) -> str:
    loaded = [load_run(path) for path in paths]
    reference = loaded[0][1]
    for path, (_, scenarios, _) in zip(paths[1:], loaded[1:]):
        if scenarios != reference:
            raise ValueError(f"{path} does not contain the same ordered seeds and targets as {paths[0]}")
    lines = [
        "# Matched benchmark comparison",
        "",
        f"Matched scenarios: {len(reference)}",
        "",
        "| Metric | " + " | ".join(labels) + " |",
        "|---|" + "---:|" * len(labels),
    ]
    for key, title, unit in METRICS:
        values = [format_value(summary.get(key), unit) for summary, _, _ in loaded]
        lines.append("| " + title + " | " + " | ".join(values) + " |")
    lines.append("")
    lines.append("All columns use identical per-trial seeds, target order, object layouts, camera poses, and lighting.")
    lines.extend(["", "## Matched outcome analysis", "", "| Comparison | First only succeeds | Second only succeeds | Exact McNemar p |", "|---|---:|---:|---:|"])
    for first_index in range(len(labels)):
        for second_index in range(first_index + 1, len(labels)):
            first_outcomes = loaded[first_index][2]
            second_outcomes = loaded[second_index][2]
            if len(first_outcomes) != len(second_outcomes):
                raise ValueError("Matched runs do not contain the same number of completed outcomes")
            first_only = sum(first and not second for first, second in zip(first_outcomes, second_outcomes))
            second_only = sum(second and not first for first, second in zip(first_outcomes, second_outcomes))
            p_value = exact_mcnemar_p(first_only, second_only)
            lines.append(f"| {labels[first_index]} vs {labels[second_index]} | {first_only} | {second_only} | {p_value:.4f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    labels = args.labels or [path.name for path in args.runs]
    if len(labels) != len(args.runs):
        raise SystemExit("--labels must contain exactly one label per run")
    markdown = comparison_markdown([path.resolve() for path in args.runs], labels)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown)
        print(args.output.resolve())
    else:
        print(markdown, end="")


if __name__ == "__main__":
    main()
