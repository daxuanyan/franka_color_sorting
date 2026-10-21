"""Compare matched baseline, visual-servo, and recovery fault benchmarks."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from compare_benchmarks import exact_mcnemar_p


def load_run(path: Path) -> dict[str, Any]:
    manifest = json.loads((path / "manifest.json").read_text())
    all_results = json.loads((path / "results.json").read_text())
    invalid = [item for item in all_results if item["status"] != "completed"]
    if invalid:
        statuses = sorted({str(item["status"]) for item in invalid})
        raise ValueError(f"{path} contains non-experimental trial statuses: {statuses}")
    results = all_results
    if len(results) != len(manifest["scenarios"]):
        raise ValueError(
            f"{path} is incomplete: {len(results)}/{len(manifest['scenarios'])} trials have results"
        )
    summary = json.loads((path / "summary.json").read_text())
    return {"path": path, "manifest": manifest, "results": results, "summary": summary}


def scenario_keys(run: dict[str, Any]) -> list[tuple[int, tuple[str, ...], str]]:
    return [
        (int(item["seed"]), tuple(item["targets"]), str(item.get("faults", {}).get("type", "none")))
        for item in run["manifest"]["scenarios"]
    ]


def rate(numerator: int, denominator: int) -> str:
    return "—" if denominator == 0 else f"{numerator}/{denominator} ({100.0 * numerator / denominator:.1f}%)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    runs = [load_run(path.resolve()) for path in args.runs]
    labels = args.labels or [run["path"].name for run in runs]
    if len(labels) != len(runs):
        raise SystemExit("--labels must contain one label per run")
    reference = scenario_keys(runs[0])
    if any(scenario_keys(run) != reference for run in runs[1:]):
        raise SystemExit("Runs are not matched by seed, targets, and fault type")

    lines = [
        "# Matched fault-injection benchmark",
        "",
        f"Matched scenarios: {len(reference)}",
        "",
        "| Metric | " + " | ".join(labels) + " |",
        "|---|" + "---:|" * len(labels),
    ]
    metric_rows: list[tuple[str, list[str]]] = []
    metric_rows.append(("Final task success", [rate(sum(bool(item["success"]) for item in run["results"]), len(run["results"])) for run in runs]))
    metric_rows.append(("First-attempt success", [rate(sum(bool(item["success"]) and int(item.get("recovery_attempts", 0)) == 0 for item in run["results"]), len(run["results"])) for run in runs]))
    metric_rows.append(("Trials invoking recovery", [str(sum(int(item.get("recovery_attempts", 0)) > 0 for item in run["results"])) for run in runs]))
    metric_rows.append(("Conditional recovery success", [
        rate(
            sum(bool(item["success"]) and int(item.get("recovery_attempts", 0)) > 0 for item in run["results"]),
            sum(int(item.get("recovery_attempts", 0)) > 0 for item in run["results"]),
        ) for run in runs
    ]))
    metric_rows.append(("Mean duration", [f'{statistics.fmean(float(item["duration_s"]) for item in run["results"]):.2f} s' for run in runs]))
    metric_rows.append(("Median duration", [f'{statistics.median(float(item["duration_s"]) for item in run["results"]):.2f} s' for run in runs]))
    metric_rows.append(("Injected fault events", [str(sum(int(item.get("fault_events", 0)) for item in run["results"])) for run in runs]))
    metric_rows.append(("Safe exhausted stops", [str(sum("RECOVERY_EXHAUSTED" in str(item.get("failure", "")) for item in run["results"])) for run in runs]))
    for name, values in metric_rows:
        lines.append("| " + name + " | " + " | ".join(values) + " |")

    fault_types = sorted({fault_type for _, _, fault_type in reference})
    lines.extend(["", "## Success by injected fault", ""])
    lines.append("| Fault type | " + " | ".join(labels) + " |")
    lines.append("|---|" + "---:|" * len(labels))
    for fault_type in fault_types:
        values = []
        for run in runs:
            group = [item for item in run["results"] if item.get("fault_type", "none") == fault_type]
            values.append(rate(sum(bool(item["success"]) for item in group), len(group)))
        lines.append("| " + fault_type + " | " + " | ".join(values) + " |")

    lines.extend(["", "## Paired outcome analysis", "", "| Comparison | First only | Second only | Exact McNemar p |", "|---|---:|---:|---:|"])
    outcomes = [[bool(item["success"]) for item in run["results"]] for run in runs]
    for first in range(len(runs)):
        for second in range(first + 1, len(runs)):
            first_only = sum(a and not b for a, b in zip(outcomes[first], outcomes[second]))
            second_only = sum(b and not a for a, b in zip(outcomes[first], outcomes[second]))
            p_value = exact_mcnemar_p(first_only, second_only)
            p_text = "<0.0001" if p_value < 0.0001 else f"{p_value:.4f}"
            lines.append(
                f"| {labels[first]} vs {labels[second]} | {first_only} | {second_only} | "
                f"{p_text} |"
            )
    lines.extend([
        "",
        "All modes use identical recorded seeds, targets, object layouts, camera/light settings, and fault parameters.",
        "Simulator truth is used for offline scoring and environment fault injection, never as a perception fallback.",
    ])
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
