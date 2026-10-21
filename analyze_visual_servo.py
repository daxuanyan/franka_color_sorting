"""Analyze matched off/single/servo runs and plot command-error convergence."""

from __future__ import annotations

import argparse
import html
import json
import math
import statistics
from pathlib import Path
from typing import Any

from compare_benchmarks import exact_mcnemar_p
from benchmark import control_truth_position


def load_run(path: Path) -> dict[str, Any]:
    manifest = json.loads((path / "manifest.json").read_text())
    results = json.loads((path / "results.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    completed = [item for item in results if item.get("status") != "generated"]
    return {"path": path, "manifest": manifest, "results": completed, "summary": summary}


def scenario_keys(run: dict[str, Any]) -> list[tuple[int, tuple[str, ...]]]:
    return [(int(item["seed"]), tuple(item["targets"])) for item in run["manifest"]["scenarios"]]


def command_series(run: dict[str, Any]) -> list[list[float]]:
    """Return per-target XY error after update 0..N (millimetres)."""
    scenarios = {int(item["trial_index"]): item for item in run["manifest"]["scenarios"]}
    results = {int(item["trial_index"]): item for item in run["results"]}
    series: list[list[float]] = []
    for trial_index, scenario in scenarios.items():
        result = results.get(trial_index)
        if result is None:
            continue
        trace_path = run["path"] / f"trial_{trial_index:04d}" / "visual_servo_trace.jsonl"
        grouped: dict[str, list[dict[str, Any]]] = {}
        if trace_path.exists():
            for line in trace_path.read_text().splitlines():
                if line.strip():
                    record = json.loads(line)
                    if record.get("accepted") is True:
                        grouped.setdefault(str(record["color"]), []).append(record)
        pose_path = run["path"] / f"trial_{trial_index:04d}" / "latest_vision_poses.json"
        initial_poses = json.loads(pose_path.read_text()).get("poses", {}) if pose_path.exists() else {}
        for color in scenario["targets"]:
            truth = control_truth_position(scenario, color)
            records = grouped.get(color, [])
            if records and isinstance(records[0].get("prior_command"), list):
                poses = [records[0]["prior_command"], *(record["command"] for record in records)]
                errors = [
                    math.hypot(float(pose[0]) - truth[0], float(pose[1]) - truth[1]) * 1000.0
                    for pose in poses
                ]
                series.append(errors)
            elif color in initial_poses:
                pose = initial_poses[color]
                series.append([math.hypot(float(pose[0]) - truth[0], float(pose[1]) - truth[1]) * 1000.0])
    return series


def mean_curve(series: list[list[float]], max_updates: int) -> list[float]:
    """Aggregate with last observation carried forward for equal weighting."""
    return [statistics.fmean(values[min(step, len(values) - 1)] for values in series) for step in range(max_updates + 1)]


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def exact_sign_p(first_better: int, second_better: int) -> float:
    """Two-sided exact sign test, excluding ties."""
    total = first_better + second_better
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, index) for index in range(min(first_better, second_better) + 1))
    return min(1.0, 2.0 * tail / (2**total))


def write_svg(path: Path, labels: list[str], curves: list[list[float]]) -> None:
    width, height = 860, 500
    left, right, top, bottom = 80, 30, 55, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    max_step = max(len(curve) for curve in curves) - 1
    max_error = max(max(curve) for curve in curves)
    y_max = max(5.0, math.ceil(max_error / 5.0) * 5.0)
    colors = ("#64748b", "#f59e0b", "#2563eb", "#16a34a")

    def x(step: int) -> float:
        return left + (plot_w * step / max(1, max_step))

    def y(value: float) -> float:
        return top + plot_h * (1.0 - value / y_max)

    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="430" y="28" text-anchor="middle" font-family="sans-serif" font-size="20" font-weight="600">Visual-servo XY command convergence</text>',
    ]
    for tick in range(6):
        value = y_max * tick / 5
        py = y(value)
        chunks.append(f'<line x1="{left}" y1="{py:.1f}" x2="{width-right}" y2="{py:.1f}" stroke="#e2e8f0"/>')
        chunks.append(f'<text x="{left-10}" y="{py+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.0f}</text>')
    for step in range(max_step + 1):
        px = x(step)
        chunks.append(f'<text x="{px:.1f}" y="{height-bottom+24}" text-anchor="middle" font-family="sans-serif" font-size="12">{step}</text>')
    chunks.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#334155"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#334155"/>',
        f'<text x="{left + plot_w/2:.1f}" y="{height-18}" text-anchor="middle" font-family="sans-serif" font-size="14">Accepted wrist updates</text>',
        f'<text x="18" y="{top + plot_h/2:.1f}" text-anchor="middle" transform="rotate(-90 18 {top + plot_h/2:.1f})" font-family="sans-serif" font-size="14">Mean XY error (mm)</text>',
    ])
    for index, (label, curve) in enumerate(zip(labels, curves)):
        padded = [curve[min(step, len(curve) - 1)] for step in range(max_step + 1)]
        points = " ".join(f"{x(step):.1f},{y(value):.1f}" for step, value in enumerate(padded))
        color = colors[index % len(colors)]
        chunks.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for step, value in enumerate(padded):
            chunks.append(f'<circle cx="{x(step):.1f}" cy="{y(value):.1f}" r="3" fill="{color}"/>')
        legend_x = left + index * 210
        chunks.append(f'<line x1="{legend_x}" y1="{height-43}" x2="{legend_x+26}" y2="{height-43}" stroke="{color}" stroke-width="3"/>')
        chunks.append(f'<text x="{legend_x+34}" y="{height-39}" font-family="sans-serif" font-size="13">{html.escape(label)}</text>')
    chunks.append("</svg>")
    path.write_text("\n".join(chunks), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = [load_run(path.resolve()) for path in args.runs]
    labels = args.labels or [run["path"].name for run in runs]
    if len(labels) != len(runs):
        raise SystemExit("--labels must contain one label per run")
    reference = scenario_keys(runs[0])
    if any(scenario_keys(run) != reference for run in runs[1:]):
        raise SystemExit("Runs are not matched: seeds or target order differ")
    all_series = [command_series(run) for run in runs]
    if any(not series for series in all_series):
        raise SystemExit("Each run must contain at least one target localization")
    max_updates = max(max(len(values) - 1 for values in series) for series in all_series)
    curves = [mean_curve(series, max_updates) for series in all_series]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    svg_path = output.with_suffix(".svg")
    write_svg(svg_path, labels, curves)

    lines = [
        "# Matched visual-servo ablation",
        "",
        f"Matched scenarios: {len(reference)}",
        "",
        f"![Visual-servo convergence]({svg_path.name})",
        "",
        "| Metric | " + " | ".join(labels) + " |",
        "|---|" + "---:|" * len(labels),
    ]
    metrics = (
        ("Success", lambda summary, _: f'{summary["successes"]}/{summary["trials_completed"]} ({summary["success_rate"]*100:.1f}%)'),
        ("Mean successful duration", lambda summary, _: fmt(summary.get("mean_success_duration_s")) + " s"),
        ("Accepted wrist updates", lambda summary, _: str(summary.get("visual_servo_updates", 0))),
        ("Rejected wrist observations", lambda summary, _: str(summary.get("wrist_rejections", 0))),
        ("Initial command XY error", lambda _summary, index: fmt(curves[index][0]) + " mm"),
        ("Final command XY error", lambda _summary, index: fmt(curves[index][-1]) + " mm"),
        ("XY improvement", lambda _summary, index: fmt(curves[index][0] - curves[index][-1]) + " mm"),
        ("Recovery attempts", lambda summary, _: str(summary.get("recovery_attempts", 0))),
    )
    for name, formatter in metrics:
        lines.append("| " + name + " | " + " | ".join(formatter(run["summary"], index) for index, run in enumerate(runs)) + " |")
    lines.extend(["", "## Paired outcome analysis", "", "| Comparison | First only | Second only | Exact McNemar p |", "|---|---:|---:|---:|"])
    outcomes = [[bool(item["success"]) for item in run["results"]] for run in runs]
    for first in range(len(runs)):
        for second in range(first + 1, len(runs)):
            first_only = sum(a and not b for a, b in zip(outcomes[first], outcomes[second]))
            second_only = sum(b and not a for a, b in zip(outcomes[first], outcomes[second]))
            lines.append(
                f"| {labels[first]} vs {labels[second]} | {first_only} | {second_only} | "
                f"{exact_mcnemar_p(first_only, second_only):.4f} |"
            )
    lines.extend([
        "",
        "## Paired terminal-error analysis",
        "",
        "| Comparison | First lower | Second lower | Ties | Mean second-vs-first improvement | Exact sign p |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    final_errors = [[values[-1] for values in series] for series in all_series]
    for first in range(len(runs)):
        for second in range(first + 1, len(runs)):
            pairs = list(zip(final_errors[first], final_errors[second]))
            first_lower = sum(a < b - 1e-9 for a, b in pairs)
            second_lower = sum(b < a - 1e-9 for a, b in pairs)
            ties = len(pairs) - first_lower - second_lower
            improvement = statistics.fmean(a - b for a, b in pairs)
            lines.append(
                f"| {labels[first]} vs {labels[second]} | {first_lower} | {second_lower} | {ties} | "
                f"{improvement:.2f} mm | {exact_sign_p(first_lower, second_lower):.4g} |"
            )
    lines.extend([
        "",
        "Curves use last-observation-carried-forward after a mode stops updating, so every target retains equal weight at every update index.",
        "Simulator object poses are used only for offline error scoring, never as controller input.",
    ])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)
    print(svg_path)


if __name__ == "__main__":
    main()
