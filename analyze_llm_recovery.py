"""Generate a compact report for matched recovery-supervisor evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_run(path: Path) -> dict[str, Any]:
    return {
        "path": path,
        "cases": json.loads((path / "cases.json").read_text(encoding="utf-8")),
        "traces": json.loads((path / "traces.json").read_text(encoding="utf-8")),
        "summary": json.loads((path / "summary.json").read_text(encoding="utf-8")),
    }


def percent(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:.1f}%"


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
    reference = [case["case_id"] for case in runs[0]["cases"]]
    for run in runs:
        ids = [case["case_id"] for case in run["cases"]]
        trace_ids = [row["case_id"] for row in run["traces"]]
        if ids != reference or trace_ids != reference:
            raise SystemExit(f"Run is incomplete or not matched: {run['path']}")

    lines = [
        "# LLM recovery-supervisor evaluation",
        "",
        f"Matched cases: {len(reference)}",
        "",
        "| Metric | " + " | ".join(labels) + " |",
        "|---|" + "---:|" * len(labels),
        "| Action accuracy | " + " | ".join(percent(run["summary"]["action_accuracy"]) for run in runs) + " |",
        "| Exact action + reason | " + " | ".join(percent(run["summary"]["exact_decision_accuracy"]) for run in runs) + " |",
        "| Policy-valid output | " + " | ".join(percent(run["summary"]["policy_valid_rate"]) for run in runs) + " |",
        "| Unsafe retry rate | " + " | ".join(percent(run["summary"]["unsafe_retry_rate"]) for run in runs) + " |",
        "| Mean latency | " + " | ".join(f'{run["summary"]["mean_latency_ms"]:.2f} ms' for run in runs) + " |",
        "| Total tokens | " + " | ".join(str(run["summary"].get("usage", {}).get("total_tokens", 0)) for run in runs) + " |",
        "",
        "## Exact accuracy by category",
        "",
        "| Category | " + " | ".join(labels) + " |",
        "|---|" + "---:|" * len(labels),
    ]
    categories = sorted(runs[0]["summary"]["by_category"])
    for category in categories:
        lines.append(
            f"| {category} | "
            + " | ".join(percent(run["summary"]["by_category"][category]["exact_decision_accuracy"]) for run in runs)
            + " |"
        )
    lines.extend([
        "",
        "The evaluator performs no robot motion. Raw simulator/tool log text is excluded from the",
        "model context; only typed feedback is provided. Every model decision is checked again by",
        "a deterministic local transition policy before an observation or retry can be executed.",
    ])
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
