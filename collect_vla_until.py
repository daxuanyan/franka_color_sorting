"""Resume randomized Isaac collection until a raw dataset has N finalized episodes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from create_vla_splits import write_split_manifest


def episode_counts(raw_root: Path) -> dict[str, int]:
    counts = {"success": 0, "failed": 0, "recording": 0}
    for path in sorted((raw_root / "episodes").glob("episode_*")):
        metadata_path = path / "episode.json"
        if not metadata_path.is_file():
            counts["recording"] += 1
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") == "recording":
            counts["recording"] += 1
        elif metadata.get("success") is True:
            counts["success"] += 1
        else:
            counts["failed"] += 1
    return counts


def next_attempt_index(output_parent: Path, prefix: str) -> int:
    index = 0
    while (output_parent / f"{prefix}_{index:04d}").exists():
        index += 1
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--target-episodes", type=int, default=300)
    parser.add_argument("--seed-base", type=int, default=2026103000)
    parser.add_argument("--output-parent", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--output-prefix", default="vla_300_topup")
    parser.add_argument("--record-fps", type=float, default=5.0)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retry-delay", type=float, default=15.0)
    parser.add_argument("--split-output", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=20261018)
    args = parser.parse_args()
    if args.target_episodes < 1:
        raise SystemExit("--target-episodes must be positive")
    if args.retry_delay < 0:
        raise SystemExit("--retry-delay must be non-negative")

    project_root = Path(__file__).resolve().parent
    raw_root = args.raw_root.expanduser().resolve()
    output_parent = args.output_parent.expanduser().resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    attempt = next_attempt_index(output_parent, args.output_prefix)

    while True:
        counts = episode_counts(raw_root)
        finalized = counts["success"] + counts["failed"]
        if counts["recording"]:
            raise SystemExit(
                f"Found {counts['recording']} unfinished recording episode(s); "
                "resolve them before resuming collection"
            )
        if finalized >= args.target_episodes:
            break

        output = output_parent / f"{args.output_prefix}_{attempt:04d}"
        seed = args.seed_base + attempt
        print(
            f"[top-up] finalized={finalized}/{args.target_episodes}, "
            f"success={counts['success']}, failed={counts['failed']}, "
            f"attempt={attempt}, seed={seed}",
            flush=True,
        )
        command = [
            sys.executable, str(project_root / "benchmark.py"),
            "--trials", "1", "--seed", str(seed),
            "--min-targets", "4", "--max-targets", "4",
            "--object-set", "heldout",
            "--perception-backend", "generic", "--semantic-backend", "clip",
            "--wrist-mode", "servo", "--record-fps", str(args.record_fps),
            "--record-dataset", str(raw_root),
            "--record-instruction", "Place all four irregular objects in the target box",
            "--output", str(output), "--timeout", str(args.timeout),
        ]
        before = finalized
        completed = subprocess.run(command, cwd=project_root, check=False)
        after_counts = episode_counts(raw_root)
        after = after_counts["success"] + after_counts["failed"]
        if completed.returncode != 0 or after == before:
            print(
                f"[top-up] attempt produced no finalized episode "
                f"(return_code={completed.returncode}); retrying after {args.retry_delay:g}s",
                flush=True,
            )
            time.sleep(args.retry_delay)
        attempt += 1

    counts = episode_counts(raw_root)
    payload = write_split_manifest(raw_root, args.split_output, seed=args.split_seed)
    split_counts = {
        key: value["count"] for key, value in payload["splits"].items()
    }
    print(
        f"[complete] finalized={counts['success'] + counts['failed']}, "
        f"success={counts['success']}, failed={counts['failed']}; "
        f"split={args.split_output.resolve()} "
        f"counts={split_counts}",
        flush=True,
    )


if __name__ == "__main__":
    main()
