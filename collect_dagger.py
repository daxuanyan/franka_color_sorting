#!/usr/bin/env python3
"""Collect resumable randomized online DAgger correction trajectories."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from benchmark import COLORS, generate_scenario


def parse_target_counts(value: str | None, count: int, seed: int) -> list[str]:
    """Return a deterministic shuffled target schedule with exact quotas."""
    if value is None:
        return [COLORS[index % len(COLORS)] for index in range(count)]
    quotas: dict[str, int] = {}
    for item in value.split(","):
        try:
            color, raw_count = item.strip().split("=", 1)
            quota = int(raw_count)
        except (ValueError, TypeError) as exc:
            raise ValueError("--target-counts must look like yellow=15,green=12") from exc
        if color not in COLORS or quota < 0 or color in quotas:
            raise ValueError(f"Invalid or duplicate target quota: {item!r}")
        quotas[color] = quota
    if set(quotas) != set(COLORS):
        raise ValueError(f"--target-counts must specify exactly: {', '.join(COLORS)}")
    if sum(quotas.values()) != count:
        raise ValueError(f"Target quotas sum to {sum(quotas.values())}, expected --count {count}")
    schedule = [color for color in COLORS for _ in range(quotas[color])]
    random.Random(seed ^ 0xDA66E2).shuffle(schedule)
    return schedule


def finalized_episodes(raw_root: Path) -> dict[int, dict]:
    episodes: dict[int, dict] = {}
    for path in sorted((raw_root / "episodes").glob("episode_*")):
        metadata_path = path / "episode.json"
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") in {"complete", "failed"}:
            episodes[int(metadata["episode_index"])] = metadata
    return episodes


def trainable_episodes(raw_root: Path) -> dict[int, dict]:
    """Return finalized episodes containing at least one valid teacher label."""
    episodes: dict[int, dict] = {}
    for index, metadata in finalized_episodes(raw_root).items():
        episode_dir = raw_root / "episodes" / f"episode_{index:06d}"
        frames_path = episode_dir / "frames.jsonl"
        valid_count = 0
        if frames_path.is_file():
            for line in frames_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                frame = json.loads(line)
                if (
                    bool(frame.get("dagger.label_valid", frame.get("phase") is not None))
                    and frame.get("action.student") is not None
                    and frame.get("action.executed") is not None
                ):
                    valid_count += 1
        if valid_count:
            metadata = dict(metadata)
            metadata["valid_label_frames"] = valid_count
            episodes[index] = metadata
    return episodes


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=30, help="Number of new finalized trajectories to collect.")
    parser.add_argument(
        "--target-counts",
        help="Exact target quotas, e.g. yellow=15,green=12,blue=8,red=5.",
    )
    parser.add_argument("--seed", type=int, default=2026101901)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/smolvla_language_full_30k/checkpoints/030000/pretrained_model"),
    )
    parser.add_argument("--python-sh", type=Path, default=Path.home() / "isaacsim" / "python.sh")
    parser.add_argument("--dagger-round", type=int, default=1)
    parser.add_argument("--dagger-beta", type=float, default=0.8)
    parser.add_argument("--intervention-threshold", type=float, default=0.75)
    parser.add_argument("--act-timeout", type=float, default=120.0)
    parser.add_argument("--process-timeout", type=float, default=240.0)
    parser.add_argument("--max-attempts", type=int, default=45)
    parser.add_argument("--camera-jitter-m", type=float, default=0.03)
    parser.add_argument("--light-min", type=float, default=600.0)
    parser.add_argument("--light-max", type=float, default=1400.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count < 1 or args.max_attempts < args.count:
        raise SystemExit("Require 1 <= --count <= --max-attempts")
    if not 0.0 <= args.dagger_beta <= 1.0:
        raise SystemExit("--dagger-beta must be within 0..1")
    if args.dagger_round < 1 or args.intervention_threshold <= 0.0:
        raise SystemExit("DAgger round and intervention threshold must be positive")
    try:
        target_schedule = parse_target_counts(args.target_counts, args.count, args.seed)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    project_root = Path(__file__).resolve().parent
    raw_root = args.raw_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    python_sh = args.python_sh.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not checkpoint.is_dir():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    if not python_sh.is_file():
        raise SystemExit(f"Isaac Sim launcher not found: {python_sh}")
    output.mkdir(parents=True, exist_ok=False)
    raw_root.mkdir(parents=True, exist_ok=True)

    initial = trainable_episodes(raw_root)
    initial_indices = set(initial)
    rng = random.Random(args.seed)
    attempt_seeds = [rng.randrange(2**32) for _ in range(args.max_attempts)]
    results: list[dict] = []
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "master_seed": args.seed,
        "requested_new_episodes": args.count,
        "initial_finalized_episodes": len(initial),
        "raw_root": str(raw_root),
        "checkpoint": str(checkpoint),
        "dagger_round": args.dagger_round,
        "dagger_beta": args.dagger_beta,
        "intervention_threshold": args.intervention_threshold,
        "target_schedule": target_schedule,
        "target_quotas": {color: target_schedule.count(color) for color in COLORS},
        "randomization": {
            "camera_jitter_m": args.camera_jitter_m,
            "light_range": [args.light_min, args.light_max],
            "object_positions": "collision-free heldout workspace sampling",
        },
    }
    atomic_json(output / "manifest.json", manifest)

    collected = 0
    for attempt, trial_seed in enumerate(attempt_seeds):
        if collected >= args.count:
            break
        # Index by accepted trajectory, so retries cannot skew target quotas.
        target = target_schedule[collected]
        scenario = generate_scenario(
            attempt,
            trial_seed,
            min_targets=1,
            max_targets=1,
            camera_jitter_m=args.camera_jitter_m,
            light_range=(args.light_min, args.light_max),
            fault_profile="none",
            object_set="heldout",
            target_policy="cycle",
        )
        scenario["targets"] = [target]
        trial_dir = output / f"attempt_{attempt:04d}_{target}"
        trial_dir.mkdir()
        scenario_path = trial_dir / "scenario.json"
        atomic_json(scenario_path, scenario)
        command = [
            str(python_sh), str(project_root / "franka_color_sorting.py"),
            "--target", target,
            "--object-set", "heldout",
            "--perception-backend", "generic",
            "--semantic-backend", "clip",
            "--wrist-mode", "servo",
            "--scenario", str(scenario_path),
            "--artifact-dir", str(trial_dir),
            "--test", "--headless",
            "--vla-checkpoint", str(checkpoint),
            "--dagger-dataset", str(raw_root),
            "--dagger-round", str(args.dagger_round),
            "--dagger-beta", str(args.dagger_beta),
            "--dagger-intervention-threshold", str(args.intervention_threshold),
            "--act-control-hz", "5",
            "--act-action-steps", "5",
            "--act-timeout", str(args.act_timeout),
            "--policy-seed", str(trial_seed),
        ]
        before = set(trainable_episodes(raw_root))
        print(
            f"[dagger {collected + 1}/{args.count}] attempt={attempt} "
            f"target={target} seed={trial_seed}",
            flush=True,
        )
        started = time.monotonic()
        timed_out = False
        try:
            completed = subprocess.run(
                command,
                cwd=project_root,
                text=True,
                capture_output=True,
                timeout=args.process_timeout,
                check=False,
            )
            stdout, stderr, return_code = completed.stdout, completed.stderr, completed.returncode
        except subprocess.TimeoutExpired as error:
            timed_out = True
            stdout = error.stdout.decode(errors="replace") if isinstance(error.stdout, bytes) else (error.stdout or "")
            stderr = error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else (error.stderr or "")
            return_code = -1
        duration = time.monotonic() - started
        (trial_dir / "run.log").write_text(stdout + "\n" + stderr, encoding="utf-8", errors="replace")
        after_metadata = trainable_episodes(raw_root)
        new_indices = sorted(set(after_metadata) - before)
        accepted_index = new_indices[-1] if new_indices else None
        accepted = accepted_index is not None and accepted_index not in initial_indices
        episode = after_metadata.get(accepted_index, {}) if accepted_index is not None else {}
        if accepted:
            collected += 1
        result = {
            "attempt": attempt,
            "seed": trial_seed,
            "target": target,
            "scenario": str(scenario_path),
            "episode_index": accepted_index,
            "accepted": accepted,
            "rollout_success": episode.get("success"),
            "num_frames": episode.get("num_frames"),
            "valid_label_frames": episode.get("valid_label_frames", 0),
            "teacher_source": episode.get("teacher_source"),
            "return_code": return_code,
            "timed_out": timed_out,
            "duration_s": round(duration, 3),
            "log": str(trial_dir / "run.log"),
        }
        results.append(result)
        atomic_json(output / "results.json", results)
        print(
            f"  accepted={accepted} episode={accepted_index} "
            f"success={episode.get('success')} frames={episode.get('num_frames')} "
            f"duration={duration:.1f}s return_code={return_code}",
            flush=True,
        )

    eligible = trainable_episodes(raw_root)
    new_finalized = sorted(set(eligible) - initial_indices)
    summary = {
        "requested": args.count,
        "collected": len(new_finalized),
        "attempts": len(results),
        "successful_rollouts": sum(
            eligible[index].get("success") is True for index in new_finalized
        ),
        "failed_rollouts": sum(
            eligible[index].get("success") is not True for index in new_finalized
        ),
        "episode_indices": new_finalized,
        "target_counts": {
            color: sum(
                eligible[index].get("targets") == [color]
                for index in new_finalized
            )
            for color in COLORS
        },
        "valid_label_frames": sum(eligible[index]["valid_label_frames"] for index in new_finalized),
        "rejected_attempts_without_labels": sum(not item["accepted"] for item in results),
    }
    atomic_json(output / "summary.json", summary)
    print(f"[complete] {json.dumps(summary, ensure_ascii=False)}", flush=True)
    if len(new_finalized) != args.count:
        raise SystemExit(f"Collected {len(new_finalized)}/{args.count} trajectories")


if __name__ == "__main__":
    main()
