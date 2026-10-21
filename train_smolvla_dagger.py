#!/usr/bin/env python3
"""Launch leakage-safe SmolVLA fine-tuning on an aggregated DAgger dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--repo-id",
        default="local/franka-heldout-language-dagger-256",
        help="LeRobot dataset identifier recorded in the training configuration.",
    )
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--save-freq", type=int, default=2_000)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--job-name", default="smolvla_dagger_r1_35_finetune_10k")
    parser.add_argument("--dagger-manifest", type=Path)
    parser.add_argument(
        "--dagger-fraction",
        type=float,
        default=0.0,
        help="Per-batch sampling probability assigned to DAgger frames; 0 disables weighting.",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    splits_path = args.splits.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not (dataset_root / "meta" / "info.json").is_file():
        raise SystemExit(f"LeRobot dataset is incomplete: {dataset_root}")
    if not splits_path.is_file():
        raise SystemExit(f"Split manifest not found: {splits_path}")
    if not checkpoint.is_dir():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    if output.exists():
        raise SystemExit(f"Output already exists: {output}")
    if args.steps < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise SystemExit("Invalid training step, batch-size, or worker count")
    if not 0 < args.learning_rate <= 1e-3:
        raise SystemExit("--learning-rate must be within (0, 1e-3]")
    if not 0 <= args.warmup_steps < args.steps:
        raise SystemExit("Require 0 <= --warmup-steps < --steps")
    if not 0.0 <= args.dagger_fraction < 1.0:
        raise SystemExit("--dagger-fraction must be within [0, 1)")
    if args.dagger_fraction > 0.0 and args.dagger_manifest is None:
        raise SystemExit("--dagger-fraction requires --dagger-manifest")

    splits = json.loads(splits_path.read_text(encoding="utf-8"))["splits"]
    train_episodes = [int(index) for index in splits["train"]["lerobot_episode_indices"]]
    eval_episodes = {
        split: {int(index) for index in splits[split]["lerobot_episode_indices"]}
        for split in ("validation", "test")
    }
    if set(train_episodes) & (eval_episodes["validation"] | eval_episodes["test"]):
        raise SystemExit("Training split overlaps validation/test")
    if eval_episodes["validation"] & eval_episodes["test"]:
        raise SystemExit("Validation and test splits overlap")

    if args.dagger_fraction > 0.0:
        manifest = args.dagger_manifest.expanduser().resolve()
        if not manifest.is_file():
            raise SystemExit(f"DAgger segment manifest not found: {manifest}")
        command = [
            sys.executable,
            str(Path(__file__).resolve().with_name("dagger_weighted_train.py")),
            f"--dagger-manifest={manifest}",
            f"--dagger-fraction={args.dagger_fraction}",
            f"--sampler-seed={args.seed}",
        ]
    else:
        executable = shutil.which("lerobot-train")
        if executable is None:
            raise SystemExit("lerobot-train is not available on PATH")
        command = [executable]
    command.extend([
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.root={dataset_root}",
        f"--dataset.episodes={json.dumps(train_episodes, separators=(',', ':'))}",
        f"--policy.path={checkpoint}",
        "--policy.push_to_hub=false",
        f"--policy.optimizer_lr={args.learning_rate}",
        f"--policy.scheduler_warmup_steps={args.warmup_steps}",
        f"--policy.scheduler_decay_steps={args.steps}",
        "--policy.scheduler_decay_lr=2.5e-6",
        f"--output_dir={output}",
        f"--job_name={args.job_name}",
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        f"--num_workers={args.num_workers}",
        f"--save_freq={min(args.save_freq, args.steps)}",
        f"--log_freq={min(args.log_freq, args.steps)}",
        "--eval_freq=0",
        f"--seed={args.seed}",
        "--wandb.enable=false",
    ])
    print(
        f"[train] episodes={len(train_episodes)} validation={len(eval_episodes['validation'])} "
        f"test={len(eval_episodes['test'])} steps={args.steps} lr={args.learning_rate:g} "
        f"dagger_fraction={args.dagger_fraction:g}",
        flush=True,
    )
    completed = subprocess.run(command, check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
