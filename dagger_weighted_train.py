#!/usr/bin/env python3
"""Run LeRobot training with an exact base/DAgger frame sampling mixture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def frame_sampling_weights(
    episode_indices: list[int], dagger_episodes: set[int], dagger_fraction: float
) -> tuple[list[float], int, int]:
    """Give both frame groups their requested total probability mass."""
    if not 0.0 < dagger_fraction < 1.0:
        raise ValueError("dagger_fraction must be within (0, 1)")
    dagger_count = sum(index in dagger_episodes for index in episode_indices)
    base_count = len(episode_indices) - dagger_count
    if dagger_count == 0 or base_count == 0:
        raise ValueError("Both base and DAgger frames are required for weighted sampling")
    dagger_weight = dagger_fraction / dagger_count
    base_weight = (1.0 - dagger_fraction) / base_count
    return (
        [dagger_weight if index in dagger_episodes else base_weight for index in episode_indices],
        base_count,
        dagger_count,
    )


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dagger-manifest", type=Path, required=True)
    parser.add_argument("--dagger-fraction", type=float, default=0.5)
    parser.add_argument("--sampler-seed", type=int, default=20261019)
    args, lerobot_args = parser.parse_known_args()
    manifest = json.loads(args.dagger_manifest.expanduser().resolve().read_text(encoding="utf-8"))
    dagger_episodes = {
        int(item["lerobot_episode_index"])
        for item in manifest
        if item.get("source") == "dagger_correction"
    }
    if not dagger_episodes:
        raise SystemExit("The manifest contains no DAgger correction episodes")

    # Import Accelerate/LeRobot before the one-shot patch. Their own DataLoader
    # type references remain the real class, so downstream isinstance checks
    # are unaffected.
    from lerobot.scripts import lerobot_train

    original_dataloader = torch.utils.data.DataLoader

    def weighted_dataloader(dataset, *loader_args, **loader_kwargs):
        # Restore immediately: only the offline training loader is replaced.
        torch.utils.data.DataLoader = original_dataloader
        if loader_kwargs.get("sampler") is not None:
            raise RuntimeError("DAgger weighting cannot be combined with another sampler")
        episode_indices = [int(value) for value in dataset.hf_dataset["episode_index"]]
        weights, base_frames, dagger_frames = frame_sampling_weights(
            episode_indices, dagger_episodes, args.dagger_fraction
        )
        generator = torch.Generator().manual_seed(args.sampler_seed)
        loader_kwargs["sampler"] = torch.utils.data.WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
            generator=generator,
        )
        loader_kwargs["shuffle"] = False
        print(
            f"[weighted-sampler] base_frames={base_frames} dagger_frames={dagger_frames} "
            f"probability={1.0 - args.dagger_fraction:.3f}/{args.dagger_fraction:.3f} "
            f"samples_per_epoch={len(weights)}",
            flush=True,
        )
        return original_dataloader(dataset, *loader_args, **loader_kwargs)

    torch.utils.data.DataLoader = weighted_dataloader
    sys.argv = [sys.argv[0], *lerobot_args]
    lerobot_train.main()


if __name__ == "__main__":
    main()
