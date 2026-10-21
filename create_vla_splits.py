"""Create deterministic, episode-level train/validation/test VLA splits."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

from vla_dataset import load_raw_episode


SPLIT_SCHEMA_VERSION = 1


def successful_episode_records(raw_root: Path) -> list[dict[str, Any]]:
    """Return successful raw episodes in the same order as LeRobot export."""
    records = []
    for path in sorted((raw_root / "episodes").glob("episode_*")):
        metadata, frames = load_raw_episode(path)
        if metadata.get("status") != "complete" or metadata.get("success") is not True:
            continue
        if not frames:
            raise ValueError(f"successful episode contains no frames: {path}")
        records.append({
            "raw_episode_index": int(metadata["episode_index"]),
            "lerobot_episode_index": len(records),
            "num_frames": len(frames),
            "instruction": metadata["instruction"],
            "targets": metadata["targets"],
        })
    return records


def make_splits(
    records: list[dict[str, Any]],
    *,
    seed: int,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, list[dict[str, Any]]]:
    if not records:
        raise ValueError("cannot split an empty successful-episode set")
    if any(not math.isfinite(value) or value < 0 for value in ratios):
        raise ValueError("split ratios must be finite and non-negative")
    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
        raise ValueError("split ratios must sum to 1")

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    exact = [len(shuffled) * ratio for ratio in ratios]
    counts = [math.floor(value) for value in exact]
    remainder = len(shuffled) - sum(counts)
    order = sorted(range(3), key=lambda index: (exact[index] - counts[index], -index), reverse=True)
    for index in order[:remainder]:
        counts[index] += 1

    train_end = counts[0]
    validation_end = train_end + counts[1]
    return {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:],
    }


def write_split_manifest(
    raw_root: Path,
    output: Path,
    *,
    seed: int,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, Any]:
    raw_root = raw_root.expanduser().resolve()
    records = successful_episode_records(raw_root)
    splits = make_splits(records, seed=seed, ratios=ratios)
    payload: dict[str, Any] = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "raw_dataset": str(raw_root),
        "seed": seed,
        "ratios": {"train": ratios[0], "validation": ratios[1], "test": ratios[2]},
        "total_successful_episodes": len(records),
        "mapping_note": (
            "lerobot_episode_index is the index assigned by export_lerobot.py when "
            "failed raw episodes are excluded"
        ),
        "splits": {},
    }
    for name, selected in splits.items():
        payload["splits"][name] = {
            "count": len(selected),
            "num_frames": sum(item["num_frames"] for item in selected),
            "raw_episode_indices": sorted(item["raw_episode_index"] for item in selected),
            "lerobot_episode_indices": sorted(item["lerobot_episode_index"] for item in selected),
        }
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("vla_datasets/splits.json"))
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    args = parser.parse_args()
    payload = write_split_manifest(
        args.raw_root,
        args.output,
        seed=args.seed,
        ratios=(args.train_ratio, args.validation_ratio, args.test_ratio),
    )
    counts = {name: value["count"] for name, value in payload["splits"].items()}
    print(
        f"Wrote episode-level split for {payload['total_successful_episodes']} "
        f"successful episodes to {args.output.resolve()}: {counts}"
    )


if __name__ == "__main__":
    main()
