"""Validate raw Isaac demonstrations and, optionally, their LeRobot export."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from vla_dataset import validate_raw_episode


def validate_raw_root(root: Path, *, skip_empty: bool = False) -> list[dict]:
    paths = sorted((root / "episodes").glob("episode_*"))
    if not paths:
        raise ValueError(f"no raw episodes found under {root}")
    validated = []
    for path in paths:
        try:
            validated.append(validate_raw_episode(path))
        except ValueError as exc:
            if skip_empty and str(exc).startswith("episode contains no frames:"):
                continue
            raise
    if not validated:
        raise ValueError(f"no non-empty raw episodes found under {root}")
    return validated


def validate_lerobot(root: Path, repo_id: str, cache_dir: Path) -> dict:
    # Hugging Face otherwise writes locks under ~/.cache, which may be read-only
    # on robot workstations and in CI containers.
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir / "datasets"))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id, root=root.expanduser().resolve())
    required = {
        "observation.images.overhead", "observation.images.wrist",
        "observation.state", "action", "next.reward", "next.done",
    }
    missing = required.difference(dataset.features)
    if missing:
        raise ValueError(f"LeRobot dataset is missing features: {sorted(missing)}")
    if len(dataset) == 0 or dataset.num_episodes == 0:
        raise ValueError("LeRobot dataset is empty")

    terminal_episodes: set[int] = set()
    state_dim = action_dim = None
    for index in range(len(dataset)):
        sample = dataset[index]
        state = np.asarray(sample["observation.state"])
        action = np.asarray(sample["action"])
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f"non-finite state/action in LeRobot frame {index}")
        state_dim, action_dim = int(state.size), int(action.size)
        for key in ("observation.images.overhead", "observation.images.wrist"):
            image = np.asarray(sample[key])
            if image.ndim != 3 or not np.isfinite(image).all():
                raise ValueError(f"invalid {key} in LeRobot frame {index}")
        if bool(np.asarray(sample["next.done"]).item()):
            terminal_episodes.add(int(np.asarray(sample["episode_index"]).item()))
    if len(terminal_episodes) != dataset.num_episodes:
        raise ValueError(
            f"found {len(terminal_episodes)} terminal frames for "
            f"{dataset.num_episodes} episodes"
        )
    return {
        "episodes": int(dataset.num_episodes), "frames": len(dataset),
        "fps": int(dataset.fps), "state_dim": state_dim, "action_dim": action_dim,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("--lerobot-root", type=Path)
    parser.add_argument("--repo-id", default="local/franka-color-sorting")
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/franka_lerobot_cache"))
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="ignore finalized zero-frame audit episodes (strict by default)",
    )
    args = parser.parse_args()

    raw_root = args.raw_root.expanduser().resolve()
    raw = validate_raw_root(raw_root, skip_empty=args.skip_empty)
    total_episode_dirs = len(list((raw_root / "episodes").glob("episode_*")))
    skipped_empty = total_episode_dirs - len(raw)
    print(
        f"Raw dataset OK: {len(raw)} episode(s), "
        f"{sum(item['num_frames'] for item in raw)} frame(s), "
        f"{sum(item['success'] for item in raw)} successful"
    )
    if skipped_empty:
        print(f"  skipped {skipped_empty} finalized zero-frame audit episode(s)")
    for item in raw:
        print(
            f"  episode_{item['episode_index']:06d}: {item['status']}, "
            f"{item['num_frames']} frames @ {item['fps']:g} FPS, "
            f"state={item['state_dim']}, action={item['action_dim']}"
        )
    if args.lerobot_root is not None:
        converted = validate_lerobot(args.lerobot_root, args.repo_id, args.cache_dir)
        print(
            f"LeRobot dataset OK: {converted['episodes']} episode(s), "
            f"{converted['frames']} frame(s) @ {converted['fps']} FPS, "
            f"state={converted['state_dim']}, action={converted['action_dim']}"
        )


if __name__ == "__main__":
    main()
