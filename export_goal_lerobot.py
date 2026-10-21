#!/usr/bin/env python3
"""Export successful raw demonstrations as goal-conditioned policy segments.

The default ``state_one_hot`` mode preserves the ACT dataset format.  The
``language`` mode keeps the original 26D robot state so a VLA policy must use
the per-segment task text instead of receiving a redundant target ID.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from export_lerobot import load_rgb_image
from vla_dataset import load_raw_episode


COLORS = ("red", "green", "blue", "yellow")


def goal_state(state: Iterable[float], object_id: str) -> np.ndarray:
    base = np.asarray(list(state), dtype=np.float32).reshape(-1)
    if base.shape != (26,):
        raise ValueError(f"Expected the recorded 26D state, got {base.shape}")
    if object_id not in COLORS:
        raise ValueError(f"Unknown goal object: {object_id!r}")
    one_hot = np.zeros(len(COLORS), dtype=np.float32)
    one_hot[COLORS.index(object_id)] = 1.0
    return np.concatenate((base, one_hot))


def conditioned_state(
    state: Iterable[float], object_id: str, conditioning: str
) -> np.ndarray:
    """Build an observation state without leaking the language-only target."""
    base = np.asarray(list(state), dtype=np.float32).reshape(-1)
    if base.shape != (26,) or not np.isfinite(base).all():
        raise ValueError(f"Expected 26 finite recorded state values, got {base.shape}")
    if object_id not in COLORS:
        raise ValueError(f"Unknown goal object: {object_id!r}")
    if conditioning == "state_one_hot":
        return goal_state(base, object_id)
    if conditioning == "language":
        return base
    raise ValueError(f"Unknown conditioning mode: {conditioning!r}")


def object_segments(frames: list[dict]) -> list[tuple[str, list[dict]]]:
    """Return contiguous labelled segments, merging adjacent equal labels."""
    segments: list[tuple[str, list[dict]]] = []
    for frame in frames:
        object_id = frame.get("object_id")
        if object_id not in COLORS:
            continue
        if segments and segments[-1][0] == object_id:
            segments[-1][1].append(frame)
        else:
            segments.append((object_id, [frame]))
    return segments


def feature_schema(joint_names: list[str], state_dim: int = 30) -> dict:
    return {
        "observation.images.overhead": {
            "dtype": "image", "shape": (256, 256, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": "image", "shape": (256, 256, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32", "shape": (state_dim,),
            "names": None,
        },
        "action": {
            "dtype": "float32", "shape": (len(joint_names),),
            "names": joint_names,
        },
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-splits", type=Path, required=True)
    parser.add_argument("--output-splits", type=Path, required=True)
    parser.add_argument(
        "--conditioning", choices=("state_one_hot", "language"),
        default="state_one_hot",
        help="Use an ACT target one-hot or language as the only target condition.",
    )
    parser.add_argument("--repo-id")
    args = parser.parse_args()
    repo_id = args.repo_id or (
        "local/franka-heldout-goal-act-256"
        if args.conditioning == "state_one_hot"
        else "local/franka-heldout-language-vla-256"
    )
    state_dim = 30 if args.conditioning == "state_one_hot" else 26

    source_split_payload = json.loads(args.source_splits.read_text(encoding="utf-8"))
    raw_to_split = {}
    for split_name, split in source_split_payload["splits"].items():
        for raw_index in split["raw_episode_indices"]:
            if raw_index in raw_to_split:
                raise ValueError(f"Raw episode {raw_index} occurs in multiple splits")
            raw_to_split[int(raw_index)] = split_name

    raw_paths = []
    for path in sorted((args.raw_root / "episodes").glob("episode_*")):
        metadata, frames = load_raw_episode(path)
        if metadata.get("status") == "complete" and metadata.get("success") is True:
            raw_paths.append((path, metadata, frames))
    if not raw_paths:
        raise SystemExit("No successful raw episodes found")

    first_metadata = raw_paths[0][1]
    fps = int(round(float(first_metadata["fps"])))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    output = args.output.expanduser().resolve()
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=output,
        robot_type=(
            "franka_panda_goal_conditioned"
            if args.conditioning == "state_one_hot"
            else "franka_panda_language_conditioned"
        ),
        features=feature_schema(first_metadata["joint_names"], state_dim),
        use_videos=False,
    )
    split_segments = {name: [] for name in source_split_payload["splits"]}
    manifest = []
    segment_index = 0
    try:
        for episode_path, metadata, frames in raw_paths:
            raw_index = int(metadata["episode_index"])
            split_name = raw_to_split.get(raw_index)
            if split_name is None:
                raise ValueError(f"Successful raw episode {raw_index} is absent from source splits")
            for object_id, segment in object_segments(frames):
                if not segment:
                    continue
                for frame_index, frame in enumerate(segment):
                    terminal = frame_index == len(segment) - 1
                    dataset.add_frame({
                        "observation.images.overhead": load_rgb_image(
                            episode_path / frame["overhead_rgb"], (256, 256)
                        ),
                        "observation.images.wrist": load_rgb_image(
                            episode_path / frame["wrist_rgb"], (256, 256)
                        ),
                        "observation.state": conditioned_state(
                            frame["observation.state"], object_id, args.conditioning
                        ),
                        "action": np.asarray(frame["action"], dtype=np.float32),
                        "next.reward": np.asarray([1.0 if terminal else 0.0], dtype=np.float32),
                        "next.done": np.asarray([terminal], dtype=bool),
                        "task": f"Place the {object_id} irregular object in the target box",
                    })
                dataset.save_episode()
                split_segments[split_name].append(segment_index)
                manifest.append({
                    "lerobot_episode_index": segment_index,
                    "raw_episode_index": raw_index,
                    "split": split_name,
                    "object_id": object_id,
                    "num_frames": len(segment),
                })
                segment_index += 1
    finally:
        dataset.finalize()

    preprocessing = {
        "schema_version": 1,
        "operation": "letterbox_resize",
        "height": 256,
        "width": 256,
        "resampling": "bilinear",
        "padding_rgb": [0, 0, 0],
        "goal_conditioning": (
            {
                "type": "state_one_hot",
                "state_indices": [26, 27, 28, 29],
                "labels": list(COLORS),
            }
            if args.conditioning == "state_one_hot"
            else {
                "type": "natural_language_task",
                "state_target_features": [],
                "templates": [
                    f"Place the {object_id} irregular object in the target box"
                    for object_id in COLORS
                ],
            }
        ),
        "segmentation": "contiguous object_id labels from successful raw demonstrations",
        "source_raw_dataset": str(args.raw_root.expanduser().resolve()),
    }
    (output / "preprocessing.json").write_text(json.dumps(preprocessing, indent=2), encoding="utf-8")
    (output / "segment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    output_split_payload = {
        "schema_version": 1,
        "source_split_file": str(args.source_splits.expanduser().resolve()),
        "mapping_note": "Goal segments inherit their raw parent episode split; no parent crosses splits.",
        "splits": {
            name: {"count": len(indices), "lerobot_episode_indices": indices}
            for name, indices in split_segments.items()
        },
    }
    args.output_splits.write_text(json.dumps(output_split_payload, indent=2), encoding="utf-8")
    print(
        f"Exported {segment_index} goal-conditioned segment(s) to {output}; "
        + ", ".join(f"{name}={len(indices)}" for name, indices in split_segments.items())
    )


if __name__ == "__main__":
    main()
