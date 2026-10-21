#!/usr/bin/env python3
"""Aggregate base demonstrations and valid DAgger corrections into LeRobot v3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from export_goal_lerobot import conditioned_state, feature_schema, object_segments
from export_lerobot import load_rgb_image
from vla_dataset import load_raw_episode


def valid_dagger_frames(frames: list[dict]) -> list[dict]:
    """Keep only frames for which the shadow expert produced a real label."""
    return [
        frame
        for frame in frames
        if bool(frame.get("dagger.label_valid", frame.get("phase") is not None))
        and frame.get("action.student") is not None
        and frame.get("action.executed") is not None
    ]


def add_frames(dataset, episode_path: Path, frames: list[dict], instruction: str, success: bool) -> None:
    last = len(frames) - 1
    for index, frame in enumerate(frames):
        terminal = index == last
        object_id = frame.get("object_id")
        dataset.add_frame({
            "observation.images.overhead": load_rgb_image(
                episode_path / frame["overhead_rgb"], (256, 256)
            ),
            "observation.images.wrist": load_rgb_image(
                episode_path / frame["wrist_rgb"], (256, 256)
            ),
            "observation.state": conditioned_state(
                frame["observation.state"], object_id, "language"
            ),
            # The raw DAgger schema deliberately stores the teacher label in
            # `action`; student and executed actions remain audit metadata.
            "action": np.asarray(frame["action"], dtype=np.float32),
            "next.reward": np.asarray([1.0 if terminal and success else 0.0], dtype=np.float32),
            "next.done": np.asarray([terminal], dtype=bool),
            "task": instruction,
        })
    dataset.save_episode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_raw_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dagger-root", type=Path, action="append", required=True)
    parser.add_argument("--source-splits", type=Path, required=True)
    parser.add_argument("--output-splits", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/franka-heldout-language-dagger-256")
    args = parser.parse_args()

    source_splits = json.loads(args.source_splits.read_text(encoding="utf-8"))
    raw_to_split: dict[int, str] = {}
    for split_name, split in source_splits["splits"].items():
        for raw_index in split["raw_episode_indices"]:
            raw_to_split[int(raw_index)] = split_name

    base_episodes = []
    for path in sorted((args.base_raw_root / "episodes").glob("episode_*")):
        metadata, frames = load_raw_episode(path)
        if metadata.get("status") == "complete" and metadata.get("success") is True:
            base_episodes.append((path, metadata, frames))
    if not base_episodes:
        raise SystemExit("No successful base demonstrations found")

    dagger_episodes = []
    for root in args.dagger_root:
        for path in sorted((root / "episodes").glob("episode_*")):
            metadata, frames = load_raw_episode(path)
            if metadata.get("dataset_kind") != "dagger_correction":
                raise ValueError(f"Not a DAgger correction episode: {path}")
            valid = valid_dagger_frames(frames)
            if valid:
                dagger_episodes.append((path, metadata, valid))
    if not dagger_episodes:
        raise SystemExit("No valid DAgger correction frames found")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    first_metadata = base_episodes[0][1]
    fps = int(round(float(first_metadata["fps"])))
    output = args.output.expanduser().resolve()
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=fps,
        root=output,
        robot_type="franka_panda_language_dagger",
        features=feature_schema(first_metadata["joint_names"], 26),
        use_videos=False,
    )
    split_indices = {name: [] for name in source_splits["splits"]}
    manifest = []
    episode_index = 0
    try:
        for path, metadata, frames in base_episodes:
            raw_index = int(metadata["episode_index"])
            split_name = raw_to_split.get(raw_index)
            if split_name is None:
                raise ValueError(f"Base episode {raw_index} is absent from source splits")
            for object_id, segment in object_segments(frames):
                instruction = f"Place the {object_id} irregular object in the target box"
                add_frames(dataset, path, segment, instruction, True)
                split_indices[split_name].append(episode_index)
                manifest.append({
                    "lerobot_episode_index": episode_index,
                    "source": "base_expert",
                    "raw_episode_index": raw_index,
                    "split": split_name,
                    "object_id": object_id,
                    "num_frames": len(segment),
                })
                episode_index += 1

        for path, metadata, frames in dagger_episodes:
            object_id = str(metadata["targets"][0])
            add_frames(dataset, path, frames, metadata["instruction"], bool(metadata.get("success")))
            split_indices["train"].append(episode_index)
            manifest.append({
                "lerobot_episode_index": episode_index,
                "source": "dagger_correction",
                "dagger_root": str(path.parents[1].resolve()),
                "raw_episode_index": int(metadata["episode_index"]),
                "split": "train",
                "object_id": object_id,
                "num_frames": len(frames),
                "rollout_success": bool(metadata.get("success")),
                "dagger_round": metadata.get("dagger_round"),
                "beta": metadata.get("beta"),
            })
            episode_index += 1
    finally:
        dataset.finalize()

    args.output_splits.write_text(json.dumps({
        "schema_version": 1,
        "source_split_file": str(args.source_splits.resolve()),
        "mapping_note": "Base parents preserve their split; all online DAgger corrections are train-only.",
        "splits": {
            name: {"count": len(indices), "lerobot_episode_indices": indices}
            for name, indices in split_indices.items()
        },
    }, indent=2), encoding="utf-8")
    (output / "segment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (output / "preprocessing.json").write_text(json.dumps({
        "schema_version": 1,
        "operation": "letterbox_resize",
        "height": 256,
        "width": 256,
        "conditioning": "natural_language_task",
        "action_label": "shadow_expert_joint_position_target",
        "dagger_invalid_labels_filtered": True,
        "base_raw_dataset": str(args.base_raw_root.resolve()),
        "dagger_roots": [str(path.resolve()) for path in args.dagger_root],
    }, indent=2), encoding="utf-8")
    print(
        f"Exported {episode_index} episodes to {output}; "
        f"DAgger episodes={len(dagger_episodes)}, valid correction frames="
        f"{sum(len(frames) for _, _, frames in dagger_episodes)}"
    )


if __name__ == "__main__":
    main()
