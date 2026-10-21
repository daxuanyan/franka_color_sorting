"""Convert raw Isaac expert episodes into a local LeRobotDataset v3 dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from vla_dataset import load_raw_episode


def episode_paths(root: Path, include_failures: bool) -> list[Path]:
    selected = []
    for path in sorted((root / "episodes").glob("episode_*")):
        metadata, _ = load_raw_episode(path)
        if metadata.get("status") == "complete" and metadata.get("success") is True:
            selected.append(path)
        elif include_failures and metadata.get("status") == "failed":
            selected.append(path)
    return selected


def load_rgb_image(path: Path, image_size: tuple[int, int] | None = None) -> np.ndarray:
    """Load RGB and optionally letterbox to (height, width) without distortion."""
    with Image.open(path) as source:
        image = source.convert("RGB")
        if image_size is not None:
            height, width = image_size
            image = ImageOps.pad(
                image, (width, height), method=Image.Resampling.BILINEAR,
                color=(0, 0, 0), centering=(0.5, 0.5),
            )
        return np.asarray(image)


def feature_schema(
    metadata: dict,
    first_frame: dict,
    episode_dir: Path,
    image_size: tuple[int, int] | None = None,
) -> dict:
    overhead_shape = load_rgb_image(
        episode_dir / first_frame["overhead_rgb"], image_size
    ).shape
    wrist_shape = load_rgb_image(
        episode_dir / first_frame["wrist_rgb"], image_size
    ).shape
    return {
        "observation.images.overhead": {
            "dtype": "image", "shape": tuple(overhead_shape),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": "image", "shape": tuple(wrist_shape),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32", "shape": (len(first_frame["observation.state"]),),
            "names": None,
        },
        "action": {
            "dtype": "float32", "shape": (len(first_frame["action"]),),
            "names": metadata["joint_names"],
        },
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repo-id", default="local/franka-color-sorting")
    parser.add_argument("--include-failures", action="store_true")
    parser.add_argument("--use-videos", action="store_true")
    parser.add_argument(
        "--image-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"),
        help="Letterbox every RGB observation to a common H W (for example 256 256).",
    )
    args = parser.parse_args()
    image_size = tuple(args.image_size) if args.image_size is not None else None
    if image_size is not None and any(value <= 0 for value in image_size):
        raise SystemExit("--image-size dimensions must be positive")

    paths = episode_paths(args.raw_root, args.include_failures)
    if not paths:
        raise SystemExit("No completed raw episodes found")
    first_metadata, first_frames = load_raw_episode(paths[0])
    if not first_frames:
        raise SystemExit("First episode contains no frames")
    fps = int(round(float(first_metadata["fps"])))
    if abs(float(first_metadata["fps"]) - fps) > 1e-6:
        raise SystemExit("LeRobot export requires an integer FPS")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=fps,
        root=args.output.expanduser().resolve(),
        robot_type="franka_panda",
        features=feature_schema(first_metadata, first_frames[0], paths[0], image_size),
        use_videos=args.use_videos,
    )
    try:
        for episode_dir in paths:
            metadata, frames = load_raw_episode(episode_dir)
            if int(round(float(metadata["fps"]))) != fps:
                raise ValueError(f"FPS mismatch in {episode_dir}")
            last = len(frames) - 1
            for index, frame in enumerate(frames):
                success_terminal = index == last and metadata.get("success") is True
                dataset.add_frame({
                    "observation.images.overhead": load_rgb_image(
                        episode_dir / frame["overhead_rgb"], image_size
                    ),
                    "observation.images.wrist": load_rgb_image(
                        episode_dir / frame["wrist_rgb"], image_size
                    ),
                    "observation.state": np.asarray(frame["observation.state"], dtype=np.float32),
                    "action": np.asarray(frame["action"], dtype=np.float32),
                    "next.reward": np.asarray([1.0 if success_terminal else 0.0], dtype=np.float32),
                    "next.done": np.asarray([index == last], dtype=bool),
                    "task": metadata["instruction"],
                })
            dataset.save_episode()
    finally:
        dataset.finalize()
    if image_size is not None:
        preprocessing = {
            "schema_version": 1,
            "operation": "letterbox_resize",
            "height": image_size[0],
            "width": image_size[1],
            "resampling": "bilinear",
            "padding_rgb": [0, 0, 0],
            "source_raw_dataset": str(args.raw_root.expanduser().resolve()),
        }
        (args.output.expanduser().resolve() / "preprocessing.json").write_text(
            json.dumps(preprocessing, indent=2), encoding="utf-8"
        )
    print(f"Exported {len(paths)} episode(s) to {args.output.resolve()}")


if __name__ == "__main__":
    main()
