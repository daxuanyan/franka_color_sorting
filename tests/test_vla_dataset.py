from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from export_lerobot import episode_paths, feature_schema, load_rgb_image
from create_vla_splits import make_splits
from collect_vla_until import next_attempt_index
from vla_dataset import (
    RawVLAEpisodeRecorder, load_raw_episode, next_episode_index,
    validate_raw_episode,
)


class RawVLAEpisodeTests(unittest.TestCase):
    def test_episode_splits_are_deterministic_disjoint_and_exhaustive(self) -> None:
        records = [
            {"raw_episode_index": index, "lerobot_episode_index": index}
            for index in range(23)
        ]
        first = make_splits(records, seed=17)
        second = make_splits(records, seed=17)
        self.assertEqual(first, second)
        self.assertEqual([len(first[name]) for name in ("train", "validation", "test")], [19, 2, 2])
        split_ids = [
            {item["raw_episode_index"] for item in first[name]}
            for name in ("train", "validation", "test")
        ]
        self.assertFalse(split_ids[0] & split_ids[1])
        self.assertFalse(split_ids[0] & split_ids[2])
        self.assertFalse(split_ids[1] & split_ids[2])
        self.assertEqual(set.union(*split_ids), set(range(23)))

    def test_topup_attempt_index_never_reuses_an_output_directory(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "topup_0000").mkdir()
            (root / "topup_0001").mkdir()
            self.assertEqual(next_attempt_index(root, "topup"), 2)

    def test_records_rate_limited_multimodal_episode(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = RawVLAEpisodeRecorder(
                root, fps=10, instruction="pick the red object", targets=("red",),
                object_set="heldout", joint_names=["j0", "j1"],
            )
            rgb_a = np.full((8, 10, 3), 20, dtype=np.uint8)
            rgb_b = np.full((6, 7, 4), 40, dtype=np.uint8)
            depth_a = np.full((8, 10), 1.25)
            depth_b = np.full((6, 7), 0.25)
            kwargs = dict(
                overhead_rgb=rgb_a, overhead_depth_m=depth_a,
                wrist_rgb=rgb_b, wrist_depth_m=depth_b,
                joint_position=np.array([0.1, 0.2]), joint_velocity=np.array([0.0, 0.1]),
                end_effector_position=np.array([0.4, -0.2, 0.3]),
                end_effector_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                gripper_width=0.04, joint_position_target=np.array([0.2, 0.3]),
                eef_position_target=np.array([0.45, -0.2, 0.2]), gripper_open_target=True,
                phase="PRE_GRASP", object_id="red",
                object_positions={"red": np.array([0.45, -0.3, 0.02])}, done=False,
            )
            self.assertTrue(recorder.add_frame(time_s=1.0, **kwargs))
            self.assertFalse(recorder.add_frame(time_s=1.05, **kwargs))
            self.assertTrue(recorder.add_frame(time_s=1.10, **kwargs))
            recorder.finalize(success=True)

            metadata, frames = load_raw_episode(recorder.episode_dir)
            self.assertEqual(metadata["num_frames"], 2)
            self.assertEqual(metadata["status"], "complete")
            self.assertEqual(len(frames[0]["observation.state"]), 12)
            self.assertTrue((recorder.episode_dir / frames[0]["wrist_depth"]).exists())
            self.assertEqual(next_episode_index(root), 1)
            summary = validate_raw_episode(recorder.episode_dir)
            self.assertEqual(summary["state_dim"], 12)
            self.assertEqual(summary["action_dim"], 2)

            frames[1]["timestamp"] = 99.0
            (recorder.episode_dir / "frames.jsonl").write_text(
                "".join(json.dumps(frame) + "\n" for frame in frames), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "invalid timestamp"):
                validate_raw_episode(recorder.episode_dir)

    def test_records_dagger_teacher_student_and_executed_actions(self) -> None:
        with TemporaryDirectory() as directory:
            recorder = RawVLAEpisodeRecorder(
                Path(directory), fps=5, instruction="Place the red object",
                targets=("red",), object_set="heldout", joint_names=["j0", "j1"],
                dataset_kind="dagger_correction",
                metadata_extra={"dagger_round": 1, "beta": 0.8},
            )
            recorder.add_frame(
                time_s=0.0,
                overhead_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
                overhead_depth_m=np.ones((4, 4)),
                wrist_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
                wrist_depth_m=np.ones((4, 4)),
                joint_position=np.zeros(2), joint_velocity=np.zeros(2),
                end_effector_position=np.zeros(3),
                end_effector_quaternion_wxyz=np.array([1, 0, 0, 0]),
                gripper_width=0.08,
                joint_position_target=np.array([0.2, 0.3]),
                student_action=np.array([0.0, 0.1]),
                executed_action=np.array([0.16, 0.26]),
                beta=0.8, expert_intervened=True,
                intervention_reason="scheduled_beta",
                label_valid=True,
                eef_position_target=None, gripper_open_target=True,
                phase="PRE_GRASP", object_id="red",
                object_positions={"red": np.zeros(3)}, done=False,
            )
            recorder.finalize(success=False, failure_reason="collection timeout")
            metadata, frames = load_raw_episode(recorder.episode_dir)
            self.assertEqual(metadata["dataset_kind"], "dagger_correction")
            self.assertEqual(frames[0]["action"], [0.2, 0.3])
            self.assertEqual(frames[0]["action.student"], [0.0, 0.1])
            self.assertEqual(frames[0]["action.executed"], [0.16, 0.26])
            self.assertTrue(frames[0]["dagger.expert_intervened"])
            self.assertTrue(frames[0]["dagger.label_valid"])

    def test_failed_episode_selection_is_explicit(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = RawVLAEpisodeRecorder(
                root, fps=10, instruction="task", targets=("red",),
                object_set="cubes", joint_names=["j0"],
            )
            recorder.finalize(success=False, failure_reason="test")
            self.assertEqual(episode_paths(root, include_failures=False), [])
            self.assertEqual(episode_paths(root, include_failures=True), [recorder.episode_dir])

    def test_feature_schema_uses_recorded_shapes(self) -> None:
        with TemporaryDirectory() as directory:
            episode = Path(directory)
            (episode / "a.png").parent.mkdir(parents=True, exist_ok=True)
            from PIL import Image
            Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8)).save(episode / "a.png")
            Image.fromarray(np.zeros((6, 7, 3), dtype=np.uint8)).save(episode / "b.png")
            schema = feature_schema(
                {"joint_names": ["j0", "j1"]},
                {"overhead_rgb": "a.png", "wrist_rgb": "b.png", "observation.state": [0] * 12, "action": [0, 0]},
                episode,
            )
            self.assertEqual(schema["observation.images.overhead"]["shape"], (4, 5, 3))
            self.assertEqual(schema["action"]["names"], ["j0", "j1"])

            resized = feature_schema(
                {"joint_names": ["j0", "j1"]},
                {"overhead_rgb": "a.png", "wrist_rgb": "b.png", "observation.state": [0] * 12, "action": [0, 0]},
                episode, (8, 8),
            )
            self.assertEqual(resized["observation.images.overhead"]["shape"], (8, 8, 3))
            self.assertEqual(resized["observation.images.wrist"]["shape"], (8, 8, 3))
            image = load_rgb_image(episode / "a.png", (8, 8))
            self.assertEqual(image.shape, (8, 8, 3))


if __name__ == "__main__":
    unittest.main()
