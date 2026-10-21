import unittest
import json
import tempfile
from pathlib import Path

from collect_dagger import parse_target_counts, trainable_episodes


class TargetScheduleTest(unittest.TestCase):
    def test_exact_failure_directed_quotas_are_reproducible(self):
        value = "yellow=15,green=12,blue=8,red=5"
        first = parse_target_counts(value, 40, 123)
        second = parse_target_counts(value, 40, 123)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 40)
        self.assertEqual({color: first.count(color) for color in set(first)}, {
            "yellow": 15, "green": 12, "blue": 8, "red": 5,
        })

    def test_rejects_wrong_total(self):
        with self.assertRaisesRegex(ValueError, "sum"):
            parse_target_counts("yellow=1,green=1,blue=1,red=1", 5, 123)

    def test_default_cycles_targets(self):
        self.assertEqual(
            parse_target_counts(None, 6, 123),
            ["red", "blue", "green", "yellow", "red", "blue"],
        )

    def test_zero_label_rejection_is_not_trainable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, valid in ((0, False), (1, True)):
                episode = root / "episodes" / f"episode_{index:06d}"
                episode.mkdir(parents=True)
                (episode / "episode.json").write_text(json.dumps({
                    "episode_index": index, "status": "failed", "success": False,
                }))
                frame = ({
                    "phase": "GRASP", "action.student": [0], "action.executed": [0],
                    "dagger.label_valid": True,
                } if valid else {"phase": None})
                (episode / "frames.jsonl").write_text(json.dumps(frame) + "\n")
            episodes = trainable_episodes(root)
            self.assertEqual(set(episodes), {1})
            self.assertEqual(episodes[1]["valid_label_frames"], 1)


if __name__ == "__main__":
    unittest.main()
