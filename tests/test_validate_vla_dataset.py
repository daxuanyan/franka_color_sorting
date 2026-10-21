import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from validate_vla_dataset import validate_raw_root


class RawRootValidationTest(unittest.TestCase):
    def test_skip_empty_is_explicit_and_keeps_valid_episodes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            empty = root / "episodes" / "episode_000000"
            valid = root / "episodes" / "episode_000001"
            empty.mkdir(parents=True)
            valid.mkdir()

            def fake_validate(path):
                if path == empty:
                    raise ValueError(f"episode contains no frames: {path}")
                return {"episode_index": 1, "num_frames": 3}

            with patch("validate_vla_dataset.validate_raw_episode", side_effect=fake_validate):
                with self.assertRaisesRegex(ValueError, "contains no frames"):
                    validate_raw_root(root)
                self.assertEqual(
                    validate_raw_root(root, skip_empty=True),
                    [{"episode_index": 1, "num_frames": 3}],
                )


if __name__ == "__main__":
    unittest.main()
