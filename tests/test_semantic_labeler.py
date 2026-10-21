from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from semantic_labeler import label_manifest, parse_label_response


class SemanticLabelerTests(unittest.TestCase):
    def test_response_parser_accepts_id_or_number_and_rejects_ambiguity(self) -> None:
        labels = {"mug": "a ceramic mug", "apple": "an apple"}
        self.assertEqual(parse_label_response("mug", labels), "mug")
        self.assertEqual(parse_label_response("Option 2", labels), "apple")
        with self.assertRaises(ValueError):
            parse_label_response("mug or apple", labels)

    def test_prototype_manifest_keeps_instance_and_semantic_ids_separate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            red_path = root / "arbitrary_a.png"
            blue_path = root / "arbitrary_b.png"
            Image.fromarray(np.full((16, 16, 3), (220, 50, 50), dtype=np.uint8)).save(red_path)
            Image.fromarray(np.full((16, 16, 3), (50, 50, 220), dtype=np.uint8)).save(blue_path)
            result = label_manifest(
                {
                    "labels": {"red": "the warm object", "blue": "the cool object"},
                    "candidates": [
                        {"instance_id": "instance_17", "crop_path": str(red_path)},
                        {"instance_id": "instance_04", "crop_path": str(blue_path)},
                    ],
                },
                "prototype",
                "unused",
            )
        self.assertEqual(result["assignments"]["instance_17"]["label"], "red")
        self.assertEqual(result["assignments"]["instance_04"]["label"], "blue")
        self.assertEqual(result["unmatched_labels"], [])


if __name__ == "__main__":
    unittest.main()
