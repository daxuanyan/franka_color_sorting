from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analyze_unseen_objects import build_report


class UnseenObjectAnalysisTests(unittest.TestCase):
    def test_report_counts_motion_and_safe_rejection_separately(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = []
            for index, (target, success, feasible) in enumerate(
                (("red", True, True), ("yellow", False, False))
            ):
                trial = root / f"trial_{index:04d}"
                artifact = trial / "latest_overhead_instances"
                artifact.mkdir(parents=True)
                instance = f"instance_{index:02d}"
                assignment = {instance: {"label": target, "confidence": 0.9}}
                payload = {
                    "instances": [{
                        "instance_id": instance,
                        "grasp_geometry": {
                            "feasible": feasible,
                            "rejection_reason": None if feasible else "unsupported taper",
                        },
                    }],
                    "assignments": assignment if feasible else {},
                    "rejected_assignments": {} if feasible else assignment,
                }
                (artifact / "generic_perception.json").write_text(json.dumps(payload))
                results.append({
                    "trial_index": index,
                    "seed": index,
                    "targets": [target],
                    "success": success,
                    "artifacts": str(trial),
                })
            (root / "results.json").write_text(json.dumps(results))
            report = build_report(root)
        self.assertEqual(report["target_semantic_detection_rate"], 1.0)
        self.assertEqual(report["safe_rejections"], 1)
        self.assertEqual(report["motion_success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
