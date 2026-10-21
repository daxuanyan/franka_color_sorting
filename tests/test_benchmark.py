from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from benchmark import (
    COLORS,
    classify_infrastructure_error,
    control_truth_position,
    generate_scenario,
    generic_perception_outcome,
    localization_errors,
    percentile,
    summarize,
    visual_servo_metrics,
    wilson_interval,
    wrist_localization_errors,
)
from compare_benchmarks import exact_mcnemar_p
from analyze_visual_servo import exact_sign_p, mean_curve


class BenchmarkTests(unittest.TestCase):
    def test_cycle_target_policy_covers_every_heldout_object(self) -> None:
        scenarios = [
            generate_scenario(index, 100 + index, object_set="heldout", target_policy="cycle")
            for index in range(4)
        ]
        self.assertEqual([item["targets"] for item in scenarios], [["red"], ["blue"], ["green"], ["yellow"]])
        self.assertTrue(all(item["object_set"] == "heldout" for item in scenarios))

    def test_infrastructure_errors_are_classified_and_excluded(self) -> None:
        self.assertEqual(
            classify_infrastructure_error("RuntimeError: Could not find assets root folder: remote"),
            "asset_root_unavailable",
        )
        self.assertEqual(
            classify_infrastructure_error("Warp CUDA error: no CUDA-capable device is detected"),
            "gpu_unavailable",
        )
        self.assertIsNone(classify_infrastructure_error("grasp verification failed"))
        summary = summarize([
            {
                "status": "infrastructure_error", "success": False, "duration_s": 60.0,
                "targets": ["red"], "localization": [],
            },
            {
                "status": "completed", "success": True, "duration_s": 10.0,
                "targets": ["red"], "localization": [],
            },
        ])
        self.assertEqual(summary["trials_completed"], 1)
        self.assertEqual(summary["infrastructure_errors"], 1)
        self.assertEqual(summary["success_rate"], 1.0)

    def test_mixed_fault_profile_is_balanced_and_shift_truth_is_recorded(self) -> None:
        scenarios = [
            generate_scenario(index, 100 + index, min_targets=1, max_targets=1, fault_profile="mixed")
            for index in range(3)
        ]
        self.assertEqual([item["faults"]["type"] for item in scenarios], [
            "pose_bias", "dynamic_shift", "sensor_degradation"
        ])
        shifted = scenarios[1]
        color = shifted["faults"]["target_shift"]["color"]
        truth = control_truth_position(shifted, color)
        spawn = shifted["spawn_positions"][color]
        self.assertGreater(math.dist(truth[:2], spawn[:2]), 0.0)

    def test_visual_servo_mean_curve_carries_final_value_forward(self) -> None:
        curve = mean_curve([[20.0, 8.0], [10.0, 6.0, 2.0]], max_updates=2)
        self.assertEqual(curve, [15.0, 7.0, 5.0])
        self.assertAlmostEqual(exact_sign_p(0, 10), 0.001953125)
        self.assertEqual(exact_sign_p(0, 0), 1.0)

    def test_scenario_is_reproducible_and_collision_free(self) -> None:
        first = generate_scenario(3, 12345)
        second = generate_scenario(3, 12345)
        self.assertEqual(first, second)
        positions = first["spawn_positions"]
        self.assertEqual(set(positions), set(COLORS))
        for index, color in enumerate(COLORS):
            for other in COLORS[index + 1 :]:
                self.assertGreaterEqual(math.dist(positions[color][:2], positions[other][:2]), 0.075)

    def test_localization_error_uses_recorded_layout(self) -> None:
        scenario = generate_scenario(0, 7, min_targets=1, max_targets=1)
        poses = {
            color: [position[0] + 0.003, position[1] + 0.004, position[2]]
            for color, position in scenario["spawn_positions"].items()
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest_vision_poses.json"
            path.write_text(json.dumps({"frame": "world", "source": "local_rgbd", "poses": poses}))
            rows = localization_errors(scenario, path)
        self.assertEqual(len(rows), 4)
        for row in rows:
            self.assertAlmostEqual(row["xy_error_mm"], 5.0)

    def test_summary_and_percentile(self) -> None:
        results = [
            {
                "status": "completed", "success": True, "duration_s": 10.0, "targets": ["red"],
                "localization": [{"xy_error_mm": 1.0, "xyz_error_mm": 2.0}],
                "recovery_events": 1, "recovery_attempts": 1, "recovery_successes": 1,
            },
            {
                "status": "completed", "success": False, "duration_s": 20.0, "targets": ["blue"],
                "localization": [{"xy_error_mm": 3.0, "xyz_error_mm": 4.0}],
            },
        ]
        summary = summarize(results)
        self.assertEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["motion_success_rate"], 0.5)
        self.assertEqual(summary["mean_xy_error_mm"], 2.0)
        self.assertEqual(summary["mean_duration_s"], 15.0)
        self.assertEqual(summary["mean_success_duration_s"], 10.0)
        self.assertEqual(summary["detection_rate"], 0.25)
        self.assertEqual(summary["recovery_events"], 1)
        self.assertEqual(summary["recovery_attempts"], 1)
        self.assertEqual(summary["recovery_successes"], 1)
        self.assertEqual(summary["by_target_count"]["1"]["success_rate"], 0.5)
        self.assertAlmostEqual(percentile([1.0, 3.0], 95), 2.9)
        self.assertAlmostEqual(wilson_interval(1, 2)[0], 0.09452865480086614)
        self.assertIsNone(wilson_interval(0, 0))
        self.assertAlmostEqual(exact_mcnemar_p(8, 1), 0.0390625)
        self.assertEqual(exact_mcnemar_p(0, 0), 1.0)

    def test_generic_perception_distinguishes_safe_rejection_from_miss(self) -> None:
        payload = {
            "instances": [
                {
                    "instance_id": "instance_00",
                    "grasp_geometry": {
                        "feasible": False,
                        "rejection_reason": "top surface relief exceeds limit",
                    },
                },
                {"instance_id": "instance_01", "grasp_geometry": {"feasible": True}},
            ],
            "assignments": {"instance_01": {"label": "blue", "confidence": 0.9}},
            "rejected_assignments": {"instance_00": {"label": "yellow", "confidence": 0.9}},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest_overhead_instances"
            path.mkdir()
            (path / "generic_perception.json").write_text(json.dumps(payload))
            outcome = generic_perception_outcome(Path(directory), ["yellow"])
        self.assertEqual(outcome["semantic_labels"], ["blue", "yellow"])
        self.assertEqual(outcome["feasible_labels"], ["blue"])
        self.assertTrue(outcome["safe_rejection"])
        self.assertIn("yellow", outcome["geometry_rejections"])

    def test_summary_excludes_safe_rejection_from_motion_success_rate(self) -> None:
        results = [
            {
                "status": "completed", "success": True, "duration_s": 10.0,
                "targets": ["red"], "localization": [], "semantic_labels": list(COLORS),
            },
            {
                "status": "completed", "success": False, "duration_s": 2.0,
                "targets": ["yellow"], "localization": [], "semantic_labels": list(COLORS),
                "safe_rejection": True,
            },
        ]
        summary = summarize(results)
        self.assertEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["safe_rejection_rate"], 0.5)
        self.assertEqual(summary["motion_success_rate"], 1.0)
        self.assertEqual(summary["semantic_detection_rate"], 1.0)

    def test_wrist_localization_reads_per_trial_artifact(self) -> None:
        scenario = generate_scenario(0, 19, min_targets=1, max_targets=1)
        color = scenario["targets"][0]
        truth = scenario["spawn_positions"][color]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"wrist_{color}_pregrasp_poses.json"
            path.write_text(json.dumps({"poses": {color: [truth[0] + 0.006, truth[1] + 0.008, truth[2]]}}))
            rows = wrist_localization_errors(scenario, Path(directory))
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["xy_error_mm"], 10.0)

    def test_wrist_localization_reads_visual_servo_sequence(self) -> None:
        scenario = generate_scenario(0, 23, min_targets=1, max_targets=1)
        color = scenario["targets"][0]
        truth = scenario["spawn_positions"][color]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for sample, offset in enumerate((0.010, 0.002), start=1):
                path = root / f"wrist_{color}_servo_{sample:03d}_poses.json"
                path.write_text(json.dumps({"poses": {color: [truth[0] + offset, truth[1], truth[2]]}}))
            # The rolling compatibility file must not be counted a third time.
            (root / f"wrist_{color}_pregrasp_poses.json").write_text(
                json.dumps({"poses": {color: truth}})
            )
            rows = wrist_localization_errors(scenario, root)
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["sample"] for row in rows], [1, 2])
        self.assertAlmostEqual(rows[0]["xy_error_mm"], 10.0)
        self.assertAlmostEqual(rows[1]["xy_error_mm"], 2.0)

    def test_visual_servo_metrics_measure_command_convergence(self) -> None:
        scenario = generate_scenario(0, 29, min_targets=1, max_targets=1)
        color = scenario["targets"][0]
        truth = scenario["spawn_positions"][color]
        records = [
            {
                "color": color, "accepted": True,
                "prior_command": [truth[0] + 0.020, truth[1], truth[2]],
                "command": [truth[0] + 0.010, truth[1], truth[2]],
            },
            {
                "color": color, "accepted": True,
                "prior_command": [truth[0] + 0.010, truth[1], truth[2]],
                "command": [truth[0] + 0.002, truth[1], truth[2]],
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "visual_servo_trace.jsonl"
            path.write_text("\n".join(json.dumps(record) for record in records))
            rows = visual_servo_metrics(scenario, Path(directory))
        self.assertEqual(rows[0]["updates"], 2)
        self.assertAlmostEqual(rows[0]["initial_xy_error_mm"], 20.0)
        self.assertAlmostEqual(rows[0]["final_xy_error_mm"], 2.0)
        self.assertAlmostEqual(rows[0]["xy_improvement_mm"], 18.0)


if __name__ == "__main__":
    unittest.main()
