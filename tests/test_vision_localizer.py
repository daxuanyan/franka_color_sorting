from __future__ import annotations

import unittest

import numpy as np

from vision_localizer import (
    LocalizationQuality,
    RGBDInstance,
    associate_instance_labels,
    assess_refinement,
    bounded_servo_update,
    extract_rgbd_instances,
    localize_coloured_blocks_with_quality,
    localize_nearest_instance_with_quality,
    top_surface_center,
)


class FakeCamera:
    def __init__(self, *, full_frame: bool = False) -> None:
        self.rgb = np.zeros((40, 40, 4), dtype=np.uint8)
        if full_frame:
            self.rgb[:, :, :3] = (20, 180, 20)
        else:
            self.rgb[15:25, 15:25, :3] = (20, 180, 20)
        self.depth = np.ones((40, 40), dtype=np.float32)

    def get_current_frame(self):
        return {"rgba": self.rgb, "distance_to_image_plane": self.depth}

    def get_rgba(self):
        return self.rgb

    def get_depth(self):
        return self.depth

    def get_world_points_from_image_coords(self, points, depths):
        return np.column_stack((points[:, 0] / 100, points[:, 1] / 100, depths))


class GenericFakeCamera:
    """Synthetic calibrated camera with two non-colour-dependent height blobs."""

    def __init__(self) -> None:
        self.rgb = np.full((80, 80, 4), 30, dtype=np.uint8)
        self.rgb[..., 3] = 255
        self.rgb[12:30, 12:30, :3] = (180, 70, 120)
        self.rgb[48:68, 48:68, :3] = (80, 180, 160)
        self.depth = np.ones((80, 80), dtype=np.float32)

    def get_current_frame(self):
        return {"rgba": self.rgb, "distance_to_image_plane": self.depth}

    def get_rgba(self):
        return self.rgb

    def get_depth(self):
        return self.depth

    def get_world_points_from_image_coords(self, points, depths):
        del depths
        columns, rows = points[:, 0], points[:, 1]
        first = (columns >= 12) & (columns < 30) & (rows >= 12) & (rows < 30)
        second = (columns >= 48) & (columns < 68) & (rows >= 48) & (rows < 68)
        height = np.where(first | second, 0.05, 0.0)
        return np.column_stack((0.25 + columns * 0.005, -0.62 + rows * 0.006, height))


class VisionQualityTests(unittest.TestCase):
    def test_geometry_instances_do_not_depend_on_colour_masks(self) -> None:
        instances = extract_rgbd_instances(GenericFakeCamera())
        self.assertEqual([item.instance_id for item in instances], ["instance_00", "instance_01"])
        self.assertTrue(all(item.quality.confidence > 0.9 for item in instances))
        np.testing.assert_allclose(instances[0].pose[2], 0.025)
        np.testing.assert_allclose(instances[1].pose[2], 0.025)

    def test_semantic_association_is_one_to_one_and_preserves_geometry(self) -> None:
        instances = extract_rgbd_instances(GenericFakeCamera())
        assignments = {
            "instance_00": {"label": "plum_object", "confidence": 0.8},
            "instance_01": {"label": "teal_object", "confidence": 0.9},
        }
        poses, qualities = associate_instance_labels(instances, assignments)
        self.assertEqual(set(poses), {"plum_object", "teal_object"})
        self.assertAlmostEqual(qualities["plum_object"].confidence, 0.8)
        with self.assertRaises(ValueError):
            associate_instance_labels(
                instances,
                {
                    "instance_00": {"label": "duplicate", "confidence": 0.8},
                    "instance_01": {"label": "duplicate", "confidence": 0.9},
                },
            )

    def test_generic_wrist_tracking_uses_nearest_instance(self) -> None:
        instances = extract_rgbd_instances(GenericFakeCamera())
        expected = np.asarray(instances[1].pose)
        pose, quality = localize_nearest_instance_with_quality(GenericFakeCamera(), expected)
        np.testing.assert_allclose(pose, expected, atol=0.004)
        self.assertIsNotNone(quality)
        pose, quality = localize_nearest_instance_with_quality(
            GenericFakeCamera(), np.array([0.75, -0.60, 0.025]), max_distance_m=0.01
        )
        self.assertIsNone(pose)
        self.assertIsNone(quality)

    def test_bounded_servo_update_filters_and_limits_motion(self) -> None:
        current = np.array([0.45, -0.40, 0.025])
        measured = current + np.array([0.030, 0.0, 0.0])
        updated, innovation, applied = bounded_servo_update(
            current, measured, alpha=0.7, max_step_m=0.008
        )
        self.assertAlmostEqual(innovation, 0.030)
        self.assertAlmostEqual(applied, 0.008)
        np.testing.assert_allclose(updated, current + np.array([0.008, 0.0, 0.0]))

        close_measurement = updated + np.array([0.003, 0.004, 0.0])
        filtered, innovation, applied = bounded_servo_update(
            updated, close_measurement, alpha=0.5, max_step_m=0.008
        )
        self.assertAlmostEqual(innovation, 0.005)
        self.assertAlmostEqual(applied, 0.0025)
        np.testing.assert_allclose(filtered, updated + np.array([0.0015, 0.002, 0.0]))

    def test_bounded_servo_update_rejects_invalid_configuration(self) -> None:
        with self.assertRaises(ValueError):
            bounded_servo_update(np.zeros(3), np.ones(3), alpha=0.0)
        with self.assertRaises(ValueError):
            bounded_servo_update(np.zeros(3), np.ones(3), max_step_m=0.0)

    def test_world_height_filter_avoids_oblique_near_edge_bias(self) -> None:
        grid = (-0.02, -0.01, 0.0, 0.01, 0.02)
        top = np.array([[x, y, 0.05] for x in grid for y in grid])
        sides = np.array([[0.02, y, z] for y in (-0.02, 0.0, 0.02) for z in (0.01, 0.025, 0.04)])
        points = np.vstack((top, sides))
        # Make one side artificially nearer than the top to model an oblique
        # camera; selecting by depth would shift the estimate toward x=0.02.
        depths = np.concatenate((np.full(len(top), 0.20), np.full(len(sides), 0.15)))
        centre, selected_depths, selected_heights = top_surface_center(points, depths)
        np.testing.assert_allclose(centre, [0.0, 0.0, 0.05], atol=1e-9)
        self.assertEqual(len(selected_depths), len(top))
        np.testing.assert_allclose(selected_heights, 0.05)

    def test_compact_planar_mask_has_high_confidence(self) -> None:
        poses, qualities = localize_coloured_blocks_with_quality(FakeCamera())
        self.assertIn("green", poses)
        self.assertAlmostEqual(qualities["green"].mask_fraction, 0.0625)
        self.assertEqual(qualities["green"].confidence, 1.0)

    def test_full_frame_mask_is_rejected_as_low_confidence(self) -> None:
        _, qualities = localize_coloured_blocks_with_quality(FakeCamera(full_frame=True))
        self.assertEqual(qualities["green"].mask_fraction, 1.0)
        self.assertEqual(qualities["green"].confidence, 0.0)

    def test_refinement_gate_checks_confidence_and_displacement(self) -> None:
        coarse = np.array([0.45, -0.40, 0.025])
        quality = LocalizationQuality(100, 100, 0.05, 0.0001, 0.7)
        accepted, _ = assess_refinement(coarse, coarse + np.array([0.003, 0.004, 0.001]), quality)
        self.assertTrue(accepted)
        accepted, reason = assess_refinement(coarse, coarse + np.array([0.020, 0.0, 0.0]), quality)
        self.assertFalse(accepted)
        self.assertIn("XY correction", reason)
        high_quality = LocalizationQuality(100, 100, 0.05, 0.0001, 0.9)
        accepted, _ = assess_refinement(coarse, coarse + np.array([0.020, 0.0, 0.0]), high_quality)
        self.assertTrue(accepted)
        accepted, reason = assess_refinement(coarse, coarse + np.array([0.035, 0.0, 0.0]), high_quality)
        self.assertFalse(accepted)
        self.assertIn("30.0mm", reason)
        poor = LocalizationQuality(1000, 1000, 1.0, 0.0, 0.0)
        accepted, reason = assess_refinement(coarse, coarse, poor)
        self.assertFalse(accepted)
        self.assertIn("low visual confidence", reason)


if __name__ == "__main__":
    unittest.main()
