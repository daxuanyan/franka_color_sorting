from __future__ import annotations

import math
import unittest

import numpy as np

from grasp_geometry import estimate_top_grasp_geometry
from object_catalog import OBJECT_IDS, object_specs


def rectangular_surface(length: float, width: float, height: float, yaw: float) -> np.ndarray:
    x, y = np.meshgrid(np.linspace(-length / 2, length / 2, 20), np.linspace(-width / 2, width / 2, 14))
    points = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, height)))
    rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    points[:, :2] = points[:, :2] @ rotation.T
    return points


class GraspGeometryTests(unittest.TestCase):
    def test_rotated_rectangular_object_recovers_minor_width_and_yaw(self) -> None:
        expected_yaw = math.radians(32)
        geometry = estimate_top_grasp_geometry(rectangular_surface(0.064, 0.028, 0.044, expected_yaw))
        self.assertTrue(geometry.feasible)
        self.assertAlmostEqual(geometry.grasp_width_m, 0.028, delta=0.003)
        self.assertAlmostEqual(geometry.dimensions_xyz[2], 0.044, delta=1e-6)
        self.assertAlmostEqual(geometry.yaw_rad, expected_yaw, delta=0.03)

    def test_symmetric_object_uses_stable_zero_yaw(self) -> None:
        geometry = estimate_top_grasp_geometry(rectangular_surface(0.05, 0.05, 0.05, 0.71))
        self.assertEqual(geometry.yaw_rad, 0.0)

    def test_ungraspable_width_is_rejected(self) -> None:
        geometry = estimate_top_grasp_geometry(rectangular_surface(0.11, 0.09, 0.04, 0.0))
        self.assertFalse(geometry.feasible)
        self.assertIn("grasp width", geometry.rejection_reason or "")

    def test_seen_and_heldout_catalogs_are_disjoint_and_complete(self) -> None:
        seen = object_specs("seen")
        heldout = object_specs("heldout")
        self.assertEqual(tuple(seen), OBJECT_IDS)
        self.assertEqual(tuple(heldout), OBJECT_IDS)
        seen_geometry = {(item.shape, item.dimensions_xyz, item.yaw_rad) for item in seen.values()}
        heldout_geometry = {(item.shape, item.dimensions_xyz, item.yaw_rad) for item in heldout.values()}
        self.assertTrue(seen_geometry.isdisjoint(heldout_geometry))


if __name__ == "__main__":
    unittest.main()
