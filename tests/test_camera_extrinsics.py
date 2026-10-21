from __future__ import annotations

import math
import unittest

import numpy as np

from camera_extrinsics import (
    DEFAULT_WRIST_CAMERA_EXTRINSICS,
    compose_pose,
    rotate_vector,
)


class CameraExtrinsicsTests(unittest.TestCase):
    def test_camera_is_side_mounted_and_canted_toward_tool_axis(self) -> None:
        calibration = DEFAULT_WRIST_CAMERA_EXTRINSICS
        # The grasp controller's nominal downward tool orientation: 180 deg Y.
        wrist_q = np.array([0.0, 0.0, 1.0, 0.0])
        eye, camera_q = compose_pose(
            np.array([0.5, -0.3, 0.25]),
            wrist_q,
            np.asarray(calibration.optical_translation_m),
            np.asarray(calibration.camera_rotation_wxyz),
        )
        np.testing.assert_allclose(eye, [0.44, -0.3, 0.195], atol=1e-9)
        optical_forward_world = rotate_vector(camera_q, np.array([0.0, 0.0, -1.0]))
        optical_forward_tool = rotate_vector(
            np.array([wrist_q[0], -wrist_q[1], -wrist_q[2], -wrist_q[3]]),
            optical_forward_world,
        )
        angle = math.radians(calibration.inward_cant_degrees)
        np.testing.assert_allclose(
            optical_forward_tool,
            [-math.sin(angle), 0.0, math.cos(angle)],
            atol=1e-9,
        )
        # The ray crosses the tool axis close to the configured 22 cm
        # pre-grasp observation distance.
        intersection_distance = calibration.optical_translation_m[0] / math.tan(angle)
        self.assertAlmostEqual(intersection_distance, 0.224, places=3)

    def test_mount_offset_rotates_rigidly_with_wrist(self) -> None:
        calibration = DEFAULT_WRIST_CAMERA_EXTRINSICS
        half = math.pi / 4.0
        wrist_q = np.array([math.cos(half), 0.0, 0.0, math.sin(half)])
        eye, _ = compose_pose(
            np.zeros(3), wrist_q,
            np.asarray(calibration.optical_translation_m),
            np.asarray(calibration.camera_rotation_wxyz),
        )
        np.testing.assert_allclose(eye, [0.0, 0.060, 0.055], atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.norm(eye)), math.hypot(0.060, 0.055))


if __name__ == "__main__":
    unittest.main()
