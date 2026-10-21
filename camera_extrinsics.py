"""Rigid wrist-camera calibration helpers independent of Isaac Sim."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class WristCameraExtrinsics:
    """Fixed transforms expressed in the Franka end-effector frame.

    The housing is mounted 85 mm to the side and 30 mm along the tool approach,
    clear of the finger closing region and the palm mesh. The inboard imager
    lies within the 90 mm-wide D455 housing, 60 mm from the tool axis and
    25 mm in front of the housing centre. Isaac/USD cameras look along local
    -Z, so a 180 degree local-X rotation points them toward the table. A fixed
    15-degree
    inward cant makes the optical axis intersect the tool centreline near the
    22 cm visual-servo working distance without any target-dependent motion.
    """

    body_translation_m: tuple[float, float, float] = (0.085, 0.0, 0.030)
    optical_translation_m: tuple[float, float, float] = (0.060, 0.0, 0.055)
    camera_rotation_wxyz: tuple[float, float, float, float] = (
        0.0,
        math.cos(math.radians(7.5)),
        0.0,
        math.sin(math.radians(7.5)),
    )
    inward_cant_degrees: float = 15.0


DEFAULT_WRIST_CAMERA_EXTRINSICS = WristCameraExtrinsics()


def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=float)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError("quaternion must be a finite wxyz vector")
    norm = float(np.linalg.norm(value))
    if norm < 1e-12:
        raise ValueError("quaternion norm must be non-zero")
    return value / norm


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product for scalar-first (wxyz) quaternions."""
    lw, lx, ly, lz = normalize_quaternion(left)
    rw, rx, ry, rz = normalize_quaternion(right)
    return normalize_quaternion(np.array([
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ]))


def rotate_vector(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quaternion = normalize_quaternion(quaternion)
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("vector must be a finite XYZ vector")
    _, x, y, z = quaternion
    axis = np.array([x, y, z])
    return (
        vector * (2.0 * quaternion[0] ** 2 - 1.0)
        + 2.0 * axis * float(np.dot(axis, vector))
        + 2.0 * quaternion[0] * np.cross(axis, vector)
    )


def compose_pose(
    parent_position: np.ndarray,
    parent_orientation: np.ndarray,
    local_translation: np.ndarray,
    local_orientation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose a fixed local pose with a moving parent pose."""
    parent_position = np.asarray(parent_position, dtype=float)
    if parent_position.shape != (3,) or not np.isfinite(parent_position).all():
        raise ValueError("parent_position must be a finite XYZ vector")
    world_position = parent_position + rotate_vector(parent_orientation, local_translation)
    world_orientation = quaternion_multiply(parent_orientation, local_orientation)
    return world_position, world_orientation
