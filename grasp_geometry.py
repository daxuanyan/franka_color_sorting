"""Category-independent top-grasp geometry estimated from RGB-D points."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class TopGraspGeometry:
    dimensions_xyz: tuple[float, float, float]
    yaw_rad: float
    grasp_width_m: float
    anisotropy: float
    feasible: bool
    rejection_reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_top_grasp_geometry(
    surface_points: np.ndarray,
    *,
    table_height_m: float = 0.0,
    min_width_m: float = 0.010,
    max_width_m: float = 0.075,
    min_height_m: float = 0.018,
    max_height_m: float = 0.100,
) -> TopGraspGeometry:
    """Estimate footprint PCA and a constrained parallel-jaw top grasp.

    The gripper's local Y axis closes across the footprint minor axis. Setting
    tool yaw to the footprint major-axis angle therefore minimizes required
    opening. Symmetric footprints deliberately use yaw=0 because their PCA
    orientation is physically meaningless.
    """
    points = np.asarray(surface_points, dtype=float)
    valid = points.ndim == 2 and points.shape[1:] == (3,)
    if not valid:
        raise ValueError("surface_points must have shape (N, 3)")
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 12:
        raise ValueError("at least 12 finite surface points are required")
    xy = points[:, :2]
    centre = np.median(xy, axis=0)
    covariance = np.cov(xy - centre, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, order]
    projected = (xy - centre) @ axes
    robust_min = np.quantile(projected, 0.01, axis=0)
    robust_max = np.quantile(projected, 0.99, axis=0)
    extents = robust_max - robust_min
    major, minor = float(max(extents)), float(min(extents))
    height = float(np.quantile(points[:, 2], 0.90) - table_height_m)
    anisotropy = major / max(minor, 1e-6)
    major_index = int(np.argmax(extents))
    major_axis = axes[:, major_index]
    yaw = math.atan2(float(major_axis[1]), float(major_axis[0]))
    # A 180-degree parallel-jaw rotation is equivalent.
    yaw = (yaw + math.pi / 2.0) % math.pi - math.pi / 2.0
    if anisotropy < 1.15:
        yaw = 0.0
    reason = None
    if not min_width_m <= minor <= max_width_m:
        reason = f"grasp width {minor:.3f}m outside [{min_width_m:.3f}, {max_width_m:.3f}]m"
    elif not min_height_m <= height <= max_height_m:
        reason = f"object height {height:.3f}m outside [{min_height_m:.3f}, {max_height_m:.3f}]m"
    return TopGraspGeometry(
        dimensions_xyz=(major, minor, height),
        yaw_rad=yaw,
        grasp_width_m=minor,
        anisotropy=anisotropy,
        feasible=reason is None,
        rejection_reason=reason,
    )
