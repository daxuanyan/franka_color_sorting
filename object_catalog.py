"""Reproducible seen/held-out primitive objects for category-agnostic grasp tests."""

from __future__ import annotations

from dataclasses import dataclass
import math


OBJECT_IDS = ("red", "blue", "green", "yellow")


@dataclass(frozen=True)
class ObjectSpec:
    object_id: str
    shape: str
    dimensions_xyz: tuple[float, float, float]
    yaw_rad: float
    color_rgb: tuple[float, float, float]
    description: str
    split: str


_COLORS = {
    "red": (0.8, 0.1, 0.1),
    "blue": (0.1, 0.1, 0.8),
    "green": (0.1, 0.7, 0.1),
    "yellow": (0.9, 0.8, 0.1),
}


def _spec(
    object_id: str,
    shape: str,
    dimensions_xyz: tuple[float, float, float],
    yaw_degrees: float,
    description: str,
    split: str,
) -> ObjectSpec:
    return ObjectSpec(
        object_id=object_id,
        shape=shape,
        dimensions_xyz=dimensions_xyz,
        yaw_rad=math.radians(yaw_degrees),
        color_rgb=_COLORS[object_id],
        description=description,
        split=split,
    )


OBJECT_SETS: dict[str, tuple[ObjectSpec, ...]] = {
    "cubes": tuple(
        _spec(object_id, "cuboid", (0.05, 0.05, 0.05), 0.0, f"a {object_id} cube", "baseline")
        for object_id in OBJECT_IDS
    ),
    "seen": (
        _spec("red", "cuboid", (0.045, 0.045, 0.045), 0.0, "a small red cube", "seen"),
        _spec("blue", "cylinder", (0.044, 0.044, 0.050), 0.0, "a blue cylinder", "seen"),
        _spec("green", "cuboid", (0.058, 0.034, 0.042), 20.0, "a green rectangular block", "seen"),
        _spec("yellow", "cylinder", (0.050, 0.050, 0.040), 0.0, "a short yellow cylinder", "seen"),
    ),
    # Shape/size/pose combinations below are never used by the seen split.
    # They intentionally include curved and tapered geometry.
    "heldout": (
        _spec("red", "cuboid", (0.068, 0.026, 0.040), 37.0, "a thin red rectangular object", "heldout"),
        _spec("blue", "sphere", (0.046, 0.046, 0.046), 0.0, "a blue sphere", "heldout"),
        _spec("green", "cylinder", (0.036, 0.036, 0.068), 0.0, "a tall narrow green cylinder", "heldout"),
        _spec("yellow", "cone", (0.048, 0.048, 0.050), 0.0, "a yellow cone", "heldout"),
    ),
}


def object_specs(name: str) -> dict[str, ObjectSpec]:
    if name not in OBJECT_SETS:
        raise ValueError(f"unknown object set {name!r}; choose from {', '.join(OBJECT_SETS)}")
    return {item.object_id: item for item in OBJECT_SETS[name]}


def semantic_vocabulary_for(name: str) -> dict[str, str]:
    return {object_id: spec.description for object_id, spec in object_specs(name).items()}

