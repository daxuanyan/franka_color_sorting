"""Typed, serialisable contracts between planner, tools, and simulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


COLORS = ("red", "blue", "green", "yellow")
GROUPS = {"warm": ("red", "yellow"), "cool": ("blue", "green"), "primary": ("red", "blue", "yellow"), "all": COLORS}


@dataclass(frozen=True)
class SceneState:
    """The intentionally small, text-only scene representation sent to the LLM."""

    available_objects: tuple[str, ...] = ("four movable colored blocks; infer their colors and layout from the camera image",)
    target_box: str = "target_box"
    arm: str = "Franka Panda; idle; gripper open"
    perception: str = "overhead RGB-D observation is available; local perception supplies pickup estimates while simulator physics verifies outcomes"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    code: str
    detail: str
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
