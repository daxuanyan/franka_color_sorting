"""Local allow-listed tools. The LLM can request, never bypass, these checks."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .contracts import COLORS, GROUPS, SceneState, ToolResult


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "name": "get_scene_state",
        "description": "Read the compact current scene state before planning a robot action.",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        "strict": True,
    },
    {
        "type": "function",
        "name": "sort_to_target_box",
        "description": "Pick the requested known colored blocks and stack them into target_box. This is a blocking, simulator-verified physical action; call it once, then inspect its result before retrying.",
        "parameters": {
            "type": "object",
            "properties": {
                "targets": {"type": "array", "items": {"type": "string", "enum": [*COLORS, *GROUPS]}, "minItems": 1},
            },
            "required": ["targets"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


class FrankaTools:
    """Tool dispatcher and physical-execution boundary."""

    def __init__(self, project_root: Path, *, dry_run: bool = False, timeout_s: int = 180) -> None:
        self.project_root = project_root
        self.dry_run = dry_run
        self.timeout_s = timeout_s
        self.scene = SceneState()
        self._attempts: dict[str, int] = {}

    def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if name == "get_scene_state":
            if arguments:
                return ToolResult(False, "INVALID_ARGUMENT", "get_scene_state takes no arguments", {})
            return ToolResult(True, "OK", "Scene state read", self.scene.to_dict())
        if name == "sort_to_target_box":
            return self._sort(arguments)
        return ToolResult(False, "UNKNOWN_TOOL", f"Tool '{name}' is not permitted", {})

    def _sort(self, arguments: dict[str, Any]) -> ToolResult:
        targets = arguments.get("targets")
        if not isinstance(targets, list) or not targets or not all(isinstance(item, str) for item in targets):
            return ToolResult(False, "INVALID_ARGUMENT", "targets must be a non-empty string list", {})
        normalized = [item.lower() for item in targets]
        invalid = [item for item in normalized if item not in (*COLORS, *GROUPS)]
        if invalid:
            return ToolResult(False, "INVALID_TARGET", f"Unknown target(s): {invalid}", {"allowed": [*COLORS, *GROUPS]})
        key = ",".join(normalized)
        if self._attempts.get(key, 0) >= 2:
            return ToolResult(False, "RETRY_LIMIT", "Physical retry limit reached; reset or ask a human to inspect the scene", {})
        self._attempts[key] = self._attempts.get(key, 0) + 1
        if self.dry_run:
            return ToolResult(True, "DRY_RUN", "Validated simulator task; simulator was not started", {"targets": normalized})
        command = [str(self.project_root.parent / "isaacsim" / "python.sh"), str(self.project_root / "franka_color_sorting.py"), "--target", *normalized]
        try:
            completed = subprocess.run(command, cwd=self.project_root, text=True, capture_output=True, timeout=self.timeout_s, check=False)
        except subprocess.TimeoutExpired:
            return ToolResult(False, "MOTION_TIMEOUT", f"Simulation exceeded {self.timeout_s}s", {})
        log = (completed.stdout + "\n" + completed.stderr)[-4000:]
        if completed.returncode == 0 and "Success: requested blocks placed in target box." in completed.stdout:
            return ToolResult(True, "OK", "Simulator verified all requested blocks in target_box", {"targets": normalized, "log_tail": log})
        return ToolResult(False, "SIMULATION_FAILED", "Simulator did not verify the requested placement", {"targets": normalized, "log_tail": log})

    def close(self) -> None:
        """The batch executor owns no background process."""
