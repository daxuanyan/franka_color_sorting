"""Local allow-listed tools. The LLM can request, never bypass, these checks."""

from __future__ import annotations

import subprocess
import json
import socket
import time
from pathlib import Path
from typing import Any

from .contracts import COLORS, GROUPS, SceneState, ToolResult
from .task_plan import LOW_CONFIDENCE_POLICIES, TaskPlan
from .hybrid_vla import (
    HYBRID_GRASP_STRATEGIES,
    HYBRID_SKILLS,
    HYBRID_START_PHASES,
    HybridVLAPlan,
)


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
        "name": "execute_sort_plan",
        "description": "Execute an ordered, locally validated block-sorting plan. Call only after get_scene_state. Never invent objects or use this tool for an ambiguous request.",
        "parameters": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "description": "Ordered actions. Use contiguous stack_level values 0, 1, ... in execution order; include each object at most once.",
                    "minItems": 1,
                    "maxItems": 4,
                    "items": {
                        "type": "object",
                        "properties": {
                            "object_id": {"type": "string", "enum": [*COLORS], "description": "An object ID present in the latest scene state."},
                            "destination": {"type": "string", "enum": ["target_box"]},
                            "stack_level": {"type": "integer", "minimum": 0, "maximum": 3},
                        },
                        "required": ["object_id", "destination", "stack_level"],
                        "additionalProperties": False,
                    },
                },
                "on_low_confidence": {"type": "string", "enum": [*LOW_CONFIDENCE_POLICIES], "description": "Use reobserve unless the user explicitly requires stopping on uncertainty."},
                "max_retries": {"type": "integer", "minimum": 0, "maximum": 1, "description": "At most one retry; use zero when the user requests no retry."},
            },
            "required": ["steps", "on_low_confidence", "max_retries"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

HYBRID_VLA_TOOL_DEFINITIONS = [
    TOOL_DEFINITIONS[0],
    {
        "type": "function",
        "name": "execute_hybrid_vla_plan",
        "description": (
            "Execute a visual-language-selected sequence of bounded robot skills. "
            "The VLA selects semantic targets, order, and high-level phase; local RGB-D, "
            "shape-aware grasping, visual servo, and physics verification execute it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array", "minItems": 1, "maxItems": 4,
                    "items": {
                        "type": "object",
                        "properties": {
                            "object_id": {"type": "string", "enum": [*COLORS]},
                            "destination": {"type": "string", "enum": ["target_box"]},
                            "stack_level": {"type": "integer", "minimum": 0, "maximum": 3},
                            "skill": {"type": "string", "enum": [*HYBRID_SKILLS]},
                            "start_phase": {"type": "string", "enum": [*HYBRID_START_PHASES]},
                            "grasp_strategy": {
                                "type": "string", "enum": [*HYBRID_GRASP_STRATEGIES]
                            },
                        },
                        "required": [
                            "object_id", "destination", "stack_level", "skill",
                            "start_phase", "grasp_strategy",
                        ],
                        "additionalProperties": False,
                    },
                },
                "on_low_confidence": {
                    "type": "string", "enum": [*LOW_CONFIDENCE_POLICIES]
                },
                "max_retries": {"type": "integer", "minimum": 0, "maximum": 1},
            },
            "required": ["steps", "on_low_confidence", "max_retries"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


class FrankaTools:
    """Tool dispatcher and physical-execution boundary."""

    def __init__(
        self,
        project_root: Path,
        *,
        dry_run: bool = False,
        timeout_s: int = 180,
        persistent_sim: bool = False,
        perception_backend: str = "color",
        semantic_backend: str = "clip",
        semantic_vocabulary: Path | None = None,
        semantic_model: str | None = None,
        semantic_min_confidence: float = 0.50,
        object_set: str = "cubes",
    ) -> None:
        self.project_root = project_root
        self.dry_run = dry_run
        self.timeout_s = timeout_s
        self.scene = SceneState()
        self._attempts: dict[str, int] = {}
        self._vision_pose_path: Path | None = None
        self._persistent_sim = persistent_sim
        self._perception_backend = perception_backend
        self._semantic_backend = semantic_backend
        self._semantic_vocabulary = semantic_vocabulary
        self._semantic_model = semantic_model
        self._semantic_min_confidence = semantic_min_confidence
        self._object_set = object_set
        self._server: subprocess.Popen[str] | None = None
        self._server_log = self.project_root / "skill_server.log"
        self._server_port: int | None = None
        self._server_log_handle = None

    def set_vision_pose_path(self, path: Path) -> None:
        self._vision_pose_path = path

    @property
    def persistent_sim(self) -> bool:
        return self._persistent_sim

    def perception_cli_args(self) -> list[str]:
        """Return the validated perception configuration for Isaac subprocesses."""
        arguments = [
            "--perception-backend", self._perception_backend,
            "--semantic-backend", self._semantic_backend,
            "--semantic-min-confidence", str(self._semantic_min_confidence),
            "--object-set", self._object_set,
        ]
        if self._semantic_vocabulary is not None:
            arguments.extend(["--semantic-vocabulary", str(self._semantic_vocabulary)])
        if self._semantic_model:
            arguments.extend(["--semantic-model", self._semantic_model])
        return arguments

    def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if name == "get_scene_state":
            if arguments:
                return ToolResult(False, "INVALID_ARGUMENT", "get_scene_state takes no arguments", {})
            if self._persistent_sim:
                remote = self._rpc({"tool": "get_scene_state"})
                if not remote.get("ok"):
                    return ToolResult(False, remote.get("code", "SIMULATION_UNAVAILABLE"), remote.get("detail", "service unavailable"), remote.get("data", {}))
                return ToolResult(True, "OK", "Current simulator scene state read", remote.get("data", {}))
            return ToolResult(True, "OK", "Scene state read", self.scene.to_dict())
        if name == "execute_sort_plan":
            try:
                plan = TaskPlan.from_arguments(arguments)
            except (TypeError, ValueError) as error:
                return ToolResult(False, "INVALID_PLAN", str(error), {})
            return self._sort({"targets": plan.targets}, max_retries=plan.max_retries)
        if name == "execute_hybrid_vla_plan":
            try:
                plan = HybridVLAPlan.from_arguments(arguments)
            except (TypeError, ValueError) as error:
                return ToolResult(False, "INVALID_HYBRID_PLAN", str(error), {})
            result = self._sort({"targets": plan.targets}, max_retries=plan.max_retries)
            data = dict(result.data)
            data["hybrid_vla_plan"] = plan.to_arguments()
            return ToolResult(result.ok, result.code, result.detail, data)
        if name == "sort_to_target_box":
            return self._sort(arguments)
        return ToolResult(False, "UNKNOWN_TOOL", f"Tool '{name}' is not permitted", {})

    def _sort(self, arguments: dict[str, Any], *, max_retries: int = 1) -> ToolResult:
        targets = arguments.get("targets")
        if not isinstance(targets, list) or not targets or not all(isinstance(item, str) for item in targets):
            return ToolResult(False, "INVALID_ARGUMENT", "targets must be a non-empty string list", {})
        normalized = [item.lower() for item in targets]
        invalid = [item for item in normalized if item not in (*COLORS, *GROUPS)]
        if invalid:
            return ToolResult(False, "INVALID_TARGET", f"Unknown target(s): {invalid}", {"allowed": [*COLORS, *GROUPS]})
        key = ",".join(normalized)
        if self._attempts.get(key, 0) >= 1 + max_retries:
            return ToolResult(False, "RETRY_LIMIT", "Physical retry limit reached; reset or ask a human to inspect the scene", {})
        self._attempts[key] = self._attempts.get(key, 0) + 1
        if self.dry_run:
            return ToolResult(True, "DRY_RUN", "Validated simulator task; simulator was not started", {"targets": normalized})
        if self._persistent_sim:
            return self._sort_persistent(normalized)
        command = [str(self.project_root.parent / "isaacsim" / "python.sh"), str(self.project_root / "franka_color_sorting.py"), "--target", *normalized]
        command.extend(self.perception_cli_args())
        if self._vision_pose_path and self._vision_pose_path.exists():
            command.extend(["--vision-poses", str(self._vision_pose_path)])
        try:
            completed = subprocess.run(command, cwd=self.project_root, text=True, capture_output=True, timeout=self.timeout_s, check=False)
        except subprocess.TimeoutExpired:
            return ToolResult(False, "MOTION_TIMEOUT", f"Simulation exceeded {self.timeout_s}s", {})
        log = (completed.stdout + "\n" + completed.stderr)[-4000:]
        if completed.returncode == 0 and "Success: requested blocks placed in target box." in completed.stdout:
            return ToolResult(True, "OK", "Simulator verified all requested blocks in target_box", {"targets": normalized, "log_tail": log})
        return ToolResult(False, "SIMULATION_FAILED", "Simulator did not verify the requested placement", {"targets": normalized, "log_tail": log})

    def capture_rgbd(self, image_path: Path) -> ToolResult:
        """Capture an overhead RGB-D observation from the existing simulator."""
        if not self._persistent_sim:
            return ToolResult(False, "UNSUPPORTED", "RGB-D capture requires persistent simulator mode", {})
        result = self._rpc({"tool": "capture_rgbd", "capture_path": str(image_path)})
        return ToolResult(bool(result.get("ok")), result.get("code", "SIMULATION_UNAVAILABLE"), result.get("detail", "capture failed"), result.get("data", {}))

    def capture_wrist_rgbd(self, image_path: Path) -> ToolResult:
        """Capture the current calibrated wrist RGB-D observation."""
        if not self._persistent_sim:
            return ToolResult(
                False, "UNSUPPORTED",
                "Wrist RGB-D capture requires persistent simulator mode", {},
            )
        result = self._rpc({"tool": "capture_wrist_rgbd", "capture_path": str(image_path)})
        return ToolResult(
            bool(result.get("ok")),
            result.get("code", "SIMULATION_UNAVAILABLE"),
            result.get("detail", "wrist capture failed"),
            result.get("data", {}),
        )

    def _sort_persistent(self, targets: list[str]) -> ToolResult:
        colors: list[str] = []
        for target in targets:
            for color in GROUPS.get(target, (target,)):
                if color not in colors:
                    colors.append(color)
        last_state: dict[str, Any] = {}
        execution_events: list[dict[str, Any]] = []
        for color in colors:
            # Take a fresh local RGB-D observation before each individual pick.
            # This happens in the same physics scene, so contact-induced object
            # movement is retained instead of being hidden by a process reset.
            capture_path = self.project_root / "observations" / "latest_overhead.png"
            captured = self.capture_rgbd(capture_path)
            if not captured.ok:
                data = dict(captured.data)
                data.update({
                    "phase": "localize",
                    "object_id": color,
                    "execution_events": execution_events + [{
                        "object_id": color, "phase": "localize",
                        "outcome": "failed", "code": captured.code,
                    }],
                })
                return ToolResult(False, captured.code, captured.detail, data)
            execution_events.append({
                "object_id": color,
                "phase": "localize",
                "outcome": "succeeded",
                "detected": captured.data.get("detected", []),
            })
            state = self._rpc({"tool": "get_scene_state"})
            if not state.get("ok"):
                return ToolResult(False, state.get("code", "SIMULATION_UNAVAILABLE"), state.get("detail", "state unavailable"), state.get("data", {}))
            last_state = state.get("data", {})
            objects = {item.get("id"): item for item in last_state.get("objects", [])}
            if objects.get(color, {}).get("location") == "target_box":
                execution_events.append({
                    "object_id": color, "phase": "verify",
                    "outcome": "already_completed",
                })
                continue
            level = len(last_state.get("target_box", {}).get("contents_bottom_to_top", []))
            execution_events.append({
                "object_id": color,
                "phase": "generate_grasp",
                "outcome": "shape_aware",
            })
            moved = self._rpc({"tool": "move_object_to_box", "object_id": color, "stack_level": level})
            moved_data = moved.get("data", {})
            execution_feedback = (
                moved_data.get("execution_feedback", {})
                if isinstance(moved_data, dict) else {}
            )
            moved_state = (
                moved_data.get("state", moved_data)
                if isinstance(moved_data, dict) else last_state
            )
            if not moved.get("ok"):
                phase = str(execution_feedback.get("phase", "verify"))
                execution_events.append({
                    "object_id": color,
                    "phase": phase,
                    "outcome": "failed",
                    "code": moved.get("code", "SIMULATION_FAILED"),
                })
                return ToolResult(
                    False,
                    moved.get("code", "SIMULATION_FAILED"),
                    moved.get("detail", "placement failed"),
                    {
                        "state": moved_state or last_state,
                        "object_id": color,
                        "phase": phase,
                        "execution_feedback": execution_feedback,
                        "execution_events": execution_events,
                    },
                )
            last_state = moved_state or last_state
            execution_events.append({
                "object_id": color,
                "phase": "verify",
                "outcome": "succeeded",
                "code": moved.get("code", "OK"),
            })
        return ToolResult(
            True,
            "OK",
            "Persistent simulator verified requested blocks in target_box",
            {
                "targets": targets,
                "state": last_state,
                "phase": "completed",
                "execution_events": execution_events,
            },
        )

    def _rpc(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._ensure_server()
        assert self._server_port is not None
        try:
            with socket.create_connection(("127.0.0.1", self._server_port), timeout=10) as connection:
                connection.settimeout(self.timeout_s + 30)
                connection.sendall(json.dumps(payload).encode("utf-8"))
                chunks: list[bytes] = []
                while True:
                    chunk = connection.recv(65536)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    if b"\n" in chunk:
                        break
            return json.loads(b"".join(chunks).decode("utf-8").strip())
        except Exception as error:
            return {"ok": False, "code": "SIMULATION_UNAVAILABLE", "detail": f"persistent skill service failed: {error}", "data": {"log_tail": self._log_tail()}}

    def _ensure_server(self) -> None:
        if self._server is not None and self._server.poll() is None:
            return
        self._server_log.parent.mkdir(parents=True, exist_ok=True)
        failures: list[str] = []
        for attempt in range(1, 3):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind(("127.0.0.1", 0))
                self._server_port = int(probe.getsockname()[1])
            mode = "w" if attempt == 1 else "a"
            self._server_log_handle = self._server_log.open(mode, encoding="utf-8")
            if attempt > 1:
                self._server_log_handle.write(
                    f"\n[agent] retrying persistent Isaac startup attempt {attempt}/2\n"
                )
                self._server_log_handle.flush()
            command = [
                str(self.project_root.parent / "isaacsim" / "python.sh"),
                str(self.project_root / "franka_color_sorting.py"),
                "--server-port", str(self._server_port),
            ]
            command.extend(self.perception_cli_args())
            self._server = subprocess.Popen(
                command,
                cwd=self.project_root,
                text=True,
                stdout=self._server_log_handle,
                stderr=subprocess.STDOUT,
            )
            deadline = time.monotonic() + 120
            ready = f"SKILL_SERVER_READY {self._server_port}"
            while time.monotonic() < deadline:
                if self._server.poll() is not None:
                    break
                if ready in self._server_log.read_text(errors="replace"):
                    return
                time.sleep(0.25)
            failures.append(self._log_tail())
            if self._server.poll() is None:
                self._server.terminate()
                try:
                    self._server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._server.kill()
                    self._server.wait(timeout=10)
            self._server_log_handle.close()
            self._server_log_handle = None
            self._server = None
            time.sleep(1.0)
        raise OSError(
            "Isaac Sim persistent service did not become ready after 2 attempts. "
            f"See {self._server_log}. Last output:\n{failures[-1]}"
        )

    def _log_tail(self) -> str:
        if not self._server_log.exists():
            return ""
        return self._server_log.read_text(errors="replace")[-3000:]

    def close(self) -> None:
        """Terminate the simulator only after the whole agent task is finished."""
        if self._server is not None and self._server.poll() is None:
            self._server.terminate()
            try:
                self._server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._server.kill()
        if self._server_log_handle is not None:
            self._server_log_handle.close()
