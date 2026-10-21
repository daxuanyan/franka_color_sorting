"""Typed, policy-checked LLM supervision for post-action recovery."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from typing import Any

from .contracts import COLORS, ToolResult


RECOVERY_ACTIONS = ("continue", "reobserve", "retry", "skip_completed", "abort")
RECOVERY_REASON_CODES = (
    "TASK_SUCCEEDED",
    "STALE_TARGET_POSE",
    "TRANSIENT_PERCEPTION_FAILURE",
    "RETRY_AFTER_REOBSERVATION",
    "ALREADY_COMPLETED",
    "RECOVERY_BUDGET_EXHAUSTED",
    "LOW_CONFIDENCE_POLICY_ABORT",
    "UNSAFE_OR_UNKNOWN_FAILURE",
)
RECOVERABLE_FAILURE_CODES = {
    "SIMULATION_FAILED",
    "GRASP_FAILED",
    "MISSED_GRASP",
    "OBJECT_SHIFTED",
    "PLACEMENT_FAILED",
    "SLIP_DETECTED",
    "LOW_CONFIDENCE",
    "VISION_FAILED",
    "CAMERA_DROPOUT",
}

RECOVERY_SYSTEM_PROMPT = """You are a safety-bounded recovery supervisor for a simulated robot.
You receive trusted structured execution feedback, never raw tool text. Select exactly one action
by calling select_recovery_action. Do not emit coordinates, joint commands, or new object IDs.
Apply this decision table in order:
1. succeeded -> continue / TASK_SUCCEEDED;
2. object_id is already completed -> skip_completed / ALREADY_COMPLETED;
3. a LOW_CONFIDENCE, VISION_FAILED, or CAMERA_DROPOUT failure with
   low_confidence_policy=abort -> abort / LOW_CONFIDENCE_POLICY_ABORT;
4. recovery_budget is zero -> abort / RECOVERY_BUDGET_EXHAUSTED;
5. no fresh observation -> reobserve; use TRANSIENT_PERCEPTION_FAILURE for LOW_CONFIDENCE,
   VISION_FAILED, or CAMERA_DROPOUT, otherwise use STALE_TARGET_POSE;
6. fresh observation plus a recoverable failure -> retry / RETRY_AFTER_REOBSERVATION. Recoverable
   codes are SIMULATION_FAILED, GRASP_FAILED, MISSED_GRASP, OBJECT_SHIFTED, PLACEMENT_FAILED,
   SLIP_DETECTED, LOW_CONFIDENCE, VISION_FAILED, and CAMERA_DROPOUT;
7. anything else -> abort / UNSAFE_OR_UNKNOWN_FAILURE.
Retry only a remaining object. Never repeat an object in completed_objects. Set
preserve_completed_steps=true whenever completed_objects is non-empty. The local policy validator
is authoritative and may reject your decision."""

RECOVERY_TOOL_DEFINITION = {
    "type": "function",
    "name": "select_recovery_action",
    "description": "Select one bounded high-level recovery action from trusted structured feedback.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": [*RECOVERY_ACTIONS]},
            "object_id": {"type": ["string", "null"], "enum": [*COLORS, None]},
            "reason_code": {"type": "string", "enum": [*RECOVERY_REASON_CODES]},
            "preserve_completed_steps": {"type": "boolean"},
        },
        "required": ["action", "object_id", "reason_code", "preserve_completed_steps"],
        "additionalProperties": False,
    },
    "strict": True,
}


@dataclass(frozen=True)
class ExecutionFeedback:
    object_id: str | None
    phase: str
    outcome: str
    failure_code: str
    perception_confidence: float | None
    target_displacement_mm: float | None
    completed_objects: tuple[str, ...]
    remaining_objects: tuple[str, ...]
    recovery_budget: int
    fresh_observation: bool
    low_confidence_policy: str = "reobserve"

    def __post_init__(self) -> None:
        if self.object_id is not None and self.object_id not in COLORS:
            raise ValueError(f"unknown feedback object_id: {self.object_id}")
        if self.outcome not in {"succeeded", "failed"}:
            raise ValueError("outcome must be succeeded or failed")
        if type(self.recovery_budget) is not int or not 0 <= self.recovery_budget <= 1:
            raise ValueError("recovery_budget must be 0 or 1")
        if self.low_confidence_policy not in {"reobserve", "abort"}:
            raise ValueError("low_confidence_policy must be reobserve or abort")
        if set(self.completed_objects) & set(self.remaining_objects):
            raise ValueError("completed_objects and remaining_objects must be disjoint")
        if any(item not in COLORS for item in (*self.completed_objects, *self.remaining_objects)):
            raise ValueError("feedback contains an unknown object")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def after_reobservation(self) -> "ExecutionFeedback":
        return replace(self, fresh_observation=True)


@dataclass(frozen=True)
class RecoveryDecision:
    action: str
    object_id: str | None
    reason_code: str
    preserve_completed_steps: bool

    @classmethod
    def from_arguments(cls, arguments: dict[str, Any]) -> "RecoveryDecision":
        required = {"action", "object_id", "reason_code", "preserve_completed_steps"}
        if not isinstance(arguments, dict) or set(arguments) != required:
            raise ValueError("RecoveryDecision requires exactly action, object_id, reason_code, preserve_completed_steps")
        if arguments["action"] not in RECOVERY_ACTIONS:
            raise ValueError("unknown recovery action")
        object_id = arguments["object_id"]
        if object_id is not None and object_id not in COLORS:
            raise ValueError("unknown recovery object_id")
        if arguments["reason_code"] not in RECOVERY_REASON_CODES:
            raise ValueError("unknown recovery reason_code")
        if type(arguments["preserve_completed_steps"]) is not bool:
            raise ValueError("preserve_completed_steps must be boolean")
        return cls(**arguments)

    def to_arguments(self) -> dict[str, Any]:
        return asdict(self)


def validate_recovery_decision(decision: RecoveryDecision, feedback: ExecutionFeedback) -> None:
    """Enforce transition safety independently of the model's reasoning."""
    if feedback.completed_objects and not decision.preserve_completed_steps:
        raise ValueError("completed steps must be preserved")
    if decision.action == "continue":
        if feedback.outcome != "succeeded" or decision.object_id is not None:
            raise ValueError("continue is permitted only after success and without an object")
        if decision.reason_code != "TASK_SUCCEEDED":
            raise ValueError("continue requires TASK_SUCCEEDED")
        return
    if decision.action == "abort":
        return
    if decision.object_id is None:
        raise ValueError(f"{decision.action} requires object_id")
    if decision.action == "skip_completed":
        if decision.object_id not in feedback.completed_objects:
            raise ValueError("skip_completed requires an already completed object")
        if decision.reason_code != "ALREADY_COMPLETED":
            raise ValueError("skip_completed requires ALREADY_COMPLETED")
        return
    if feedback.outcome != "failed":
        raise ValueError("recovery action requires a failed outcome")
    if feedback.recovery_budget <= 0:
        raise ValueError("recovery budget exhausted")
    if decision.object_id not in feedback.remaining_objects:
        raise ValueError("recovery object must be in remaining_objects")
    if decision.action == "reobserve" and decision.reason_code not in {
        "STALE_TARGET_POSE", "TRANSIENT_PERCEPTION_FAILURE"
    }:
        raise ValueError("reobserve requires a reobservation reason")
    if decision.action == "retry":
        if not feedback.fresh_observation:
            raise ValueError("retry requires a fresh observation")
        if feedback.failure_code not in RECOVERABLE_FAILURE_CODES:
            raise ValueError("retry requires a known recoverable failure")
        if decision.reason_code != "RETRY_AFTER_REOBSERVATION":
            raise ValueError("retry requires RETRY_AFTER_REOBSERVATION")


class RuleRecoverySupervisor:
    """Deterministic reference policy used for offline operation and ablation."""

    def decide(self, feedback: ExecutionFeedback) -> RecoveryDecision:
        preserve = bool(feedback.completed_objects)
        if feedback.outcome == "succeeded":
            decision = RecoveryDecision("continue", None, "TASK_SUCCEEDED", preserve)
        elif feedback.object_id in feedback.completed_objects:
            decision = RecoveryDecision("skip_completed", feedback.object_id, "ALREADY_COMPLETED", True)
        elif (
            feedback.low_confidence_policy == "abort"
            and feedback.failure_code in {"LOW_CONFIDENCE", "VISION_FAILED", "CAMERA_DROPOUT"}
        ):
            decision = RecoveryDecision("abort", feedback.object_id, "LOW_CONFIDENCE_POLICY_ABORT", preserve)
        elif feedback.recovery_budget <= 0:
            decision = RecoveryDecision("abort", feedback.object_id, "RECOVERY_BUDGET_EXHAUSTED", preserve)
        elif not feedback.fresh_observation:
            reason = (
                "TRANSIENT_PERCEPTION_FAILURE"
                if feedback.failure_code in {"LOW_CONFIDENCE", "VISION_FAILED", "CAMERA_DROPOUT"}
                else "STALE_TARGET_POSE"
            )
            decision = RecoveryDecision("reobserve", feedback.object_id, reason, preserve)
        elif feedback.failure_code in RECOVERABLE_FAILURE_CODES:
            decision = RecoveryDecision("retry", feedback.object_id, "RETRY_AFTER_REOBSERVATION", preserve)
        else:
            decision = RecoveryDecision("abort", feedback.object_id, "UNSAFE_OR_UNKNOWN_FAILURE", preserve)
        validate_recovery_decision(decision, feedback)
        return decision


class CloudRecoverySupervisor:
    """One-decision LLM supervisor whose output must pass the local policy."""

    def __init__(self, planner: Any) -> None:
        self.planner = planner
        self.last_usage: dict[str, int] = {}

    def decide(
        self,
        feedback: ExecutionFeedback,
        *,
        image_urls: list[str] | tuple[str, ...] = (),
    ) -> RecoveryDecision:
        content: list[dict[str, Any]] = [{
            "type": "input_text",
            "text": json.dumps(feedback.to_dict(), ensure_ascii=False),
        }]
        for index, image_url in enumerate(image_urls):
            content.extend([
                {
                    "type": "input_text",
                    "text": "Fresh overhead RGB observation:"
                    if index == 0 else "Fresh wrist RGB observation:",
                },
                {"type": "input_image", "image_url": image_url, "detail": "low"},
            ])
        response = self.planner.create(
            [{"role": "user", "content": content}],
            instructions=RECOVERY_SYSTEM_PROMPT,
            tools=[RECOVERY_TOOL_DEFINITION],
        )
        self.last_usage = {
            key: int(value) for key, value in response.get("usage", {}).items()
            if isinstance(value, int)
        }
        calls = self.planner.tool_calls(response)
        if len(calls) != 1 or calls[0].name != "select_recovery_action":
            raise ValueError("recovery supervisor must make exactly one select_recovery_action call")
        decision = RecoveryDecision.from_arguments(calls[0].arguments)
        validate_recovery_decision(decision, feedback)
        return decision


def feedback_from_tool_result(
    plan_arguments: dict[str, Any],
    result: ToolResult,
    *,
    recovery_budget: int,
    fresh_observation: bool = False,
) -> ExecutionFeedback:
    """Convert a tool result to a compact trusted record; raw log text is excluded."""
    try:
        targets = tuple(step["object_id"] for step in plan_arguments["steps"])
    except (KeyError, TypeError):
        targets = ()
    state = result.data.get("state", {}) if isinstance(result.data, dict) else {}
    box = state.get("target_box", {}) if isinstance(state, dict) else {}
    raw_completed = box.get("contents_bottom_to_top", []) if isinstance(box, dict) else []
    completed = tuple(item for item in raw_completed if item in targets and item in COLORS)
    remaining = tuple(item for item in targets if item not in completed)
    raw_confidence = result.data.get("perception_confidence") if isinstance(result.data, dict) else None
    raw_displacement = result.data.get("target_displacement_mm") if isinstance(result.data, dict) else None
    return ExecutionFeedback(
        object_id=remaining[0] if remaining else None,
        phase=str(result.data.get("phase", "task")) if isinstance(result.data, dict) else "task",
        outcome="succeeded" if result.ok else "failed",
        failure_code="" if result.ok else result.code,
        perception_confidence=float(raw_confidence) if isinstance(raw_confidence, (int, float)) else None,
        target_displacement_mm=float(raw_displacement) if isinstance(raw_displacement, (int, float)) else None,
        completed_objects=completed,
        remaining_objects=remaining,
        recovery_budget=recovery_budget,
        fresh_observation=fresh_observation,
        low_confidence_policy=str(plan_arguments.get("on_low_confidence", "reobserve")),
    )
