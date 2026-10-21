"""Validated task plans compiled by an LLM and executed by local robot code."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .contracts import COLORS


LOW_CONFIDENCE_POLICIES = ("reobserve", "abort")


@dataclass(frozen=True)
class TaskStep:
    object_id: str
    destination: str
    stack_level: int


@dataclass(frozen=True)
class TaskPlan:
    steps: tuple[TaskStep, ...]
    on_low_confidence: str
    max_retries: int

    @classmethod
    def from_arguments(cls, arguments: dict[str, Any]) -> "TaskPlan":
        if set(arguments) != {"steps", "on_low_confidence", "max_retries"}:
            raise ValueError("TaskPlan requires exactly steps, on_low_confidence, and max_retries")
        raw_steps = arguments["steps"]
        if not isinstance(raw_steps, list) or not 1 <= len(raw_steps) <= len(COLORS):
            raise ValueError("steps must contain between one and four actions")
        steps: list[TaskStep] = []
        seen: set[str] = set()
        for index, raw in enumerate(raw_steps):
            if not isinstance(raw, dict) or set(raw) != {"object_id", "destination", "stack_level"}:
                raise ValueError(f"step {index} has an invalid schema")
            object_id = raw["object_id"]
            if object_id not in COLORS:
                raise ValueError(f"step {index} has unknown object_id: {object_id}")
            if object_id in seen:
                raise ValueError(f"object {object_id} appears more than once")
            if raw["destination"] != "target_box":
                raise ValueError("target_box is the only permitted destination")
            if type(raw["stack_level"]) is not int or raw["stack_level"] != index:
                raise ValueError("stack_level must be contiguous and match execution order")
            seen.add(object_id)
            steps.append(TaskStep(object_id, "target_box", index))
        policy = arguments["on_low_confidence"]
        if policy not in LOW_CONFIDENCE_POLICIES:
            raise ValueError(f"on_low_confidence must be one of: {', '.join(LOW_CONFIDENCE_POLICIES)}")
        retries = arguments["max_retries"]
        if type(retries) is not int or not 0 <= retries <= 1:
            raise ValueError("max_retries must be 0 or 1")
        return cls(tuple(steps), policy, retries)

    @classmethod
    def for_targets(
        cls, targets: list[str] | tuple[str, ...], *, on_low_confidence: str = "reobserve", max_retries: int = 1
    ) -> "TaskPlan":
        return cls.from_arguments(
            {
                "steps": [
                    {"object_id": object_id, "destination": "target_box", "stack_level": index}
                    for index, object_id in enumerate(targets)
                ],
                "on_low_confidence": on_low_confidence,
                "max_retries": max_retries,
            }
        )

    @property
    def targets(self) -> list[str]:
        return [step.object_id for step in self.steps]

    def to_arguments(self) -> dict[str, Any]:
        return {
            "steps": [asdict(step) for step in self.steps],
            "on_low_confidence": self.on_low_confidence,
            "max_retries": self.max_retries,
        }
