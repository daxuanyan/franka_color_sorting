"""Validated high-level skill contract for the hierarchical Hybrid VLA."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .contracts import COLORS
from .task_plan import LOW_CONFIDENCE_POLICIES, TaskPlan


HYBRID_SKILLS = ("pick_and_place",)
HYBRID_START_PHASES = ("localize", "recover")
HYBRID_GRASP_STRATEGIES = ("shape_aware",)


@dataclass(frozen=True)
class HybridVLAStep:
    object_id: str
    destination: str
    stack_level: int
    skill: str
    start_phase: str
    grasp_strategy: str


@dataclass(frozen=True)
class HybridVLAPlan:
    """A visual-language decision whose metric execution remains local."""

    steps: tuple[HybridVLAStep, ...]
    on_low_confidence: str
    max_retries: int

    @classmethod
    def from_arguments(cls, arguments: dict[str, Any]) -> "HybridVLAPlan":
        if not isinstance(arguments, dict) or set(arguments) != {
            "steps", "on_low_confidence", "max_retries"
        }:
            raise ValueError(
                "HybridVLAPlan requires exactly steps, on_low_confidence, and max_retries"
            )
        raw_steps = arguments["steps"]
        if not isinstance(raw_steps, list) or not 1 <= len(raw_steps) <= len(COLORS):
            raise ValueError("steps must contain between one and four skills")
        required = {
            "object_id", "destination", "stack_level", "skill",
            "start_phase", "grasp_strategy",
        }
        steps: list[HybridVLAStep] = []
        seen: set[str] = set()
        for index, raw in enumerate(raw_steps):
            if not isinstance(raw, dict) or set(raw) != required:
                raise ValueError(f"hybrid step {index} has an invalid schema")
            if raw["object_id"] not in COLORS or raw["object_id"] in seen:
                raise ValueError(f"hybrid step {index} has an unknown or duplicate object")
            if raw["destination"] != "target_box":
                raise ValueError("target_box is the only permitted destination")
            if type(raw["stack_level"]) is not int or raw["stack_level"] != index:
                raise ValueError("stack_level must be contiguous and match execution order")
            if raw["skill"] not in HYBRID_SKILLS:
                raise ValueError("unsupported hybrid skill")
            if raw["start_phase"] not in HYBRID_START_PHASES:
                raise ValueError("unsupported hybrid start phase")
            if index == 0 and raw["start_phase"] == "recover":
                raise ValueError("a new task must start with localize, not recover")
            if raw["grasp_strategy"] not in HYBRID_GRASP_STRATEGIES:
                raise ValueError("unsupported grasp strategy")
            seen.add(raw["object_id"])
            steps.append(HybridVLAStep(**raw))
        policy = arguments["on_low_confidence"]
        if policy not in LOW_CONFIDENCE_POLICIES:
            raise ValueError("invalid low-confidence policy")
        retries = arguments["max_retries"]
        if type(retries) is not int or not 0 <= retries <= 1:
            raise ValueError("max_retries must be 0 or 1")
        return cls(tuple(steps), policy, retries)

    @classmethod
    def for_targets(
        cls,
        targets: list[str] | tuple[str, ...],
        *,
        on_low_confidence: str = "reobserve",
        max_retries: int = 1,
        recovery: bool = False,
    ) -> "HybridVLAPlan":
        return cls.from_arguments({
            "steps": [
                {
                    "object_id": object_id,
                    "destination": "target_box",
                    "stack_level": index,
                    "skill": "pick_and_place",
                    "start_phase": "recover" if recovery and index > 0 else "localize",
                    "grasp_strategy": "shape_aware",
                }
                for index, object_id in enumerate(targets)
            ],
            "on_low_confidence": on_low_confidence,
            "max_retries": max_retries,
        })

    @property
    def targets(self) -> list[str]:
        return [step.object_id for step in self.steps]

    def to_task_plan(self) -> TaskPlan:
        return TaskPlan.for_targets(
            self.targets,
            on_low_confidence=self.on_low_confidence,
            max_retries=self.max_retries,
        )

    def to_arguments(self) -> dict[str, Any]:
        return {
            "steps": [asdict(step) for step in self.steps],
            "on_low_confidence": self.on_low_confidence,
            "max_retries": self.max_retries,
        }
