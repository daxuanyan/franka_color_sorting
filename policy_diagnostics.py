"""Stage-aware diagnostics for learned-policy rollouts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image


def infer_failure_stage(
    *,
    placed: bool,
    min_eef_distance_m: float,
    max_lift_m: float,
    max_displacement_m: float,
) -> str:
    """Infer a coarse failure stage from physics telemetry, not policy internals."""
    if placed:
        return "completed"
    if min_eef_distance_m > 0.14:
        return "approach"
    if max_lift_m < 0.015 and max_displacement_m < 0.010:
        return "contact_or_grasp"
    if max_lift_m < 0.030:
        return "lift"
    return "transport_or_release"


class PolicyRolloutTelemetry:
    """Persist compact JSONL command traces and one rollout summary."""

    def __init__(
        self,
        artifact_dir: Path,
        initial_positions: Mapping[str, np.ndarray],
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.artifact_dir / "policy_telemetry.jsonl"
        self.summary_path = self.artifact_dir / "policy_rollout_summary.json"
        self.trace_path.unlink(missing_ok=True)
        self.initial_positions = {
            name: np.asarray(position, dtype=float).reshape(3).copy()
            for name, position in initial_positions.items()
        }
        self.metrics = {
            name: {
                "min_eef_distance_m": float("inf"),
                "max_lift_m": 0.0,
                "max_displacement_m": 0.0,
                "final_position": position.astype(float).tolist(),
            }
            for name, position in self.initial_positions.items()
        }
        self.command_records = 0

    def update(
        self,
        end_effector_position: np.ndarray,
        object_positions: Mapping[str, np.ndarray],
    ) -> None:
        eef = np.asarray(end_effector_position, dtype=float).reshape(3)
        for name, initial in self.initial_positions.items():
            position = np.asarray(object_positions[name], dtype=float).reshape(3)
            metric = self.metrics[name]
            metric["min_eef_distance_m"] = min(
                float(metric["min_eef_distance_m"]), float(np.linalg.norm(eef - position))
            )
            metric["max_lift_m"] = max(
                float(metric["max_lift_m"]), float(position[2] - initial[2])
            )
            metric["max_displacement_m"] = max(
                float(metric["max_displacement_m"]), float(np.linalg.norm(position - initial))
            )
            metric["final_position"] = position.astype(float).tolist()

    def record_command(
        self,
        *,
        time_s: float,
        goal: str,
        inference_count: int,
        end_effector_position: np.ndarray,
        gripper_width_m: float,
        diagnostics: Mapping,
    ) -> None:
        record = {
            "time_s": float(time_s),
            "goal": goal,
            "inference_count": int(inference_count),
            "end_effector_position": np.asarray(
                end_effector_position, dtype=float
            ).reshape(3).tolist(),
            "gripper_width_m": float(gripper_width_m),
            **dict(diagnostics),
        }
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.command_records += 1

    def finalize(
        self,
        *,
        success: bool,
        placed: Mapping[str, bool],
        runtime,
        elapsed_s: float,
    ) -> dict:
        targets = {}
        for name, metric in self.metrics.items():
            finite_distance = float(metric["min_eef_distance_m"])
            if not np.isfinite(finite_distance):
                finite_distance = -1.0
            normalized = {
                **metric,
                "min_eef_distance_m": finite_distance,
                "placed": bool(placed.get(name, False)),
            }
            normalized["failure_stage"] = infer_failure_stage(
                placed=normalized["placed"],
                min_eef_distance_m=finite_distance,
                max_lift_m=float(normalized["max_lift_m"]),
                max_displacement_m=float(normalized["max_displacement_m"]),
            )
            targets[name] = normalized
        command_count = int(runtime.command_count)
        summary = {
            "schema_version": 1,
            "success": bool(success),
            "elapsed_s": float(elapsed_s),
            "inference_count": int(runtime.inference_count),
            "command_count": command_count,
            "command_records": self.command_records,
            "clipped_command_count": int(runtime.clipped_command_count),
            "clipped_command_rate": (
                float(runtime.clipped_command_count) / command_count if command_count else 0.0
            ),
            "arm_step_clipped_command_count": int(
                runtime.arm_step_clipped_command_count
            ),
            "arm_step_clipped_command_rate": (
                float(runtime.arm_step_clipped_command_count) / command_count
                if command_count else 0.0
            ),
            "joint_limit_clipped_command_count": int(
                runtime.joint_limit_clipped_command_count
            ),
            "joint_limit_clipped_command_rate": (
                float(runtime.joint_limit_clipped_command_count) / command_count
                if command_count else 0.0
            ),
            "clipped_joint_value_count": int(runtime.clipped_joint_value_count),
            "max_filter_correction": float(runtime.max_filter_correction),
            "adaptive_reobservation_count": int(runtime.adaptive_reobservation_count),
            "targets": targets,
        }
        self.summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return summary


def save_policy_observation(
    artifact_dir: Path,
    sample_index: int,
    *,
    overhead_rgb: np.ndarray,
    overhead_depth_m: np.ndarray,
    wrist_rgb: np.ndarray,
    wrist_depth_m: np.ndarray,
) -> None:
    """Save an opt-in RGB-D observation without affecting policy inputs."""
    root = Path(artifact_dir) / "policy_observations"
    root.mkdir(parents=True, exist_ok=True)
    stem = f"observation_{int(sample_index):04d}"
    Image.fromarray(np.asarray(overhead_rgb[..., :3], dtype=np.uint8)).save(
        root / f"{stem}_overhead.png"
    )
    Image.fromarray(np.asarray(wrist_rgb[..., :3], dtype=np.uint8)).save(
        root / f"{stem}_wrist.png"
    )
    np.savez_compressed(
        root / f"{stem}_depth.npz",
        overhead_depth_m=np.asarray(overhead_depth_m, dtype=np.float32),
        wrist_depth_m=np.asarray(wrist_depth_m, dtype=np.float32),
    )
