"""Run reproducible randomized RGB-D Franka sorting benchmarks.

The orchestrator intentionally starts a fresh Isaac Sim process per trial so
physics, controller, and render-product state cannot leak between trials.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from object_catalog import object_specs


COLORS = ("red", "blue", "green", "yellow")
FAULT_TYPES = ("pose_bias", "dynamic_shift", "sensor_degradation")
SUCCESS_MARKER = "Success: requested blocks placed in target box."
INFRASTRUCTURE_ERROR_MARKERS = {
    "asset_root_unavailable": "Could not find assets root folder",
    "gpu_unavailable": "no CUDA-capable device is detected",
    "invalid_cli_configuration": "error: A learned-policy checkpoint is a standalone rollout mode",
}


def sample_positions(
    rng: random.Random,
    *,
    x_range: tuple[float, float] = (0.38, 0.62),
    y_range: tuple[float, float] = (-0.52, -0.24),
    min_spacing: float = 0.075,
    object_set: str = "cubes",
) -> dict[str, list[float]]:
    """Sample collision-free cube centres in the reachable source area."""
    positions: dict[str, list[float]] = {}
    specs = object_specs(object_set)
    for color in COLORS:
        for _ in range(1000):
            candidate = [
                rng.uniform(*x_range),
                rng.uniform(*y_range),
                specs[color].dimensions_xyz[2] / 2.0,
            ]
            if all(math.dist(candidate[:2], existing[:2]) >= min_spacing for existing in positions.values()):
                positions[color] = candidate
                break
        else:
            raise RuntimeError("Could not sample a collision-free layout; enlarge the region or reduce min spacing")
    return positions


def generate_scenario(
    trial_index: int,
    trial_seed: int,
    *,
    min_targets: int = 1,
    max_targets: int = 2,
    camera_jitter_m: float = 0.03,
    light_range: tuple[float, float] = (600.0, 1400.0),
    fault_profile: str = "none",
    object_set: str = "cubes",
    target_policy: str = "random",
) -> dict[str, Any]:
    """Generate one deterministic scenario from its recorded trial seed."""
    rng = random.Random(trial_seed)
    if target_policy == "cycle":
        targets = [COLORS[trial_index % len(COLORS)]]
    elif target_policy == "random":
        count = rng.randint(min_targets, max_targets)
        targets = rng.sample(list(COLORS), count)
    else:
        raise ValueError(f"unknown target policy: {target_policy}")
    eye = [
        0.50 + rng.uniform(-camera_jitter_m, camera_jitter_m),
        -0.17 + rng.uniform(-camera_jitter_m, camera_jitter_m),
        1.60 + rng.uniform(-camera_jitter_m, camera_jitter_m),
    ]
    target = [
        0.50 + rng.uniform(-camera_jitter_m / 2, camera_jitter_m / 2),
        -0.17 + rng.uniform(-camera_jitter_m / 2, camera_jitter_m / 2),
        0.0,
    ]
    # Multi-shape objects need a larger image margin than the legacy cubes;
    # keep the held-out benchmark inside the calibrated overhead field of view.
    positions = sample_positions(
        rng,
        y_range=(-0.48, -0.24) if object_set != "cubes" else (-0.52, -0.24),
        object_set=object_set,
    )
    faults = generate_faults(rng, trial_index, targets, positions, fault_profile)
    return {
        "schema_version": 1,
        "trial_index": trial_index,
        "seed": trial_seed,
        "targets": targets,
        "spawn_positions": positions,
        "camera": {"eye": eye, "target": target},
        "dome_light_intensity": rng.uniform(*light_range),
        "faults": faults,
        "object_set": object_set,
    }


def generate_faults(
    rng: random.Random,
    trial_index: int,
    targets: list[str],
    positions: dict[str, list[float]],
    profile: str,
) -> dict[str, Any]:
    """Generate deterministic, recorded fault parameters for one scenario."""
    fault_type = FAULT_TYPES[trial_index % len(FAULT_TYPES)] if profile == "mixed" else profile
    faults: dict[str, Any] = {"type": fault_type}
    if fault_type == "none":
        return faults
    if fault_type == "pose_bias":
        magnitude = rng.uniform(0.020, 0.040)
        angle = rng.uniform(0.0, 2.0 * math.pi)
        faults["initial_overhead_xy_bias_m"] = [magnitude * math.cos(angle), magnitude * math.sin(angle)]
    elif fault_type == "dynamic_shift":
        color = targets[0]
        origin = positions[color]
        for _ in range(100):
            magnitude = rng.uniform(0.020, 0.038)
            angle = rng.uniform(0.0, 2.0 * math.pi)
            delta = [magnitude * math.cos(angle), magnitude * math.sin(angle)]
            moved = [origin[0] + delta[0], origin[1] + delta[1]]
            if not (0.32 <= moved[0] <= 0.68 and -0.57 <= moved[1] <= -0.19):
                continue
            if all(
                other == color or math.dist(moved, position[:2]) >= 0.065
                for other, position in positions.items()
            ):
                faults["target_shift"] = {"color": color, "xy_m": delta}
                break
        else:
            raise RuntimeError("Could not sample a collision-free dynamic target shift")
    elif fault_type == "sensor_degradation":
        magnitude = rng.uniform(0.015, 0.030)
        angle = rng.uniform(0.0, 2.0 * math.pi)
        faults.update(
            {
                "initial_overhead_xy_bias_m": [magnitude * math.cos(angle), magnitude * math.sin(angle)],
                "wrist_dropout_probability": rng.uniform(0.30, 0.50),
                "wrist_noise_std_m": rng.uniform(0.0025, 0.0050),
            }
        )
    else:
        raise ValueError(f"Unknown fault profile: {profile}")
    return faults


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percent / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - rank) + ordered[high] * (rank - low)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    """Return a two-sided Wilson score interval for a binomial proportion."""
    if total == 0:
        return None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) / denominator
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def classify_infrastructure_error(log: str) -> str | None:
    """Identify launcher/runtime failures that must not count as robot failures."""
    for reason, marker in INFRASTRUCTURE_ERROR_MARKERS.items():
        if marker in log:
            return reason
    return None


def localization_errors(scenario: dict[str, Any], pose_path: Path) -> list[dict[str, Any]]:
    """Compare visual estimates with authored positions without affecting control."""
    if not pose_path.exists():
        return []
    payload = json.loads(pose_path.read_text())
    estimates = payload.get("poses", {})
    rows: list[dict[str, Any]] = []
    for color, truth in scenario["spawn_positions"].items():
        estimate = estimates.get(color)
        if not isinstance(estimate, list) or len(estimate) != 3:
            continue
        dx, dy, dz = (float(estimate[i]) - float(truth[i]) for i in range(3))
        rows.append(
            {
                "color": color,
                "xy_error_mm": math.hypot(dx, dy) * 1000.0,
                "xyz_error_mm": math.sqrt(dx * dx + dy * dy + dz * dz) * 1000.0,
            }
        )
    return rows


def control_truth_position(scenario: dict[str, Any], color: str) -> list[float]:
    """Final physical source pose after any recorded external disturbance."""
    truth = list(scenario["spawn_positions"][color])
    shift = scenario.get("faults", {}).get("target_shift")
    if isinstance(shift, dict) and shift.get("color") == color:
        truth[0] += float(shift["xy_m"][0])
        truth[1] += float(shift["xy_m"][1])
    return truth


def wrist_localization_errors(scenario: dict[str, Any], trial_dir: Path) -> list[dict[str, Any]]:
    """Measure each target's close-range wrist estimate against authored truth."""
    rows: list[dict[str, Any]] = []
    for color in scenario["targets"]:
        paths = sorted(trial_dir.glob(f"wrist_{color}_servo_*_poses.json"))
        if not paths:
            legacy = trial_dir / f"wrist_{color}_pregrasp_poses.json"
            paths = [legacy] if legacy.exists() else []
        for sample_index, path in enumerate(paths, start=1):
            estimate = json.loads(path.read_text()).get("poses", {}).get(color)
            if not isinstance(estimate, list) or len(estimate) != 3:
                continue
            truth = control_truth_position(scenario, color)
            dx, dy, dz = (float(estimate[i]) - float(truth[i]) for i in range(3))
            rows.append(
                {
                    "color": color,
                    "sample": sample_index,
                    "xy_error_mm": math.hypot(dx, dy) * 1000.0,
                    "xyz_error_mm": math.sqrt(dx * dx + dy * dy + dz * dz) * 1000.0,
                }
            )
    return rows


def visual_servo_metrics(scenario: dict[str, Any], trial_dir: Path) -> list[dict[str, Any]]:
    """Measure accepted visual-servo command convergence for each target."""
    path = trial_dir / "visual_servo_trace.jsonl"
    if not path.exists():
        return []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        color = record.get("color")
        if color in scenario["targets"] and record.get("accepted") is True:
            grouped.setdefault(color, []).append(record)
    rows: list[dict[str, Any]] = []
    for color, records in grouped.items():
        initial = records[0].get("prior_command")
        final = records[-1].get("command")
        if not isinstance(initial, list) or not isinstance(final, list) or len(initial) != 3 or len(final) != 3:
            continue
        truth = control_truth_position(scenario, color)
        initial_error = math.hypot(float(initial[0]) - truth[0], float(initial[1]) - truth[1]) * 1000.0
        final_error = math.hypot(float(final[0]) - truth[0], float(final[1]) - truth[1]) * 1000.0
        rows.append(
            {
                "color": color,
                "updates": len(records),
                "initial_xy_error_mm": initial_error,
                "final_xy_error_mm": final_error,
                "xy_improvement_mm": initial_error - final_error,
            }
        )
    return rows


def generic_perception_outcome(trial_dir: Path, targets: list[str]) -> dict[str, Any]:
    """Separate semantic detections from geometry-based grasp rejection."""
    path = trial_dir / "latest_overhead_instances" / "generic_perception.json"
    empty = {
        "semantic_labels": [],
        "feasible_labels": [],
        "geometry_rejections": {},
        "safe_rejection": False,
    }
    if not path.exists():
        return empty
    payload = json.loads(path.read_text())
    accepted = payload.get("assignments", {})
    rejected = payload.get("rejected_assignments", {})
    semantic_threshold = float(payload.get("min_semantic_confidence", 0.5))
    instances = {
        item.get("instance_id"): item
        for item in payload.get("instances", [])
        if isinstance(item, dict) and isinstance(item.get("instance_id"), str)
    }
    semantic_labels = sorted({
        assignment.get("label")
        for assignment in (*accepted.values(), *rejected.values())
        if (
            isinstance(assignment, dict)
            and isinstance(assignment.get("label"), str)
            and float(assignment.get("confidence", 0.0)) >= semantic_threshold
        )
    })
    feasible_labels = sorted({
        assignment.get("label")
        for assignment in accepted.values()
        if isinstance(assignment, dict) and isinstance(assignment.get("label"), str)
    })
    geometry_rejections: dict[str, str] = {}
    for instance_id, assignment in rejected.items():
        if not isinstance(assignment, dict):
            continue
        label = assignment.get("label")
        if float(assignment.get("confidence", 0.0)) < semantic_threshold:
            continue
        descriptor = instances.get(instance_id, {}).get("grasp_geometry", {})
        reason = descriptor.get("rejection_reason") if isinstance(descriptor, dict) else None
        if isinstance(label, str) and isinstance(reason, str) and reason:
            geometry_rejections[label] = reason
    return {
        "semantic_labels": semantic_labels,
        "feasible_labels": feasible_labels,
        "geometry_rejections": geometry_rejections,
        "safe_rejection": bool(targets) and all(target in geometry_rejections for target in targets),
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [item for item in results if item["status"] == "completed"]
    infrastructure_errors = [item for item in results if item["status"] == "infrastructure_error"]
    successes = [item for item in completed if item["success"]]
    xy_errors = [row["xy_error_mm"] for item in completed for row in item["localization"]]
    xyz_errors = [row["xyz_error_mm"] for item in completed for row in item["localization"]]
    wrist_xy_errors = [row["xy_error_mm"] for item in completed for row in item.get("wrist_localization", [])]
    servo_rows = [row for item in completed for row in item.get("visual_servo", [])]
    durations = [float(item["duration_s"]) for item in completed]
    successful_durations = [float(item["duration_s"]) for item in successes]
    by_target_count: dict[str, dict[str, Any]] = {}
    for count in range(1, len(COLORS) + 1):
        group = [item for item in completed if len(item["targets"]) == count]
        if group:
            group_successes = sum(bool(item["success"]) for item in group)
            by_target_count[str(count)] = {
                "trials": len(group),
                "successes": group_successes,
                "success_rate": group_successes / len(group),
            }
    by_fault_type: dict[str, dict[str, Any]] = {}
    for fault_type in ("none", *FAULT_TYPES):
        group = [item for item in completed if item.get("fault_type", "none") == fault_type]
        if group:
            group_successes = sum(bool(item["success"]) for item in group)
            by_fault_type[fault_type] = {
                "trials": len(group),
                "successes": group_successes,
                "success_rate": group_successes / len(group),
                "first_attempt_successes": sum(
                    bool(item["success"]) and int(item.get("recovery_attempts", 0)) == 0 for item in group
                ),
            }
    attempted_recoveries = [item for item in completed if int(item.get("recovery_attempts", 0)) > 0]
    safe_rejections = [item for item in completed if item.get("safe_rejection") is True]
    motion_trials = [item for item in completed if item.get("safe_rejection") is not True]
    motion_successes = [item for item in motion_trials if item["success"]]
    semantic_detection_count = sum(len(item.get("semantic_labels", [])) for item in completed)
    policy_diagnostics = [
        item["policy_diagnostics"]
        for item in completed
        if isinstance(item.get("policy_diagnostics"), dict)
    ]
    failure_stages: dict[str, int] = {}
    for diagnostics in policy_diagnostics:
        for target in diagnostics.get("targets", {}).values():
            stage = str(target.get("failure_stage", "unknown"))
            failure_stages[stage] = failure_stages.get(stage, 0) + 1
    return {
        "schema_version": 1,
        "trials_planned": len(results),
        "trials_completed": len(completed),
        "infrastructure_errors": len(infrastructure_errors),
        "successes": len(successes),
        "success_rate": len(successes) / len(completed) if completed else None,
        "success_rate_ci95": wilson_interval(len(successes), len(completed)),
        "safe_rejections": len(safe_rejections),
        "safe_rejection_rate": len(safe_rejections) / len(completed) if completed else None,
        "motion_trials": len(motion_trials),
        "motion_successes": len(motion_successes),
        "motion_success_rate": len(motion_successes) / len(motion_trials) if motion_trials else None,
        "mean_duration_s": statistics.fmean(durations) if durations else None,
        "mean_success_duration_s": statistics.fmean(successful_durations) if successful_durations else None,
        "localization_samples": len(xy_errors),
        "detection_rate": len(xy_errors) / (len(completed) * len(COLORS)) if completed else None,
        "semantic_detection_rate": (
            semantic_detection_count / (len(completed) * len(COLORS)) if completed else None
        ),
        "mean_xy_error_mm": statistics.fmean(xy_errors) if xy_errors else None,
        "rmse_xy_error_mm": math.sqrt(statistics.fmean(value * value for value in xy_errors)) if xy_errors else None,
        "p95_xy_error_mm": percentile(xy_errors, 95),
        "max_xy_error_mm": max(xy_errors) if xy_errors else None,
        "mean_xyz_error_mm": statistics.fmean(xyz_errors) if xyz_errors else None,
        "p95_xyz_error_mm": percentile(xyz_errors, 95),
        "wrist_localization_samples": len(wrist_xy_errors),
        "mean_wrist_xy_error_mm": statistics.fmean(wrist_xy_errors) if wrist_xy_errors else None,
        "p95_wrist_xy_error_mm": percentile(wrist_xy_errors, 95),
        "max_wrist_xy_error_mm": max(wrist_xy_errors) if wrist_xy_errors else None,
        "visual_servo_targets": len(servo_rows),
        "visual_servo_updates": sum(int(row["updates"]) for row in servo_rows),
        "mean_visual_servo_initial_xy_error_mm": statistics.fmean(
            row["initial_xy_error_mm"] for row in servo_rows
        ) if servo_rows else None,
        "mean_visual_servo_final_xy_error_mm": statistics.fmean(
            row["final_xy_error_mm"] for row in servo_rows
        ) if servo_rows else None,
        "mean_visual_servo_xy_improvement_mm": statistics.fmean(
            row["xy_improvement_mm"] for row in servo_rows
        ) if servo_rows else None,
        "placements_verified": sum(int(item.get("placements_verified", 0)) for item in completed),
        "recovery_events": sum(int(item.get("recovery_events", 0)) for item in completed),
        "recovery_attempts": sum(int(item.get("recovery_attempts", 0)) for item in completed),
        "recovery_successes": sum(int(item.get("recovery_successes", 0)) for item in completed),
        "trials_with_recovery": len(attempted_recoveries),
        "conditional_recovery_success_rate": (
            sum(bool(item["success"]) for item in attempted_recoveries) / len(attempted_recoveries)
            if attempted_recoveries else None
        ),
        "fault_events": sum(int(item.get("fault_events", 0)) for item in completed),
        "wrist_refinements": sum(int(item.get("wrist_refinements", 0)) for item in completed),
        "wrist_rejections": sum(int(item.get("wrist_rejections", 0)) for item in completed),
        "policy_diagnostic_trials": len(policy_diagnostics),
        "mean_policy_clipped_command_rate": (
            statistics.fmean(float(item.get("clipped_command_rate", 0.0)) for item in policy_diagnostics)
            if policy_diagnostics else None
        ),
        "mean_policy_arm_step_clipped_command_rate": (
            statistics.fmean(
                float(item.get("arm_step_clipped_command_rate", 0.0))
                for item in policy_diagnostics
            ) if policy_diagnostics else None
        ),
        "policy_failure_stages": failure_stages,
        "by_target_count": by_target_count,
        "by_fault_type": by_fault_type,
    }


def write_results(output_dir: Path, results: list[dict[str, Any]]) -> None:
    (output_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    (output_dir / "summary.json").write_text(json.dumps(summarize(results), indent=2, ensure_ascii=False))
    columns = [
        "trial_index", "seed", "targets", "fault_type", "status", "outcome", "success", "return_code", "duration_s",
        "detected_colors", "mean_xy_error_mm", "max_xy_error_mm", "placements_verified",
        "semantic_labels", "feasible_labels", "safe_rejection", "geometry_rejections",
        "mean_wrist_xy_error_mm", "recovery_events", "recovery_attempts", "recovery_successes",
        "wrist_refinements", "wrist_rejections", "visual_servo_updates", "fault_events",
        "mean_visual_servo_final_xy_error_mm", "policy_failure_stage",
        "policy_clipped_command_rate", "policy_arm_step_clipped_command_rate",
        "adaptive_reobservations", "failure",
    ]
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for item in results:
            errors = [row["xy_error_mm"] for row in item["localization"]]
            wrist_errors = [row["xy_error_mm"] for row in item.get("wrist_localization", [])]
            policy_diagnostics = item.get("policy_diagnostics") or {}
            target_diagnostics = policy_diagnostics.get("targets", {})
            first_target_diagnostics = next(iter(target_diagnostics.values()), {})
            writer.writerow(
                {
                    "trial_index": item["trial_index"],
                    "seed": item["seed"],
                    "targets": " ".join(item["targets"]),
                    "fault_type": item.get("fault_type", "none"),
                    "status": item["status"],
                    "outcome": item.get("outcome", "success" if item["success"] else "failure"),
                    "success": item["success"],
                    "return_code": item["return_code"],
                    "duration_s": round(float(item["duration_s"]), 3),
                    "detected_colors": " ".join(row["color"] for row in item["localization"]),
                    "mean_xy_error_mm": round(statistics.fmean(errors), 3) if errors else "",
                    "max_xy_error_mm": round(max(errors), 3) if errors else "",
                    "placements_verified": item.get("placements_verified", 0),
                    "semantic_labels": " ".join(item.get("semantic_labels", [])),
                    "feasible_labels": " ".join(item.get("feasible_labels", [])),
                    "safe_rejection": item.get("safe_rejection", False),
                    "geometry_rejections": json.dumps(item.get("geometry_rejections", {}), ensure_ascii=False),
                    "mean_wrist_xy_error_mm": round(statistics.fmean(wrist_errors), 3) if wrist_errors else "",
                    "recovery_events": item.get("recovery_events", 0),
                    "recovery_attempts": item.get("recovery_attempts", 0),
                    "recovery_successes": item.get("recovery_successes", 0),
                    "wrist_refinements": item.get("wrist_refinements", 0),
                    "wrist_rejections": item.get("wrist_rejections", 0),
                    "visual_servo_updates": sum(row["updates"] for row in item.get("visual_servo", [])),
                    "fault_events": item.get("fault_events", 0),
                    "mean_visual_servo_final_xy_error_mm": round(
                        statistics.fmean(row["final_xy_error_mm"] for row in item.get("visual_servo", [])), 3
                    ) if item.get("visual_servo") else "",
                    "policy_failure_stage": first_target_diagnostics.get("failure_stage", ""),
                    "policy_clipped_command_rate": policy_diagnostics.get(
                        "clipped_command_rate", ""
                    ),
                    "policy_arm_step_clipped_command_rate": policy_diagnostics.get(
                        "arm_step_clipped_command_rate", ""
                    ),
                    "adaptive_reobservations": policy_diagnostics.get(
                        "adaptive_reobservation_count", ""
                    ),
                    "failure": item["failure"],
                }
            )


def run_trial(
    project_root: Path,
    python_sh: Path,
    trial_dir: Path,
    scenario: dict[str, Any],
    *,
    timeout_s: int,
    wrist_mode: str,
    disable_wrist_gate: bool,
    max_recovery_attempts: int,
    visual_servo_rate_hz: float,
    visual_servo_alpha: float,
    visual_servo_max_step_mm: float,
    perception_backend: str,
    semantic_backend: str,
    semantic_vocabulary: Path | None,
    semantic_model: str | None,
    semantic_min_confidence: float,
    object_set: str,
    record_dataset: Path | None,
    record_fps: float,
    record_instruction: str | None,
    vla_checkpoint: Path | None,
    act_control_hz: float,
    act_action_steps: int,
    act_adaptive_reobserve: bool,
    act_precision_action_steps: int,
    act_telemetry_images_every: int,
    act_max_arm_step: float,
    act_timeout: float,
) -> dict[str, Any]:
    trial_dir.mkdir(parents=True, exist_ok=True)
    scenario_path = trial_dir / "scenario.json"
    scenario_path.write_text(json.dumps(scenario, indent=2))
    image_path = trial_dir / "latest_overhead.png"
    command = [
        str(python_sh), str(project_root / "franka_color_sorting.py"),
        "--target", *scenario["targets"],
        "--scenario", str(scenario_path), "--headless", "--test",
        "--artifact-dir", str(trial_dir),
        "--max-recovery-attempts", str(max_recovery_attempts),
        "--visual-servo-rate-hz", str(visual_servo_rate_hz),
        "--visual-servo-alpha", str(visual_servo_alpha),
        "--visual-servo-max-step-mm", str(visual_servo_max_step_mm),
        "--wrist-mode", wrist_mode,
        "--perception-backend", perception_backend,
        "--semantic-backend", semantic_backend,
        "--semantic-min-confidence", str(semantic_min_confidence),
        "--object-set", object_set,
    ]
    if vla_checkpoint is None:
        command.extend(["--rgbd", "--rgbd-output", str(image_path)])
    if semantic_vocabulary is not None:
        command.extend(["--semantic-vocabulary", str(semantic_vocabulary)])
    if semantic_model:
        command.extend(["--semantic-model", semantic_model])
    if disable_wrist_gate:
        command.append("--disable-wrist-gate")
    if record_dataset is not None:
        instruction = record_instruction or (
            "Place " + ", ".join(scenario["targets"]) + " in the target box"
        )
        command.extend([
            "--record-dataset", str(record_dataset),
            "--record-fps", str(record_fps),
            "--record-instruction", instruction,
        ])
    if vla_checkpoint is not None:
        command.extend([
            "--vla-checkpoint", str(vla_checkpoint),
            "--act-control-hz", str(act_control_hz),
            "--act-action-steps", str(act_action_steps),
            "--act-precision-action-steps", str(act_precision_action_steps),
            "--act-telemetry-images-every", str(act_telemetry_images_every),
            "--act-max-arm-step", str(act_max_arm_step),
            "--act-timeout", str(act_timeout),
            # Pair policy sampling with the authored scenario seed so two
            # checkpoints see identical scenes and policy noise streams.
            "--policy-seed", str(scenario["seed"]),
        ])
        if act_adaptive_reobserve:
            command.append("--act-adaptive-reobserve")
    started = time.monotonic()
    timed_out = False
    try:
        completed = subprocess.run(
            command, cwd=project_root, text=True, capture_output=True, timeout=timeout_s, check=False
        )
        stdout, stderr, return_code = completed.stdout, completed.stderr, completed.returncode
    except subprocess.TimeoutExpired as error:
        timed_out = True
        stdout = error.stdout.decode(errors="replace") if isinstance(error.stdout, bytes) else (error.stdout or "")
        stderr = error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else (error.stderr or "")
        return_code = -1
    duration = time.monotonic() - started
    log = stdout + "\n" + stderr
    (trial_dir / "run.log").write_text(log, errors="replace")
    success = not timed_out and return_code == 0 and SUCCESS_MARKER in stdout
    infrastructure_error = None if success else classify_infrastructure_error(log)
    pose_path = trial_dir / "latest_vision_poses.json"
    errors = localization_errors(scenario, pose_path)
    wrist_errors = wrist_localization_errors(scenario, trial_dir)
    servo_metrics = visual_servo_metrics(scenario, trial_dir)
    perception = generic_perception_outcome(trial_dir, scenario["targets"])
    policy_summary_path = trial_dir / "policy_rollout_summary.json"
    policy_diagnostics = (
        json.loads(policy_summary_path.read_text(encoding="utf-8"))
        if policy_summary_path.is_file()
        else None
    )
    if infrastructure_error:
        failure = f"infrastructure_error:{infrastructure_error}"
    elif success:
        failure = ""
    elif timed_out:
        failure = f"timeout after {timeout_s}s"
    else:
        meaningful = [line.strip() for line in log.splitlines() if line.strip()]
        failure = meaningful[-1][-500:] if meaningful else f"process exited {return_code}"
    outcome = "success" if success else ("safe_rejection" if perception["safe_rejection"] else "failure")
    return {
        "trial_index": scenario["trial_index"],
        "seed": scenario["seed"],
        "targets": scenario["targets"],
        "fault_type": scenario.get("faults", {}).get("type", "none"),
        "status": "infrastructure_error" if infrastructure_error else "completed",
        "success": success,
        "outcome": outcome,
        "return_code": return_code,
        "duration_s": duration,
        "localization": errors,
        "wrist_localization": wrist_errors,
        "visual_servo": servo_metrics,
        **perception,
        "placements_verified": stdout.count("placement verified; marking complete"),
        "recovery_events": stdout.count("[recovery-required]"),
        "recovery_attempts": stdout.count("[recovery] attempt "),
        "recovery_successes": stdout.count("[recovery] succeeded after "),
        "wrist_refinements": stdout.count(" accepted:"),
        "wrist_rejections": stdout.count(" rejected:"),
        "fault_events": stdout.count("[fault-injection]"),
        "policy_diagnostics": policy_diagnostics,
        "failure": failure,
        "artifacts": str(trial_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trial-seed", type=int, help="Reproduce one exact recorded scenario seed (requires --trials 1).")
    parser.add_argument("--min-targets", type=int, default=1)
    parser.add_argument("--max-targets", type=int, default=2)
    parser.add_argument("--camera-jitter-m", type=float, default=0.03)
    parser.add_argument(
        "--fault-profile",
        choices=("none", "mixed", *FAULT_TYPES),
        default="none",
        help="Deterministic robustness fault injected into each recorded scenario.",
    )
    parser.add_argument("--light-min", type=float, default=600.0)
    parser.add_argument("--light-max", type=float, default=1400.0)
    parser.add_argument("--timeout", type=int, default=240, help="Per-trial timeout in seconds.")
    parser.add_argument("--python-sh", type=Path, default=Path.home() / "isaacsim" / "python.sh")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume an existing output directory after verifying its scenarios match exactly.",
    )
    parser.add_argument("--generate-only", action="store_true", help="Generate and validate scenarios without launching Isaac Sim.")
    parser.add_argument("--wrist-mode", choices=("off", "single", "servo"), help="Wrist feedback ablation mode (default: servo).")
    parser.add_argument("--disable-wrist-refinement", action="store_true", help="Deprecated alias for --wrist-mode off.")
    parser.add_argument("--disable-wrist-gate", action="store_true", help="Ablation only: use wrist estimates without the safety gate.")
    parser.add_argument("--visual-servo-rate-hz", type=float, default=10.0)
    parser.add_argument("--visual-servo-alpha", type=float, default=0.70)
    parser.add_argument("--visual-servo-max-step-mm", type=float, default=8.0)
    parser.add_argument("--max-recovery-attempts", type=int, choices=(0, 1), default=1)
    parser.add_argument("--perception-backend", choices=("color", "generic"), default="color")
    parser.add_argument("--semantic-backend", choices=("clip", "smolvlm", "prototype"), default="clip")
    parser.add_argument("--semantic-vocabulary", type=Path, help="JSON mapping scene IDs to visual descriptions.")
    parser.add_argument("--semantic-model", help="Optional semantic model override.")
    parser.add_argument("--semantic-min-confidence", type=float, default=0.50)
    parser.add_argument(
        "--object-set", choices=("cubes", "seen", "heldout"), default="cubes",
        help="Use cubes or the seen/held-out multi-shape evaluation split.",
    )
    parser.add_argument(
        "--target-policy", choices=("random", "cycle"), default="random",
        help="cycle evaluates red/blue/green/yellow once per four trials.",
    )
    parser.add_argument(
        "--record-dataset", type=Path,
        help="Record every randomized expert rollout into this raw VLA dataset.",
    )
    parser.add_argument("--record-fps", type=float, default=10.0)
    parser.add_argument(
        "--record-instruction",
        help="Language instruction stored in every episode; defaults to the sampled targets.",
    )
    parser.add_argument(
        "--vla-checkpoint", type=Path,
        help="Run a pure single-target SmolVLA policy instead of the expert controller.",
    )
    parser.add_argument("--act-control-hz", type=float, default=5.0)
    parser.add_argument("--act-action-steps", type=int, default=5)
    parser.add_argument("--act-adaptive-reobserve", action="store_true")
    parser.add_argument("--act-precision-action-steps", type=int, default=1)
    parser.add_argument("--act-telemetry-images-every", type=int, default=0)
    parser.add_argument("--act-max-arm-step", type=float, default=0.12)
    parser.add_argument("--act-timeout", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trials < 1:
        raise SystemExit("--trials must be at least 1")
    if args.resume and args.output is None:
        raise SystemExit("--resume requires an explicit --output directory")
    if args.resume and args.generate_only:
        raise SystemExit("--resume cannot be combined with --generate-only")
    if args.trial_seed is not None and args.trials != 1:
        raise SystemExit("--trial-seed requires --trials 1")
    if args.disable_wrist_refinement and args.wrist_mode not in (None, "off"):
        raise SystemExit("--disable-wrist-refinement cannot be combined with a non-off --wrist-mode")
    args.wrist_mode = "off" if args.disable_wrist_refinement else (args.wrist_mode or "servo")
    if args.wrist_mode == "off" and args.disable_wrist_gate:
        raise SystemExit("--disable-wrist-gate requires --wrist-mode single or servo")
    if args.visual_servo_rate_hz <= 0.0:
        raise SystemExit("--visual-servo-rate-hz must be positive")
    if not 0.0 < args.visual_servo_alpha <= 1.0:
        raise SystemExit("--visual-servo-alpha must be within (0, 1]")
    if args.visual_servo_max_step_mm <= 0.0:
        raise SystemExit("--visual-servo-max-step-mm must be positive")
    if not 0.0 <= args.semantic_min_confidence <= 1.0:
        raise SystemExit("--semantic-min-confidence must be within 0..1")
    if not 0.0 < args.record_fps <= 20.0:
        raise SystemExit("--record-fps must be within (0, 20]")
    if not 1 <= args.min_targets <= args.max_targets <= len(COLORS):
        raise SystemExit("target counts must satisfy 1 <= min <= max <= 4")
    if args.vla_checkpoint is not None and (args.min_targets != 1 or args.max_targets != 1):
        raise SystemExit("--vla-checkpoint currently requires --min-targets 1 --max-targets 1")
    if args.act_control_hz <= 0 or args.act_action_steps < 1 or args.act_max_arm_step <= 0 or args.act_timeout <= 0:
        raise SystemExit("VLA control rate, action steps, arm step, and timeout must be positive")
    if not 1 <= args.act_precision_action_steps <= args.act_action_steps:
        raise SystemExit("--act-precision-action-steps must be within --act-action-steps")
    if args.act_telemetry_images_every < 0:
        raise SystemExit("--act-telemetry-images-every must be non-negative")
    if not 0.0 <= args.camera_jitter_m <= 0.10:
        raise SystemExit("--camera-jitter-m must be within 0..0.10")
    if not 100.0 <= args.light_min <= args.light_max <= 5000.0:
        raise SystemExit("lighting bounds must satisfy 100 <= min <= max <= 5000")
    if args.vla_checkpoint is not None:
        args.vla_checkpoint = args.vla_checkpoint.expanduser().resolve()
        if not (args.vla_checkpoint / "model.safetensors").is_file():
            raise SystemExit(f"VLA checkpoint is incomplete: {args.vla_checkpoint}")
    project_root = Path(__file__).resolve().parent
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (args.output or project_root / "benchmark_results" / f"run_{timestamp}").resolve()
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    master = random.Random(args.seed)
    trial_seeds = [args.trial_seed] if args.trial_seed is not None else [master.randrange(2**32) for _ in range(args.trials)]
    scenarios = [
        generate_scenario(
            index,
            trial_seed,
            min_targets=args.min_targets,
            max_targets=args.max_targets,
            camera_jitter_m=args.camera_jitter_m,
            light_range=(args.light_min, args.light_max),
            fault_profile=args.fault_profile,
            object_set=args.object_set,
            target_policy=args.target_policy,
        )
        for index, trial_seed in enumerate(trial_seeds)
    ]
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "master_seed": args.seed,
        "platform": platform.platform(),
        "python": sys.version,
        "arguments": vars(args) | {
            "python_sh": str(args.python_sh),
            "output": str(output_dir),
            "semantic_vocabulary": (
                str(args.semantic_vocabulary) if args.semantic_vocabulary is not None else None
            ),
            "record_dataset": (
                str(args.record_dataset.resolve()) if args.record_dataset is not None else None
            ),
            "vla_checkpoint": str(args.vla_checkpoint) if args.vla_checkpoint is not None else None,
        },
        "scenarios": scenarios,
    }
    manifest_path = output_dir / "manifest.json"
    results_path = output_dir / "results.json"
    if args.resume:
        if not manifest_path.is_file() or not results_path.is_file():
            raise SystemExit("Resume requires existing manifest.json and results.json")
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("scenarios") != scenarios:
            raise SystemExit("Resume refused: generated scenarios do not match the existing manifest")
        results = json.loads(results_path.read_text(encoding="utf-8"))
        completed_indices = [int(item["trial_index"]) for item in results]
        if len(completed_indices) != len(set(completed_indices)):
            raise SystemExit("Resume refused: results contain duplicate trial indices")
        if any(index < 0 or index >= len(scenarios) for index in completed_indices):
            raise SystemExit("Resume refused: results contain an out-of-range trial index")
        print(f"[resume] keeping {len(results)}/{len(scenarios)} completed trial(s)", flush=True)
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        results = []
    if args.generate_only:
        for scenario in scenarios:
            trial_dir = output_dir / f"trial_{scenario['trial_index']:04d}"
            trial_dir.mkdir()
            (trial_dir / "scenario.json").write_text(json.dumps(scenario, indent=2))
            results.append(
                {
                    "trial_index": scenario["trial_index"], "seed": scenario["seed"],
                    "targets": scenario["targets"],
                    "fault_type": scenario.get("faults", {}).get("type", "none"),
                    "status": "generated", "success": False,
                    "return_code": 0, "duration_s": 0.0, "localization": [], "failure": "",
                    "artifacts": str(trial_dir),
                }
            )
        write_results(output_dir, results)
        print(f"Generated {len(results)} scenarios in {output_dir}")
        return
    if not args.python_sh.is_file():
        raise SystemExit(f"Isaac Sim launcher not found: {args.python_sh}")
    for scenario in scenarios:
        index = scenario["trial_index"]
        if any(int(item["trial_index"]) == index for item in results):
            continue
        print(f"[{index + 1}/{len(scenarios)}] seed={scenario['seed']} targets={scenario['targets']}", flush=True)
        result = run_trial(
            project_root, args.python_sh.resolve(), output_dir / f"trial_{index:04d}", scenario,
            timeout_s=args.timeout,
            wrist_mode=args.wrist_mode,
            disable_wrist_gate=args.disable_wrist_gate,
            max_recovery_attempts=args.max_recovery_attempts,
            visual_servo_rate_hz=args.visual_servo_rate_hz,
            visual_servo_alpha=args.visual_servo_alpha,
            visual_servo_max_step_mm=args.visual_servo_max_step_mm,
            perception_backend=args.perception_backend,
            semantic_backend=args.semantic_backend,
            semantic_vocabulary=args.semantic_vocabulary,
            semantic_model=args.semantic_model,
            semantic_min_confidence=args.semantic_min_confidence,
            object_set=args.object_set,
            record_dataset=(args.record_dataset.resolve() if args.record_dataset is not None else None),
            record_fps=args.record_fps,
            record_instruction=args.record_instruction,
            vla_checkpoint=args.vla_checkpoint,
            act_control_hz=args.act_control_hz,
            act_action_steps=args.act_action_steps,
            act_adaptive_reobserve=args.act_adaptive_reobserve,
            act_precision_action_steps=args.act_precision_action_steps,
            act_telemetry_images_every=args.act_telemetry_images_every,
            act_max_arm_step=args.act_max_arm_step,
            act_timeout=args.act_timeout,
        )
        results.append(result)
        write_results(output_dir, results)
        print(
            f"  status={result['status']} success={result['success']} duration={result['duration_s']:.1f}s "
            f"detections={len(result['localization'])}",
            flush=True,
        )
        if result["status"] == "infrastructure_error":
            summary = summarize(results)
            print(json.dumps(summary, indent=2))
            raise SystemExit(
                f"Benchmark stopped: {result['failure']}. Fix the runtime and start a fresh output directory."
            )
    summary = summarize(results)
    print(json.dumps(summary, indent=2))
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
