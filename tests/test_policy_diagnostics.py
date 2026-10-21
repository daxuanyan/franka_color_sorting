from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from policy_diagnostics import PolicyRolloutTelemetry, infer_failure_stage


def test_failure_stage_classification():
    assert infer_failure_stage(
        placed=False, min_eef_distance_m=0.20, max_lift_m=0.0,
        max_displacement_m=0.0,
    ) == "approach"
    assert infer_failure_stage(
        placed=False, min_eef_distance_m=0.08, max_lift_m=0.0,
        max_displacement_m=0.0,
    ) == "contact_or_grasp"
    assert infer_failure_stage(
        placed=False, min_eef_distance_m=0.08, max_lift_m=0.04,
        max_displacement_m=0.05,
    ) == "transport_or_release"
    assert infer_failure_stage(
        placed=True, min_eef_distance_m=0.08, max_lift_m=0.04,
        max_displacement_m=0.50,
    ) == "completed"


def test_rollout_telemetry_writes_trace_and_summary(tmp_path):
    telemetry = PolicyRolloutTelemetry(tmp_path, {"red": np.array([0.5, -0.4, 0.02])})
    telemetry.update(
        np.array([0.5, -0.4, 0.10]),
        {"red": np.array([0.5, -0.4, 0.04])},
    )
    telemetry.record_command(
        time_s=1.0,
        goal="red",
        inference_count=1,
        end_effector_position=np.array([0.5, -0.4, 0.10]),
        gripper_width_m=0.04,
        diagnostics={"clipped": True, "selected_chunk_steps": 1},
    )
    runtime = SimpleNamespace(
        command_count=1,
        inference_count=1,
        clipped_command_count=1,
        arm_step_clipped_command_count=1,
        joint_limit_clipped_command_count=0,
        clipped_joint_value_count=2,
        max_filter_correction=0.2,
        adaptive_reobservation_count=1,
    )
    summary = telemetry.finalize(
        success=False, placed={"red": False}, runtime=runtime, elapsed_s=2.0
    )
    assert summary["clipped_command_rate"] == 1.0
    assert summary["arm_step_clipped_command_rate"] == 1.0
    assert summary["targets"]["red"]["max_lift_m"] == 0.02
    assert len((tmp_path / "policy_telemetry.jsonl").read_text().splitlines()) == 1
    assert json.loads((tmp_path / "policy_rollout_summary.json").read_text())["success"] is False
