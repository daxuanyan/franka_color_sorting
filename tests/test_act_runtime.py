import numpy as np
import pytest

from act_runtime import (
    PANDA_LOWER_LIMITS,
    PANDA_UPPER_LIMITS,
    adaptive_action_count,
    build_robot_state,
    decode_rgb_png,
    encode_rgb_png,
    language_goal_instruction,
    letterbox_rgb,
    safety_filter_action,
    safety_filter_action_with_diagnostics,
)


def test_letterbox_rgb_matches_requested_shape_and_black_padding():
    source = np.full((20, 40, 3), 255, dtype=np.uint8)
    result = letterbox_rgb(source, 32, 32)
    assert result.shape == (32, 32, 3)
    assert np.all(result[0] == 0)
    assert np.all(result[16] == 255)


def test_build_robot_state_layout():
    state = build_robot_state(
        np.arange(9), np.arange(9) + 10, [20, 21, 22], [1, 0, 0, 0], 0.05
    )
    assert state.shape == (26,)
    assert state[:9].tolist() == list(range(9))
    assert state[-1] == pytest.approx(0.05)


def test_safety_filter_clips_limits_and_arm_delta():
    current = np.zeros(9, dtype=np.float32)
    current[3] = -0.5
    requested = np.full(9, 100.0, dtype=np.float32)
    filtered = safety_filter_action(requested, current, max_arm_step=0.2)
    assert np.all(filtered <= PANDA_UPPER_LIMITS)
    assert np.all(filtered >= PANDA_LOWER_LIMITS)
    assert np.max(np.abs(filtered[:7] - current[:7])) <= 0.200001
    assert filtered[7] == pytest.approx(0.04)
    assert filtered[8] == pytest.approx(0.04)


def test_safety_filter_rejects_non_finite_output():
    with pytest.raises(ValueError, match="finite"):
        safety_filter_action(np.full(9, np.nan), np.zeros(9))


def test_safety_filter_reports_policy_corrections():
    current = np.zeros(9, dtype=np.float32)
    current[3] = -0.5
    requested = current.copy()
    requested[0] = 0.5
    filtered, report = safety_filter_action_with_diagnostics(
        requested, current, max_arm_step=0.1
    )
    assert filtered[0] == pytest.approx(0.1)
    assert report["clipped"] is True
    assert report["arm_step_clips"] == 1
    assert report["max_abs_correction"] == pytest.approx(0.4)


def test_adaptive_action_count_shortens_risky_chunks():
    current = np.zeros(9, dtype=np.float32)
    current[3] = -0.5
    smooth = np.repeat(current[None, :], 5, axis=0)
    smooth[:, 0] = np.linspace(0.01, 0.05, 5)
    count, reasons = adaptive_action_count(smooth, current, max_arm_step=0.12)
    assert count == 5
    assert reasons == ()

    risky = smooth.copy()
    risky[0, 0] = 0.10
    count, reasons = adaptive_action_count(risky, current, max_arm_step=0.12)
    assert count == 1
    assert "large_arm_step" in reasons

    gripper = smooth.copy()
    gripper[0, 7:] = 0.02
    count, reasons = adaptive_action_count(gripper, current, max_arm_step=0.12)
    assert count == 1
    assert "gripper_transition" in reasons


def test_png_ipc_round_trip():
    image = np.arange(12 * 8 * 3, dtype=np.uint8).reshape(12, 8, 3)
    assert np.array_equal(decode_rgb_png(encode_rgb_png(image)), image)


def test_language_goal_instruction_matches_training_template():
    assert language_goal_instruction("green") == (
        "Place the green irregular object in the target box"
    )
    with pytest.raises(ValueError, match="Unknown"):
        language_goal_instruction("purple")
    adaptive_action_count,
