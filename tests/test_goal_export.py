import numpy as np
import pytest

from export_goal_lerobot import COLORS, conditioned_state, goal_state, object_segments


def test_goal_state_appends_documented_one_hot_order():
    state = goal_state(np.arange(26), "blue")
    assert state.shape == (30,)
    assert state[-4:].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert COLORS == ("red", "green", "blue", "yellow")


def test_object_segments_drop_unlabelled_and_split_on_goal_change():
    frames = [
        {"frame_index": 0, "object_id": None},
        {"frame_index": 1, "object_id": "red"},
        {"frame_index": 2, "object_id": "red"},
        {"frame_index": 3, "object_id": "blue"},
    ]
    segments = object_segments(frames)
    assert [(label, [f["frame_index"] for f in rows]) for label, rows in segments] == [
        ("red", [1, 2]),
        ("blue", [3]),
    ]


def test_goal_state_rejects_unknown_goal():
    with pytest.raises(ValueError, match="Unknown goal"):
        goal_state(np.zeros(26), "purple")


def test_language_conditioning_keeps_26d_state_without_target_leakage():
    source = np.arange(26, dtype=np.float32)
    red = conditioned_state(source, "red", "language")
    blue = conditioned_state(source, "blue", "language")
    assert red.shape == (26,)
    assert np.array_equal(red, source)
    assert np.array_equal(blue, source)


def test_conditioning_rejects_unknown_mode():
    with pytest.raises(ValueError, match="conditioning mode"):
        conditioned_state(np.zeros(26), "red", "textish")
