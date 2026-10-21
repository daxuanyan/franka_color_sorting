from export_dagger_lerobot import valid_dagger_frames


def test_valid_dagger_frames_excludes_post_teacher_hold_frames():
    frames = [
        {"phase": "APPROACH", "action.student": [0], "action.executed": [0]},
        {"phase": None, "action.student": [0], "action.executed": [0]},
        {
            "phase": None,
            "dagger.label_valid": True,
            "action.student": [0],
            "action.executed": [0],
        },
        {"phase": "GRASP", "action.student": None, "action.executed": [0]},
    ]
    assert valid_dagger_frames(frames) == [frames[0], frames[2]]
