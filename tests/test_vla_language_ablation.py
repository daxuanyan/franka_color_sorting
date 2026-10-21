import pytest

from evaluate_vla_language_ablation import counterfactual_instruction


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "Place the red irregular object in the target box",
            "Place the green irregular object in the target box",
        ),
        (
            "Place the green irregular object in the target box",
            "Place the blue irregular object in the target box",
        ),
        (
            "Place the blue irregular object in the target box",
            "Place the yellow irregular object in the target box",
        ),
        (
            "Place the yellow irregular object in the target box",
            "Place the red irregular object in the target box",
        ),
    ],
)
def test_counterfactual_instruction_rotates_to_a_different_goal(source, expected):
    assert counterfactual_instruction(source) == expected


def test_counterfactual_instruction_rejects_ambiguous_task():
    with pytest.raises(ValueError, match="exactly one color"):
        counterfactual_instruction("Place the red or blue object")
