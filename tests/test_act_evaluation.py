import json

import pytest
import torch

from evaluate_act_checkpoints import ErrorAccumulator, discover_checkpoints, load_split_indices


def test_error_accumulator_masks_padding_and_reports_per_dimension():
    target = torch.zeros((1, 3, 2))
    prediction = torch.tensor([[[1.0, -1.0], [3.0, 4.0], [100.0, 100.0]]])
    valid = torch.tensor([[True, True, False]])
    accumulator = ErrorAccumulator(action_dim=2)
    accumulator.update(prediction, target, valid)

    result = accumulator.result(["a", "b"])
    assert result["mae"] == pytest.approx(2.25)
    assert result["rmse"] == pytest.approx((27.0 / 4.0) ** 0.5)
    assert result["valid_action_steps"] == 2
    assert result["per_dimension"]["a"]["mae"] == pytest.approx(2.0)
    assert result["per_dimension"]["b"]["mae"] == pytest.approx(2.5)


def test_load_split_indices(tmp_path):
    path = tmp_path / "splits.json"
    path.write_text(json.dumps({"splits": {"validation": {"lerobot_episode_indices": [3, 7]}}}))
    assert load_split_indices(path, "validation") == [3, 7]


def test_discover_checkpoints_ignores_incomplete_directories(tmp_path):
    for name in ("010000", "005000"):
        model = tmp_path / name / "pretrained_model"
        model.mkdir(parents=True)
        (model / "model.safetensors").touch()
    (tmp_path / "015000").mkdir()
    (tmp_path / "last").symlink_to("010000")

    assert [path.parent.name for path in discover_checkpoints(tmp_path)] == ["005000", "010000"]
