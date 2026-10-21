#!/usr/bin/env python3
"""Evaluate action-policy checkpoints on held-out LeRobot episodes.

The script selects a checkpoint using validation data and can then evaluate the
selected checkpoint once on the test split.  Predictions and targets are
compared in the original (unnormalized) robot action space.  ACT and generative
VLA policies are loaded from the policy type stored in each checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


@dataclass
class ErrorAccumulator:
    """Accumulate masked action errors without retaining predictions."""

    action_dim: int
    abs_sum: float = 0.0
    sq_sum: float = 0.0
    value_count: int = 0
    per_dim_abs_sum: list[float] = field(default_factory=list)
    per_dim_sq_sum: list[float] = field(default_factory=list)
    per_dim_count: int = 0

    def __post_init__(self) -> None:
        if not self.per_dim_abs_sum:
            self.per_dim_abs_sum = [0.0] * self.action_dim
        if not self.per_dim_sq_sum:
            self.per_dim_sq_sum = [0.0] * self.action_dim

    def update(self, prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> None:
        if prediction.shape != target.shape:
            raise ValueError(f"Prediction/target shape mismatch: {prediction.shape} != {target.shape}")
        if prediction.ndim != 3 or prediction.shape[-1] != self.action_dim:
            raise ValueError(f"Expected [batch, time, {self.action_dim}], got {prediction.shape}")
        if valid.shape != prediction.shape[:2]:
            raise ValueError(f"Mask shape mismatch: {valid.shape} != {prediction.shape[:2]}")

        error = (prediction.float() - target.float()).cpu()
        valid_cpu = valid.bool().cpu()
        selected = error[valid_cpu]
        if selected.numel() == 0:
            return
        absolute = selected.abs()
        squared = selected.square()
        self.abs_sum += absolute.sum().item()
        self.sq_sum += squared.sum().item()
        self.value_count += selected.numel()
        self.per_dim_count += selected.shape[0]
        dim_abs = absolute.sum(dim=0).tolist()
        dim_sq = squared.sum(dim=0).tolist()
        self.per_dim_abs_sum = [a + b for a, b in zip(self.per_dim_abs_sum, dim_abs)]
        self.per_dim_sq_sum = [a + b for a, b in zip(self.per_dim_sq_sum, dim_sq)]

    def result(self, action_names: list[str]) -> dict[str, Any]:
        if self.value_count == 0 or self.per_dim_count == 0:
            raise RuntimeError("No valid action values were evaluated")
        return {
            "mae": self.abs_sum / self.value_count,
            "rmse": math.sqrt(self.sq_sum / self.value_count),
            "valid_action_steps": self.per_dim_count,
            "valid_values": self.value_count,
            "per_dimension": {
                name: {
                    "mae": self.per_dim_abs_sum[i] / self.per_dim_count,
                    "rmse": math.sqrt(self.per_dim_sq_sum[i] / self.per_dim_count),
                }
                for i, name in enumerate(action_names)
            },
        }


def load_split_indices(path: Path, split: str) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        indices = payload["splits"][split]["lerobot_episode_indices"]
    except KeyError as exc:
        raise ValueError(f"Split {split!r} is missing from {path}") from exc
    if not indices:
        raise ValueError(f"Split {split!r} contains no episodes")
    return [int(index) for index in indices]


def discover_checkpoints(root: Path) -> list[Path]:
    checkpoints = []
    for step_dir in root.glob("[0-9]*"):
        model_dir = step_dir / "pretrained_model"
        if step_dir.is_dir() and step_dir.name.isdigit() and (model_dir / "model.safetensors").is_file():
            checkpoints.append(model_dir)
    checkpoints.sort(key=lambda path: int(path.parent.name))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found below {root}")
    return checkpoints


def make_dataset(dataset_root: Path, repo_id: str, episodes: list[int], chunk_size: int):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # ACT predicts actions at the current frame and the next chunk_size - 1 frames.
    fps = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))["fps"]
    return LeRobotDataset(
        repo_id,
        root=dataset_root,
        episodes=episodes,
        delta_timestamps={"action": [index / fps for index in range(chunk_size)]},
    )


def evaluate_checkpoint(
    checkpoint: Path,
    dataset,
    *,
    device: str,
    batch_size: int,
    num_workers: int,
    seed: int = 20261018,
) -> dict[str, Any]:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    started = time.perf_counter()
    policy_config = PreTrainedConfig.from_pretrained(checkpoint)
    policy_class = get_policy_class(policy_config.type)
    policy = policy_class.from_pretrained(checkpoint, config=policy_config)
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    action_feature = dataset.meta.features["action"]
    action_dim = int(action_feature["shape"][0])
    action_names = action_feature.get("names") or [f"action_{index}" for index in range(action_dim)]
    horizons = sorted({1, int(policy.config.n_action_steps), int(policy.config.chunk_size)})
    accumulators = {horizon: ErrorAccumulator(action_dim) for horizon in horizons}
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=num_workers > 0,
    )

    frame_count = 0
    # Flow-matching VLA policies sample an initial noise tensor. Resetting the
    # generator for every checkpoint gives each model the same evaluation
    # noise stream and makes validation ranking reproducible.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        for batch in loader:
            target = batch["action"].clone()
            valid = ~batch["action_is_pad"].bool()
            processed = preprocessor(batch)
            prediction = policy.predict_action_chunk(processed)
            prediction = postprocessor(prediction)
            prediction = prediction.cpu()
            for horizon, accumulator in accumulators.items():
                accumulator.update(
                    prediction[:, :horizon],
                    target[:, :horizon],
                    valid[:, :horizon],
                )
            frame_count += target.shape[0]

    step = int(checkpoint.parent.name)
    metrics = {
        "step": step,
        "policy_type": policy.config.type,
        "seed": seed,
        "checkpoint": str(checkpoint.resolve()),
        "episodes": len(dataset.episodes),
        "frames": frame_count,
        "seconds": time.perf_counter() - started,
        "metrics": {
            ("first_step" if horizon == 1 else f"horizon_{horizon}"): accumulator.result(action_names)
            for horizon, accumulator in accumulators.items()
        },
    }
    del policy, preprocessor, postprocessor, loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("vla_datasets/heldout_300_act_256"))
    parser.add_argument("--repo-id", default="local/franka-heldout-300-act-256")
    parser.add_argument("--splits", type=Path, default=Path("vla_datasets/heldout_300_splits.json"))
    parser.add_argument(
        "--checkpoint-root", type=Path, default=Path("outputs/act_full_145_50k/checkpoints")
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/act_full_145_50k/evaluation.json"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--selection-horizon", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--skip-test", action="store_true", help="Only rank checkpoints on validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_paths = discover_checkpoints(args.checkpoint_root)
    # All checkpoints from a run share the same ACT temporal configuration.
    config = json.loads((checkpoint_paths[0] / "config.json").read_text(encoding="utf-8"))
    chunk_size = int(config["chunk_size"])
    selection_key = "first_step" if args.selection_horizon == 1 else f"horizon_{args.selection_horizon}"
    if args.selection_horizon not in {1, int(config["n_action_steps"]), chunk_size}:
        raise ValueError(
            "selection-horizon must be 1, n_action_steps "
            f"({config['n_action_steps']}), or chunk_size ({chunk_size})"
        )

    validation_episodes = load_split_indices(args.splits, "validation")
    validation_dataset = make_dataset(args.dataset_root, args.repo_id, validation_episodes, chunk_size)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "dataset_root": str(args.dataset_root.resolve()),
        "split_file": str(args.splits.resolve()),
        "selection_metric": f"validation.{selection_key}.mae",
        "validation": [],
        "best_checkpoint": None,
        "test": None,
    }

    for position, checkpoint in enumerate(checkpoint_paths, start=1):
        print(f"[validation {position}/{len(checkpoint_paths)}] {checkpoint.parent.name}", flush=True)
        result = evaluate_checkpoint(
            checkpoint,
            validation_dataset,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
        )
        payload["validation"].append(result)
        score = result["metrics"][selection_key]["mae"]
        print(f"  {selection_key} MAE={score:.6f} ({result['seconds']:.1f}s)", flush=True)
        atomic_write_json(args.output, payload)

    best = min(payload["validation"], key=lambda item: item["metrics"][selection_key]["mae"])
    payload["best_checkpoint"] = {
        "step": best["step"],
        "checkpoint": best["checkpoint"],
        "validation_score": best["metrics"][selection_key]["mae"],
    }
    print(
        f"[selected] step={best['step']} validation {selection_key} MAE="
        f"{best['metrics'][selection_key]['mae']:.6f}",
        flush=True,
    )

    if not args.skip_test:
        test_episodes = load_split_indices(args.splits, "test")
        test_dataset = make_dataset(args.dataset_root, args.repo_id, test_episodes, chunk_size)
        print(f"[test] evaluating selected checkpoint {best['step']}", flush=True)
        payload["test"] = evaluate_checkpoint(
            Path(best["checkpoint"]),
            test_dataset,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
        )
        print(
            f"  {selection_key} MAE={payload['test']['metrics'][selection_key]['mae']:.6f}",
            flush=True,
        )

    atomic_write_json(args.output, payload)
    print(f"[complete] wrote {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
