#!/usr/bin/env python3
"""Measure whether a language-conditioned policy uses its task instruction.

The same held-out observations and the same flow-matching noise are evaluated
twice: once with the recorded instruction and once with a deliberately wrong
target instruction.  A useful language-conditioned policy should produce
different actions and normally incur higher imitation error under the wrong
instruction.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from evaluate_act_checkpoints import ErrorAccumulator, atomic_write_json, load_split_indices, make_dataset


COLORS = ("red", "green", "blue", "yellow")


def counterfactual_instruction(task: str) -> str:
    """Replace the task's color with the next color, guaranteeing a wrong goal."""
    lowered = task.lower()
    matches = [color for color in COLORS if color in lowered]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one color in task, got {task!r}")
    source = matches[0]
    target = COLORS[(COLORS.index(source) + 1) % len(COLORS)]
    start = lowered.index(source)
    return task[:start] + target + task[start + len(source) :]


def clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Copy a loader batch because LeRobot preprocessors may mutate mappings."""
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else list(value) if isinstance(value, list) else value
        for key, value in batch.items()
    }


def evaluate_language_ablation(
    checkpoint: Path,
    dataset,
    *,
    device: str,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> dict[str, Any]:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    started = time.perf_counter()
    config = PreTrainedConfig.from_pretrained(checkpoint)
    policy = get_policy_class(config.type).from_pretrained(checkpoint, config=config)
    policy.to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    action_feature = dataset.meta.features["action"]
    action_dim = int(action_feature["shape"][0])
    action_names = action_feature.get("names") or [f"action_{i}" for i in range(action_dim)]
    horizons = sorted({1, int(config.n_action_steps), int(config.chunk_size)})
    correct_errors = {h: ErrorAccumulator(action_dim) for h in horizons}
    wrong_errors = {h: ErrorAccumulator(action_dim) for h in horizons}
    prediction_differences = {h: ErrorAccumulator(action_dim) for h in horizons}
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=num_workers > 0,
    )

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    frame_count = 0
    task_pairs: set[tuple[str, str]] = set()
    with torch.inference_mode():
        for raw_batch in loader:
            target = raw_batch["action"].clone()
            valid = ~raw_batch["action_is_pad"].bool()

            correct_batch = clone_batch(raw_batch)
            wrong_batch = clone_batch(raw_batch)
            wrong_tasks = [counterfactual_instruction(task) for task in raw_batch["task"]]
            wrong_batch["task"] = wrong_tasks
            task_pairs.update(zip(raw_batch["task"], wrong_tasks))

            # Save the generator state immediately before sampling. Restoring it
            # gives both conditions exactly the same initial flow-matching noise.
            correct_processed = preprocessor(correct_batch)
            cpu_rng = torch.random.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            correct_prediction = postprocessor(policy.predict_action_chunk(correct_processed)).cpu()

            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            wrong_processed = preprocessor(wrong_batch)
            wrong_prediction = postprocessor(policy.predict_action_chunk(wrong_processed)).cpu()

            for horizon in horizons:
                correct_errors[horizon].update(
                    correct_prediction[:, :horizon], target[:, :horizon], valid[:, :horizon]
                )
                wrong_errors[horizon].update(
                    wrong_prediction[:, :horizon], target[:, :horizon], valid[:, :horizon]
                )
                prediction_differences[horizon].update(
                    wrong_prediction[:, :horizon],
                    correct_prediction[:, :horizon],
                    valid[:, :horizon],
                )
            frame_count += target.shape[0]

    def metric_name(horizon: int) -> str:
        return "first_step" if horizon == 1 else f"horizon_{horizon}"

    metrics: dict[str, Any] = {}
    for horizon in horizons:
        correct = correct_errors[horizon].result(action_names)
        wrong = wrong_errors[horizon].result(action_names)
        difference = prediction_differences[horizon].result(action_names)
        metrics[metric_name(horizon)] = {
            "correct_instruction": correct,
            "wrong_instruction": wrong,
            "wrong_vs_correct_mae_change": wrong["mae"] - correct["mae"],
            "wrong_vs_correct_relative_change_percent": 100.0 * (wrong["mae"] / correct["mae"] - 1.0),
            "prediction_divergence": difference,
        }

    result = {
        "schema_version": 1,
        "checkpoint": str(checkpoint.resolve()),
        "policy_type": config.type,
        "seed": seed,
        "episodes": len(dataset.episodes),
        "frames": frame_count,
        "counterfactual": "cyclic target replacement: red->green->blue->yellow->red",
        "paired_noise": True,
        "task_pairs": [list(pair) for pair in sorted(task_pairs)],
        "seconds": time.perf_counter() - started,
        "metrics": metrics,
    }
    del policy, preprocessor, postprocessor, loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("vla_datasets/heldout_300_language_vla_256")
    )
    parser.add_argument("--repo-id", default="local/franka-heldout-300-language-vla-256")
    parser.add_argument("--splits", type=Path, default=Path("vla_datasets/heldout_300_language_splits.json"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/smolvla_language_full_30k/checkpoints/030000/pretrained_model"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/smolvla_language_full_30k/language_ablation.json")
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20261018)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads((args.checkpoint / "config.json").read_text(encoding="utf-8"))
    episodes = load_split_indices(args.splits, args.split)
    dataset = make_dataset(args.dataset_root, args.repo_id, episodes, int(config["chunk_size"]))
    print(
        f"[language-ablation] split={args.split} episodes={len(episodes)} checkpoint={args.checkpoint}",
        flush=True,
    )
    result = evaluate_language_ablation(
        args.checkpoint,
        dataset,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    atomic_write_json(args.output, result)
    for name, metric in result["metrics"].items():
        print(
            f"  {name}: correct={metric['correct_instruction']['mae']:.6f} "
            f"wrong={metric['wrong_instruction']['mae']:.6f} "
            f"change={metric['wrong_vs_correct_relative_change_percent']:+.2f}% "
            f"prediction_delta={metric['prediction_divergence']['mae']:.6f}",
            flush=True,
        )
    print(f"[complete] wrote {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
