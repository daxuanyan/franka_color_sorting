"""Crash-tolerant raw episode storage for Isaac-to-VLA demonstrations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


SCHEMA_VERSION = 1


def next_episode_index(root: Path) -> int:
    episodes = root / "episodes"
    if not episodes.exists():
        return 0
    indices = []
    for path in episodes.glob("episode_*"):
        try:
            indices.append(int(path.name.removeprefix("episode_")))
        except ValueError:
            continue
    return max(indices, default=-1) + 1


class RawVLAEpisodeRecorder:
    """Write one self-contained episode without importing LeRobot in Isaac Python."""

    def __init__(
        self,
        root: Path,
        *,
        fps: float,
        instruction: str,
        targets: tuple[str, ...],
        object_set: str,
        joint_names: list[str],
        episode_index: int | None = None,
        dataset_kind: str = "expert_demonstration",
        metadata_extra: dict[str, Any] | None = None,
    ) -> None:
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("recording fps must be positive")
        if not instruction.strip():
            raise ValueError("recording instruction must not be empty")
        self.root = Path(root).expanduser().resolve()
        self.fps = float(fps)
        self.instruction = instruction.strip()
        self.episode_index = next_episode_index(self.root) if episode_index is None else episode_index
        if self.episode_index < 0:
            raise ValueError("episode_index must be non-negative")
        self.episode_dir = self.root / "episodes" / f"episode_{self.episode_index:06d}"
        if self.episode_dir.exists():
            raise FileExistsError(f"episode already exists: {self.episode_dir}")
        for relative in (
            "images/overhead", "images/wrist", "depth/overhead", "depth/wrist"
        ):
            (self.episode_dir / relative).mkdir(parents=True, exist_ok=False)
        self._frames_path = self.episode_dir / "frames.jsonl"
        self._frames = self._frames_path.open("x", encoding="utf-8")
        self._next_sample_time: float | None = None
        self._start_time: float | None = None
        self._last_time: float | None = None
        self._frame_count = 0
        self._finalized = False
        self._metadata: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "episode_index": self.episode_index,
            "status": "recording",
            "fps": self.fps,
            "instruction": self.instruction,
            "targets": list(targets),
            "object_set": object_set,
            "robot_type": "franka_panda",
            "joint_names": list(joint_names),
            "state_layout": {
                "joint_position": len(joint_names),
                "joint_velocity": len(joint_names),
                "end_effector_position": 3,
                "end_effector_quaternion_wxyz": 4,
                "gripper_width": 1,
            },
            "action_layout": {"joint_position_target": len(joint_names)},
            "dataset_kind": str(dataset_kind),
            "cameras": {},
        }
        if metadata_extra:
            protected = set(self._metadata)
            overlap = protected.intersection(metadata_extra)
            if overlap:
                raise ValueError(f"metadata_extra cannot replace protected keys: {sorted(overlap)}")
            self._metadata.update(metadata_extra)
        self._write_metadata()

    @property
    def finalized(self) -> bool:
        return self._finalized

    def due(self, time_s: float) -> bool:
        if self._finalized:
            return False
        return self._next_sample_time is None or time_s + 1e-9 >= self._next_sample_time

    @staticmethod
    def _rgb(value: np.ndarray, name: str) -> np.ndarray:
        image = np.asarray(value)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError(f"{name} RGB must have shape HxWx3+")
        return np.asarray(image[..., :3], dtype=np.uint8)

    @staticmethod
    def _depth_mm(value: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
        depth = np.asarray(value, dtype=float)
        if depth.shape != shape:
            raise ValueError(f"{name} depth shape {depth.shape} does not match RGB {shape}")
        valid = np.isfinite(depth) & (depth > 0)
        millimetres = np.zeros(depth.shape, dtype=np.uint16)
        millimetres[valid] = np.clip(np.rint(depth[valid] * 1000.0), 1, 65535).astype(np.uint16)
        return millimetres

    @staticmethod
    def _vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=float).reshape(-1)
        if vector.shape != (size,) or not np.isfinite(vector).all():
            raise ValueError(f"{name} must be a finite vector of length {size}")
        return vector

    def add_frame(
        self,
        *,
        time_s: float,
        overhead_rgb: np.ndarray,
        overhead_depth_m: np.ndarray,
        wrist_rgb: np.ndarray,
        wrist_depth_m: np.ndarray,
        joint_position: np.ndarray,
        joint_velocity: np.ndarray,
        end_effector_position: np.ndarray,
        end_effector_quaternion_wxyz: np.ndarray,
        gripper_width: float,
        joint_position_target: np.ndarray,
        eef_position_target: np.ndarray | None,
        gripper_open_target: bool | None,
        phase: str | None,
        object_id: str | None,
        object_positions: dict[str, np.ndarray],
        done: bool,
        student_action: np.ndarray | None = None,
        executed_action: np.ndarray | None = None,
        beta: float | None = None,
        expert_intervened: bool | None = None,
        intervention_reason: str | None = None,
        label_valid: bool | None = None,
    ) -> bool:
        """Record a frame when due; return False when rate limiting skips it."""
        if not self.due(time_s):
            return False
        if not np.isfinite(time_s):
            raise ValueError("time_s must be finite")
        overhead = self._rgb(overhead_rgb, "overhead")
        wrist = self._rgb(wrist_rgb, "wrist")
        overhead_depth = self._depth_mm(overhead_depth_m, overhead.shape[:2], "overhead")
        wrist_depth = self._depth_mm(wrist_depth_m, wrist.shape[:2], "wrist")
        joint_count = len(self._metadata["joint_names"])
        q = self._vector(joint_position, joint_count, "joint_position")
        dq = self._vector(joint_velocity, joint_count, "joint_velocity")
        eef_position = self._vector(end_effector_position, 3, "end_effector_position")
        eef_orientation = self._vector(
            end_effector_quaternion_wxyz, 4, "end_effector_quaternion_wxyz"
        )
        action = self._vector(joint_position_target, joint_count, "joint_position_target")
        student = None
        if student_action is not None:
            student = self._vector(student_action, joint_count, "student_action")
        executed = None
        if executed_action is not None:
            executed = self._vector(executed_action, joint_count, "executed_action")
        if beta is not None and (not np.isfinite(beta) or not 0.0 <= beta <= 1.0):
            raise ValueError("beta must be finite and within 0..1")
        if not np.isfinite(gripper_width):
            raise ValueError("gripper_width must be finite")
        target = None
        if eef_position_target is not None:
            target = self._vector(eef_position_target, 3, "eef_position_target").tolist()

        index = self._frame_count
        relative_paths = {
            "overhead_rgb": f"images/overhead/{index:06d}.png",
            "wrist_rgb": f"images/wrist/{index:06d}.png",
            "overhead_depth": f"depth/overhead/{index:06d}.png",
            "wrist_depth": f"depth/wrist/{index:06d}.png",
        }
        Image.fromarray(overhead).save(self.episode_dir / relative_paths["overhead_rgb"])
        Image.fromarray(wrist).save(self.episode_dir / relative_paths["wrist_rgb"])
        Image.fromarray(overhead_depth).save(
            self.episode_dir / relative_paths["overhead_depth"]
        )
        Image.fromarray(wrist_depth).save(
            self.episode_dir / relative_paths["wrist_depth"]
        )
        state = np.concatenate((q, dq, eef_position, eef_orientation, [float(gripper_width)]))
        record = {
            "frame_index": index,
            "timestamp": index / self.fps,
            "simulation_time": float(time_s),
            "task": self.instruction,
            "observation.state": state.tolist(),
            "observation.joint_position": q.tolist(),
            "observation.joint_velocity": dq.tolist(),
            "observation.end_effector_position": eef_position.tolist(),
            "observation.end_effector_quaternion_wxyz": eef_orientation.tolist(),
            "observation.gripper_width": float(gripper_width),
            "action": action.tolist(),
            "action.student": None if student is None else student.tolist(),
            "action.executed": None if executed is None else executed.tolist(),
            "dagger.beta": None if beta is None else float(beta),
            "dagger.expert_intervened": expert_intervened,
            "dagger.intervention_reason": intervention_reason,
            "dagger.label_valid": label_valid,
            "action.end_effector_position_target": target,
            "action.gripper_open_target": gripper_open_target,
            "phase": phase,
            "object_id": object_id,
            "ground_truth.object_positions": {
                name: self._vector(position, 3, f"object_positions[{name}]").tolist()
                for name, position in object_positions.items()
            },
            "done": bool(done),
            **relative_paths,
        }
        self._frames.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._frames.flush()
        self._metadata["cameras"] = {
            "overhead": {"rgb_shape": list(overhead.shape), "depth_unit": "millimetres"},
            "wrist": {"rgb_shape": list(wrist.shape), "depth_unit": "millimetres"},
        }
        self._start_time = float(time_s) if self._start_time is None else self._start_time
        self._last_time = float(time_s)
        self._frame_count += 1
        self._next_sample_time = float(time_s) + 1.0 / self.fps
        return True

    def finalize(self, *, success: bool, failure_reason: str | None = None) -> None:
        if self._finalized:
            return
        self._frames.close()
        self._metadata.update({
            "status": "complete" if success else "failed",
            "success": bool(success),
            "failure_reason": failure_reason,
            "num_frames": self._frame_count,
            "duration_s": (
                0.0 if self._start_time is None or self._last_time is None
                else self._last_time - self._start_time
            ),
        })
        self._write_metadata()
        self._finalized = True

    def _write_metadata(self) -> None:
        path = self.episode_dir / "episode.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self._metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)


def load_raw_episode(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    episode = Path(path)
    metadata = json.loads((episode / "episode.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported raw VLA schema in {episode}")
    frames = [
        json.loads(line)
        for line in (episode / "frames.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(frames) != metadata.get("num_frames"):
        raise ValueError(f"frame count mismatch in {episode}")
    return metadata, frames


def validate_raw_episode(path: Path) -> dict[str, Any]:
    """Deep-check a raw episode and return a compact machine-readable summary."""
    episode = Path(path)
    metadata, frames = load_raw_episode(episode)
    if metadata.get("status") not in {"complete", "failed"}:
        raise ValueError(f"episode is not finalized: {episode}")
    if not frames:
        raise ValueError(f"episode contains no frames: {episode}")

    joint_count = len(metadata.get("joint_names", []))
    expected_state_size = 2 * joint_count + 3 + 4 + 1
    previous_simulation_time = -np.inf
    for index, frame in enumerate(frames):
        if frame.get("frame_index") != index:
            raise ValueError(f"non-sequential frame_index at frame {index} in {episode}")
        expected_timestamp = index / float(metadata["fps"])
        if not np.isclose(frame.get("timestamp"), expected_timestamp, atol=1e-6):
            raise ValueError(f"invalid timestamp at frame {index} in {episode}")
        simulation_time = float(frame.get("simulation_time", np.nan))
        if not np.isfinite(simulation_time) or simulation_time < previous_simulation_time:
            raise ValueError(f"non-monotonic simulation time at frame {index} in {episode}")
        previous_simulation_time = simulation_time

        state = np.asarray(frame.get("observation.state"), dtype=float)
        action = np.asarray(frame.get("action"), dtype=float)
        if state.shape != (expected_state_size,) or not np.isfinite(state).all():
            raise ValueError(f"invalid state at frame {index} in {episode}")
        if action.shape != (joint_count,) or not np.isfinite(action).all():
            raise ValueError(f"invalid action at frame {index} in {episode}")

        for camera in ("overhead", "wrist"):
            rgb_path = episode / frame[f"{camera}_rgb"]
            depth_path = episode / frame[f"{camera}_depth"]
            if not rgb_path.is_file() or not depth_path.is_file():
                raise ValueError(f"missing {camera} image at frame {index} in {episode}")
            with Image.open(rgb_path) as image:
                rgb = np.asarray(image.convert("RGB"))
            with Image.open(depth_path) as image:
                depth = np.asarray(image)
            expected_shape = tuple(metadata["cameras"][camera]["rgb_shape"])
            if rgb.shape != expected_shape:
                raise ValueError(f"invalid {camera} RGB shape at frame {index} in {episode}")
            if depth.shape != expected_shape[:2] or depth.dtype != np.uint16:
                raise ValueError(f"invalid {camera} depth at frame {index} in {episode}")

    return {
        "episode_index": int(metadata["episode_index"]),
        "status": metadata["status"],
        "success": bool(metadata.get("success", False)),
        "num_frames": len(frames),
        "fps": float(metadata["fps"]),
        "state_dim": expected_state_size,
        "action_dim": joint_count,
        "instruction": metadata["instruction"],
    }
