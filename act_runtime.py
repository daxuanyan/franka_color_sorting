"""Runtime adapter for executing trained LeRobot action policies in Isaac Sim."""

from __future__ import annotations

import atexit
import base64
from collections import deque
from io import BytesIO
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import uuid

import numpy as np
from PIL import Image, ImageOps


PANDA_JOINT_NAMES = (
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
    "panda_finger_joint1",
    "panda_finger_joint2",
)
ACT_GOAL_COLORS = ("red", "green", "blue", "yellow")
LANGUAGE_POLICY_TYPES = frozenset({"smolvla"})
PANDA_LOWER_LIMITS = np.array(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0, 0.0],
    dtype=np.float32,
)
PANDA_UPPER_LIMITS = np.array(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04, 0.04],
    dtype=np.float32,
)


def letterbox_rgb(rgb: np.ndarray, height: int = 256, width: int = 256) -> np.ndarray:
    """Match the aspect-preserving preprocessing used during dataset export."""
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError(f"RGB image must be HxWx3+, got {array.shape}")
    image = Image.fromarray(np.asarray(array[..., :3], dtype=np.uint8), mode="RGB")
    image = ImageOps.pad(
        image,
        (width, height),
        method=Image.Resampling.BILINEAR,
        color=(0, 0, 0),
        centering=(0.5, 0.5),
    )
    return np.asarray(image)


def build_robot_state(
    joint_position: np.ndarray,
    joint_velocity: np.ndarray,
    end_effector_position: np.ndarray,
    end_effector_quaternion_wxyz: np.ndarray,
    gripper_width: float,
) -> np.ndarray:
    state = np.concatenate(
        (
            np.asarray(joint_position, dtype=np.float32).reshape(-1),
            np.asarray(joint_velocity, dtype=np.float32).reshape(-1),
            np.asarray(end_effector_position, dtype=np.float32).reshape(-1),
            np.asarray(end_effector_quaternion_wxyz, dtype=np.float32).reshape(-1),
            np.asarray([gripper_width], dtype=np.float32),
        )
    )
    if state.shape != (26,) or not np.isfinite(state).all():
        raise ValueError(f"ACT state must contain 26 finite values, got {state.shape}")
    return state


def language_goal_instruction(color: str) -> str:
    """Return the exact single-object instruction used for SmolVLA training."""
    if color not in ACT_GOAL_COLORS:
        raise ValueError(f"Unknown language-policy goal: {color!r}")
    return f"Place the {color} irregular object in the target box"


def safety_filter_action(
    action: np.ndarray,
    current_joint_position: np.ndarray,
    *,
    max_arm_step: float = 0.35,
) -> np.ndarray:
    """Reject invalid output and bound absolute joint targets and arm motion."""
    filtered, _ = safety_filter_action_with_diagnostics(
        action, current_joint_position, max_arm_step=max_arm_step
    )
    return filtered


def safety_filter_action_with_diagnostics(
    action: np.ndarray,
    current_joint_position: np.ndarray,
    *,
    max_arm_step: float = 0.35,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    """Apply the safety filter and report how much it changed the policy output."""
    target = np.asarray(action, dtype=np.float32).reshape(-1)
    current = np.asarray(current_joint_position, dtype=np.float32).reshape(-1)
    if target.shape != (9,) or current.shape != (9,):
        raise ValueError("ACT action and current joint position must both have length 9")
    if not np.isfinite(target).all() or not np.isfinite(current).all():
        raise ValueError("ACT action and current joint position must be finite")
    if max_arm_step <= 0:
        raise ValueError("max_arm_step must be positive")
    bounded = np.clip(target, PANDA_LOWER_LIMITS, PANDA_UPPER_LIMITS)
    joint_limit_clips = int(np.count_nonzero(np.abs(bounded - target) > 1e-6))
    filtered = bounded.copy()
    requested_arm_delta = filtered[:7] - current[:7]
    arm_delta = np.clip(filtered[:7] - current[:7], -max_arm_step, max_arm_step)
    arm_step_clips = int(np.count_nonzero(np.abs(arm_delta - requested_arm_delta) > 1e-6))
    filtered[:7] = current[:7] + arm_delta
    correction = np.abs(filtered - target)
    return filtered, {
        "clipped": bool(joint_limit_clips or arm_step_clips),
        "joint_limit_clips": joint_limit_clips,
        "arm_step_clips": arm_step_clips,
        "max_abs_correction": float(np.max(correction)),
    }


def adaptive_action_count(
    predicted: np.ndarray,
    current_joint_position: np.ndarray,
    *,
    max_arm_step: float,
    precision_action_steps: int = 1,
    arm_step_ratio: float = 0.75,
    gripper_step: float = 0.004,
) -> tuple[int, tuple[str, ...]]:
    """Choose a short execution horizon for risky arm or gripper transitions.

    The decision uses only the robot state and the predicted action chunk. It
    therefore does not leak simulator object poses into policy execution.
    """
    actions = np.asarray(predicted, dtype=np.float32)
    current = np.asarray(current_joint_position, dtype=np.float32).reshape(-1)
    if actions.ndim != 2 or actions.shape[1] != 9 or not len(actions):
        raise ValueError("predicted actions must have shape Nx9")
    if current.shape != (9,) or not np.isfinite(actions).all() or not np.isfinite(current).all():
        raise ValueError("adaptive action inputs must be finite with a 9D current state")
    if max_arm_step <= 0 or not 1 <= precision_action_steps <= len(actions):
        raise ValueError("invalid adaptive action-step configuration")
    if not 0.0 < arm_step_ratio <= 1.0 or gripper_step <= 0:
        raise ValueError("adaptive thresholds must be positive")

    previous = np.vstack((current, actions[:-1]))
    reasons: list[str] = []
    if np.any(np.abs(actions[:, :7] - previous[:, :7]) >= max_arm_step * arm_step_ratio):
        reasons.append("large_arm_step")
    if np.any(np.abs(actions[:, 7:] - previous[:, 7:]) >= gripper_step):
        reasons.append("gripper_transition")
    selected = precision_action_steps if reasons else len(actions)
    return selected, tuple(reasons)


class ACTRuntimePolicy:
    """Load a LeRobot action policy and periodically refresh its action chunk.

    The historical class name is retained for compatibility. Checkpoints are
    now loaded from their saved policy type, including ACT and SmolVLA.
    """

    def __init__(
        self,
        checkpoint: Path,
        *,
        device: str = "cuda",
        control_hz: float = 5.0,
        action_steps: int | None = None,
        max_arm_step: float = 0.35,
        inference_seed: int = 20261018,
        adaptive_reobserve: bool = False,
        precision_action_steps: int = 1,
        adaptive_arm_step_ratio: float = 0.75,
        adaptive_gripper_step: float = 0.004,
    ) -> None:
        if control_hz <= 0:
            raise ValueError("control_hz must be positive")
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not (self.checkpoint / "model.safetensors").is_file():
            raise FileNotFoundError(f"ACT checkpoint is incomplete: {self.checkpoint}")

        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Action policy requested CUDA, but torch.cuda.is_available() is false")
        self._torch = torch
        self.device = device
        policy_config = PreTrainedConfig.from_pretrained(self.checkpoint)
        policy_class = get_policy_class(policy_config.type)
        self.policy = policy_class.from_pretrained(self.checkpoint, config=policy_config)
        self.policy.to(device)
        self.policy.eval()
        self.policy_type = str(self.policy.config.type)
        self.language_conditioned = self.policy_type in LANGUAGE_POLICY_TYPES
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=str(self.checkpoint),
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        shape = self.policy.config.input_features["observation.images.overhead"].shape
        self.height, self.width = int(shape[-2]), int(shape[-1])
        configured_steps = int(self.policy.config.n_action_steps)
        self.action_steps = configured_steps if action_steps is None else int(action_steps)
        if not 1 <= self.action_steps <= int(self.policy.config.chunk_size):
            raise ValueError("action_steps must be within the trained policy chunk")
        self.period_s = 1.0 / float(control_hz)
        self.max_arm_step = float(max_arm_step)
        self.adaptive_reobserve = bool(adaptive_reobserve)
        self.precision_action_steps = int(precision_action_steps)
        self.adaptive_arm_step_ratio = float(adaptive_arm_step_ratio)
        self.adaptive_gripper_step = float(adaptive_gripper_step)
        if not 1 <= self.precision_action_steps <= self.action_steps:
            raise ValueError("precision_action_steps must be within action_steps")
        self.inference_seed = int(inference_seed)
        torch.manual_seed(self.inference_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.inference_seed)
        self._actions: deque[np.ndarray] = deque()
        self._next_action_time: float | None = None
        self.inference_count = 0
        self.command_count = 0
        self.clipped_command_count = 0
        self.arm_step_clipped_command_count = 0
        self.joint_limit_clipped_command_count = 0
        self.clipped_joint_value_count = 0
        self.max_filter_correction = 0.0
        self.adaptive_reobservation_count = 0
        self.last_command_diagnostics: dict = {}

    def reset(self, time_s: float) -> None:
        self._actions.clear()
        self._next_action_time = float(time_s)
        self.inference_count = 0
        self.command_count = 0
        self.clipped_command_count = 0
        self.arm_step_clipped_command_count = 0
        self.joint_limit_clipped_command_count = 0
        self.clipped_joint_value_count = 0
        self.max_filter_correction = 0.0
        self.adaptive_reobservation_count = 0
        self.last_command_diagnostics = {}
        self.policy.reset()

    def due(self, time_s: float) -> bool:
        """Whether the simulator should acquire a new control-rate sample."""
        return self._next_action_time is None or float(time_s) + 1e-9 >= self._next_action_time

    def _image_tensor(self, rgb: np.ndarray):
        image = letterbox_rgb(rgb, self.height, self.width)
        return self._torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255.0)

    def predict_chunk(
        self,
        overhead_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        state: np.ndarray,
        instruction: str | None = None,
    ) -> np.ndarray:
        state_array = np.asarray(state, dtype=np.float32).reshape(-1)
        expected_state_dim = int(
            self.policy.config.input_features["observation.state"].shape[0]
        )
        if state_array.shape != (expected_state_dim,) or not np.isfinite(state_array).all():
            raise ValueError(
                f"Policy observation.state must contain {expected_state_dim} finite values"
            )
        if self.language_conditioned and not (instruction and instruction.strip()):
            raise ValueError(f"{self.policy_type} requires a non-empty language instruction")
        sample = {
            "observation.images.overhead": self._image_tensor(overhead_rgb),
            "observation.images.wrist": self._image_tensor(wrist_rgb),
            "observation.state": self._torch.from_numpy(state_array.copy()),
        }
        if self.language_conditioned:
            sample["task"] = instruction.strip()
        with self._torch.inference_mode():
            processed = self.preprocessor(sample)
            predicted = self.policy.predict_action_chunk(processed)
            predicted = self.postprocessor(predicted).cpu().numpy()[0, : self.action_steps]
        if predicted.shape != (self.action_steps, 9) or not np.isfinite(predicted).all():
            raise RuntimeError(f"Policy produced an invalid action chunk: {predicted.shape}")
        return np.asarray(predicted, dtype=np.float32)

    def observe(
        self,
        overhead_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        state: np.ndarray,
        instruction: str | None = None,
        current_joint_position: np.ndarray | None = None,
    ) -> None:
        predicted = self.predict_chunk(overhead_rgb, wrist_rgb, state, instruction)
        selected = len(predicted)
        reasons: tuple[str, ...] = ()
        if self.adaptive_reobserve and current_joint_position is not None:
            selected, reasons = adaptive_action_count(
                predicted,
                current_joint_position,
                max_arm_step=self.max_arm_step,
                precision_action_steps=self.precision_action_steps,
                arm_step_ratio=self.adaptive_arm_step_ratio,
                gripper_step=self.adaptive_gripper_step,
            )
            self.adaptive_reobservation_count += int(selected < len(predicted))
        self._actions.extend(
            np.asarray(action, dtype=np.float32) for action in predicted[:selected]
        )
        self.last_command_diagnostics = {
            "selected_chunk_steps": int(selected),
            "adaptive_reasons": list(reasons),
        }
        self.inference_count += 1

    def command(
        self,
        time_s: float,
        *,
        overhead_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        state: np.ndarray,
        current_joint_position: np.ndarray,
        instruction: str | None = None,
    ) -> np.ndarray | None:
        """Return a new target at control_hz; return None to hold the prior target."""
        now = float(time_s)
        if self._next_action_time is None:
            self.reset(now)
        assert self._next_action_time is not None
        if now + 1e-9 < self._next_action_time:
            return None
        if not self._actions:
            self.observe(
                overhead_rgb, wrist_rgb, state, instruction,
                current_joint_position=current_joint_position,
            )
        raw_action = self._actions.popleft()
        # Avoid a catch-up burst if rendering or inference takes longer than one period.
        self._next_action_time = now + self.period_s
        filtered, report = safety_filter_action_with_diagnostics(
            raw_action,
            current_joint_position,
            max_arm_step=self.max_arm_step,
        )
        self.command_count += 1
        self.clipped_command_count += int(report["clipped"])
        self.arm_step_clipped_command_count += int(report["arm_step_clips"] > 0)
        self.joint_limit_clipped_command_count += int(report["joint_limit_clips"] > 0)
        self.clipped_joint_value_count += int(report["joint_limit_clips"]) + int(
            report["arm_step_clips"]
        )
        self.max_filter_correction = max(
            self.max_filter_correction, float(report["max_abs_correction"])
        )
        self.last_command_diagnostics |= {
            "raw_action": raw_action.astype(float).tolist(),
            "filtered_action": filtered.astype(float).tolist(),
            **report,
            "queued_actions_remaining": len(self._actions),
        }
        return filtered


def encode_rgb_png(rgb: np.ndarray) -> str:
    image = Image.fromarray(np.asarray(rgb[..., :3], dtype=np.uint8), mode="RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_rgb_png(encoded: str) -> np.ndarray:
    with Image.open(BytesIO(base64.b64decode(encoded))) as image:
        return np.asarray(image.convert("RGB"))


class ACTSidecarPolicy:
    """Isaac-compatible policy client backed by the system Python training env."""

    def __init__(
        self,
        checkpoint: Path,
        *,
        python_executable: Path = Path("/usr/bin/python3"),
        device: str = "cuda",
        control_hz: float = 5.0,
        action_steps: int = 5,
        max_arm_step: float = 0.35,
        inference_seed: int = 20261018,
        adaptive_reobserve: bool = False,
        precision_action_steps: int = 1,
        adaptive_arm_step_ratio: float = 0.75,
        adaptive_gripper_step: float = 0.004,
        startup_timeout_s: float = 60.0,
    ) -> None:
        if control_hz <= 0 or action_steps <= 0 or max_arm_step <= 0:
            raise ValueError("Policy sidecar rate, action steps, and maximum arm step must be positive")
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.action_steps = int(action_steps)
        self.period_s = 1.0 / float(control_hz)
        self.max_arm_step = float(max_arm_step)
        self.adaptive_reobserve = bool(adaptive_reobserve)
        self.precision_action_steps = int(precision_action_steps)
        self.adaptive_arm_step_ratio = float(adaptive_arm_step_ratio)
        self.adaptive_gripper_step = float(adaptive_gripper_step)
        if not 1 <= self.precision_action_steps <= self.action_steps:
            raise ValueError("precision_action_steps must be within action_steps")
        self.inference_count = 0
        self.command_count = 0
        self.clipped_command_count = 0
        self.arm_step_clipped_command_count = 0
        self.joint_limit_clipped_command_count = 0
        self.clipped_joint_value_count = 0
        self.max_filter_correction = 0.0
        self.adaptive_reobservation_count = 0
        self.last_command_diagnostics: dict = {}
        self._actions: deque[np.ndarray] = deque()
        self._next_action_time: float | None = None
        self._closed = False

        socket_name = f"franka_act_{os.getpid()}_{uuid.uuid4().hex}.sock"
        self._socket_path = Path(tempfile.gettempdir()) / socket_name
        server = Path(__file__).resolve().with_name("act_inference_server.py")
        command = [
            str(Path(python_executable).expanduser()),
            str(server),
            "--checkpoint",
            str(self.checkpoint),
            "--socket",
            str(self._socket_path),
            "--device",
            device,
            "--action-steps",
            str(self.action_steps),
            "--inference-seed",
            str(int(inference_seed)),
        ]
        sidecar_environment = os.environ.copy()
        # Isaac's launcher exports a Python 3.12 PYTHONHOME/PYTHONPATH.  The
        # ACT sidecar deliberately uses the system Python 3.10 training env.
        sidecar_environment.pop("PYTHONHOME", None)
        sidecar_environment.pop("PYTHONPATH", None)
        self._process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=sidecar_environment,
        )
        assert self._process.stdout is not None
        deadline = time.monotonic() + startup_timeout_s
        while time.monotonic() < deadline:
            if self._socket_path.exists():
                break
            if self._process.poll() is not None:
                output = self._process.stdout.read()
                raise RuntimeError("ACT sidecar exited during startup:\n" + output[-8000:])
            time.sleep(0.05)
        else:
            self._process.terminate()
            raise TimeoutError("Timed out waiting for ACT inference sidecar")

        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket.connect(str(self._socket_path))
        self._stream = self._socket.makefile("rwb")
        description = self._request({"op": "describe"})
        self.state_dim = int(description["state_dim"])
        self.policy_type = str(description["policy_type"])
        self.language_conditioned = bool(description["language_conditioned"])
        if int(description["action_steps"]) != self.action_steps:
            raise RuntimeError("Policy sidecar action-step configuration mismatch")
        atexit.register(self.close)

    def _request(self, payload: dict) -> dict:
        self._stream.write(json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n")
        self._stream.flush()
        line = self._stream.readline()
        if not line:
            raise RuntimeError("ACT sidecar disconnected")
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(f"ACT sidecar inference failed: {response.get('error', 'unknown error')}")
        return response

    def reset(self, time_s: float) -> None:
        self._actions.clear()
        self._next_action_time = float(time_s)
        self.inference_count = 0
        self.command_count = 0
        self.clipped_command_count = 0
        self.arm_step_clipped_command_count = 0
        self.joint_limit_clipped_command_count = 0
        self.clipped_joint_value_count = 0
        self.max_filter_correction = 0.0
        self.adaptive_reobservation_count = 0
        self.last_command_diagnostics = {}
        self._request({"op": "reset"})

    def due(self, time_s: float) -> bool:
        return self._next_action_time is None or float(time_s) + 1e-9 >= self._next_action_time

    def observe(
        self,
        overhead_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        state: np.ndarray,
        instruction: str | None = None,
        current_joint_position: np.ndarray | None = None,
    ) -> None:
        state_array = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_array.shape != (self.state_dim,) or not np.isfinite(state_array).all():
            raise ValueError(
                f"Policy observation.state must contain {self.state_dim} finite values"
            )
        if self.language_conditioned and not (instruction and instruction.strip()):
            raise ValueError(f"{self.policy_type} requires a non-empty language instruction")
        # Resize before IPC to keep each request small and deterministic.
        overhead = letterbox_rgb(overhead_rgb, 256, 256)
        wrist = letterbox_rgb(wrist_rgb, 256, 256)
        response = self._request(
            {
                "op": "infer",
                "overhead_png": encode_rgb_png(overhead),
                "wrist_png": encode_rgb_png(wrist),
                "state": state_array.tolist(),
                "instruction": instruction,
            }
        )
        predicted = np.asarray(response["actions"], dtype=np.float32)
        if predicted.shape != (self.action_steps, 9) or not np.isfinite(predicted).all():
            raise RuntimeError(f"Policy sidecar returned invalid actions: {predicted.shape}")
        selected = len(predicted)
        reasons: tuple[str, ...] = ()
        if self.adaptive_reobserve and current_joint_position is not None:
            selected, reasons = adaptive_action_count(
                predicted,
                current_joint_position,
                max_arm_step=self.max_arm_step,
                precision_action_steps=self.precision_action_steps,
                arm_step_ratio=self.adaptive_arm_step_ratio,
                gripper_step=self.adaptive_gripper_step,
            )
            self.adaptive_reobservation_count += int(selected < len(predicted))
        self._actions.extend(predicted[:selected])
        self.last_command_diagnostics = {
            "selected_chunk_steps": int(selected),
            "adaptive_reasons": list(reasons),
        }
        self.inference_count += 1

    def command(
        self,
        time_s: float,
        *,
        overhead_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        state: np.ndarray,
        current_joint_position: np.ndarray,
        instruction: str | None = None,
    ) -> np.ndarray | None:
        now = float(time_s)
        if self._next_action_time is None:
            self.reset(now)
        assert self._next_action_time is not None
        if now + 1e-9 < self._next_action_time:
            return None
        if not self._actions:
            self.observe(
                overhead_rgb, wrist_rgb, state, instruction,
                current_joint_position=current_joint_position,
            )
        raw_action = self._actions.popleft()
        self._next_action_time = now + self.period_s
        filtered, report = safety_filter_action_with_diagnostics(
            raw_action, current_joint_position, max_arm_step=self.max_arm_step
        )
        self.command_count += 1
        self.clipped_command_count += int(report["clipped"])
        self.arm_step_clipped_command_count += int(report["arm_step_clips"] > 0)
        self.joint_limit_clipped_command_count += int(report["joint_limit_clips"] > 0)
        self.clipped_joint_value_count += int(report["joint_limit_clips"]) + int(
            report["arm_step_clips"]
        )
        self.max_filter_correction = max(
            self.max_filter_correction, float(report["max_abs_correction"])
        )
        self.last_command_diagnostics |= {
            "raw_action": raw_action.astype(float).tolist(),
            "filtered_action": filtered.astype(float).tolist(),
            **report,
            "queued_actions_remaining": len(self._actions),
        }
        return filtered

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._request({"op": "close"})
        except Exception:
            pass
        try:
            self._stream.close()
            self._socket.close()
        except Exception:
            pass
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            self._process.wait(timeout=5)
        self._socket_path.unlink(missing_ok=True)
