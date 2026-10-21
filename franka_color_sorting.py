"""Sort visually identified objects into a physical target box with a Franka Panda.

Run from the Isaac Sim root, for example:
    ./python.sh /path/to/franka_color_sorting/franka_color_sorting.py --target red

The controller accepts either the reproducible colour-mask baseline or generic
RGB-D geometry proposals associated with open-vocabulary semantic prompts.
"""

from __future__ import annotations

import argparse
import json
import queue
import socket
import sys
import threading
import traceback
from dataclasses import dataclass, replace
from pathlib import Path

from isaacsim import SimulationApp


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--target",
    nargs="+",
    default=["red"],
    help="Colors (red blue green yellow) or categories (warm cool primary all).",
)
parser.add_argument("--headless", action="store_true", help="Run without the desktop viewport.")
parser.add_argument("--test", action="store_true", help="Fail if the requested blocks are not placed in time.")
parser.add_argument("--server-port", type=int, help="Run a persistent JSON-lines skill server on localhost.")
parser.add_argument("--capture-path", help="Save one calibrated overhead RGB observation, then exit.")
parser.add_argument(
    "--rgbd",
    action="store_true",
    help="Capture overhead RGB-D in this scene and use its local poses for pickup; never fall back to scene poses.",
)
parser.add_argument("--wrist-capture-path", help="Save one RGB-D image from the camera mounted on Franka's wrist, then exit.")
parser.add_argument("--preview-wrist-camera", action="store_true", help="Open Isaac Sim and keep the mounted RealSense D455 visible for inspection.")
parser.add_argument("--vision-poses", help="JSON RGB-D pose estimates used for pickup approach (not success evaluation).")
parser.add_argument("--scenario", help="JSON benchmark scenario containing spawn, camera, and lighting randomization.")
parser.add_argument("--rgbd-output", help="RGB image path used by --rgbd (poses are written beside it).")
parser.add_argument(
    "--perception-backend", choices=("color", "generic"), default="color",
    help="Instance perception backend: colour-mask baseline or generic RGB-D geometry plus semantic labels.",
)
parser.add_argument(
    "--semantic-backend", choices=("clip", "smolvlm", "prototype"), default="clip",
    help="Crop classifier used by --perception-backend generic.",
)
parser.add_argument("--semantic-vocabulary", help="JSON mapping scene object IDs to open-vocabulary descriptions.")
parser.add_argument("--semantic-model", help="Optional local/Hugging Face model override for the semantic backend.")
parser.add_argument("--semantic-min-confidence", type=float, default=0.50, help="Reject generic semantic assignments below this relative confidence.")
parser.add_argument(
    "--object-set", choices=("cubes", "seen", "heldout"), default="cubes",
    help="Scene asset split. heldout contains unseen shape/size/pose combinations for evaluation.",
)
parser.add_argument("--wrist-mode", choices=("off", "single", "servo"), help="Wrist feedback ablation mode (default: servo).")
parser.add_argument("--disable-wrist-refinement", action="store_true", help="Deprecated alias for --wrist-mode off.")
parser.add_argument("--disable-wrist-gate", action="store_true", help="Ablation only: accept visible wrist estimates without quality/displacement gating.")
parser.add_argument("--visual-servo-rate-hz", type=float, default=10.0, help="Wrist RGB-D corrections per second during approach.")
parser.add_argument("--visual-servo-alpha", type=float, default=0.70, help="EMA gain for accepted wrist RGB-D corrections.")
parser.add_argument("--visual-servo-max-step-mm", type=float, default=8.0, help="Maximum XYZ target correction per wrist frame.")
parser.add_argument("--max-recovery-attempts", type=int, choices=(0, 1), default=1, help="Fresh RGB-D retries after a verified grasp/placement failure.")
parser.add_argument("--artifact-dir", help="Directory for wrist RGB-D debug artifacts (defaults to observations/).")
parser.add_argument("--record-dataset", type=Path, help="Record this expert run as a raw VLA episode under this dataset root.")
parser.add_argument("--record-instruction", help="Natural-language task stored with every recorded VLA frame.")
parser.add_argument("--record-fps", type=float, default=10.0, help="VLA demonstration sampling rate (maximum 20 Hz).")
parser.add_argument("--record-episode-index", type=int, help="Explicit raw episode index; default selects the next free index.")
parser.add_argument("--dagger-dataset", type=Path, help="Record online SmolVLA states with shadow-expert action labels.")
parser.add_argument("--dagger-beta", type=float, default=0.8, help="Teacher fraction in the executed DAgger action (0..1).")
parser.add_argument("--dagger-round", type=int, default=1, help="Positive DAgger aggregation round stored in metadata.")
parser.add_argument(
    "--dagger-intervention-threshold",
    type=float,
    default=0.75,
    help="Use the full teacher action when arm-action L2 disagreement exceeds this value.",
)
parser.add_argument("--act-checkpoint", type=Path, help="Run the trained ACT policy from this pretrained_model directory.")
parser.add_argument(
    "--vla-checkpoint",
    type=Path,
    help="Run a language-conditioned SmolVLA policy from this pretrained_model directory.",
)
parser.add_argument(
    "--vla-instruction",
    help="Natural-language goal for a single-target SmolVLA rollout (default uses the training template).",
)
parser.add_argument("--act-control-hz", type=float, default=5.0, help="ACT joint-target execution rate (must match demonstrations).")
parser.add_argument("--act-action-steps", type=int, default=5, help="Actions executed before ACT observes both cameras again.")
parser.add_argument(
    "--act-adaptive-reobserve", action="store_true",
    help="Shorten risky action chunks and reacquire both cameras before continuing.",
)
parser.add_argument(
    "--act-precision-action-steps", type=int, default=1,
    help="Actions retained from a risky chunk when --act-adaptive-reobserve is enabled.",
)
parser.add_argument(
    "--act-telemetry-images-every", type=int, default=0,
    help="Save one RGB-D diagnostic sample every N policy inferences (0 disables images).",
)
parser.add_argument("--act-max-arm-step", type=float, default=0.35, help="Maximum per-command arm-joint change in radians.")
parser.add_argument("--act-timeout", type=float, default=300.0, help="Maximum ACT rollout duration in simulation seconds.")
parser.add_argument("--policy-seed", type=int, default=20261018, help="Reproducible learned-policy sampling seed.")
args, _ = parser.parse_known_args()
if args.disable_wrist_refinement and args.wrist_mode not in (None, "off"):
    parser.error("--disable-wrist-refinement cannot be combined with a non-off --wrist-mode")
args.wrist_mode = "off" if args.disable_wrist_refinement else (args.wrist_mode or "servo")
if args.visual_servo_rate_hz <= 0.0:
    parser.error("--visual-servo-rate-hz must be positive")
if not 0.0 < args.visual_servo_alpha <= 1.0:
    parser.error("--visual-servo-alpha must be within (0, 1]")
if args.visual_servo_max_step_mm <= 0.0:
    parser.error("--visual-servo-max-step-mm must be positive")
if not 0.0 <= args.semantic_min_confidence <= 1.0:
    parser.error("--semantic-min-confidence must be within 0..1")
if not 0.0 < args.record_fps <= 20.0:
    parser.error("--record-fps must be within (0, 20]")
if args.record_episode_index is not None and args.record_episode_index < 0:
    parser.error("--record-episode-index must be non-negative")
if not 0.0 <= args.dagger_beta <= 1.0:
    parser.error("--dagger-beta must be within 0..1")
if args.dagger_round <= 0 or args.dagger_intervention_threshold <= 0:
    parser.error("--dagger-round and --dagger-intervention-threshold must be positive")
if args.record_dataset and args.dagger_dataset:
    parser.error("Choose only one of --record-dataset and --dagger-dataset")
if args.record_dataset and (args.capture_path or args.wrist_capture_path or args.preview_wrist_camera or args.server_port):
    parser.error("--record-dataset requires a one-shot manipulation task, not capture/preview/server mode")
if args.act_control_hz <= 0 or args.act_max_arm_step <= 0 or args.act_timeout <= 0:
    parser.error("ACT rate, maximum arm step, and timeout must be positive")
if args.act_action_steps <= 0:
    parser.error("--act-action-steps must be positive")
if not 1 <= args.act_precision_action_steps <= args.act_action_steps:
    parser.error("--act-precision-action-steps must be within --act-action-steps")
if args.act_telemetry_images_every < 0:
    parser.error("--act-telemetry-images-every must be non-negative")
if args.act_checkpoint and args.vla_checkpoint:
    parser.error("Choose only one of --act-checkpoint and --vla-checkpoint")
policy_checkpoint = args.vla_checkpoint or args.act_checkpoint
if args.vla_instruction and not args.vla_checkpoint:
    parser.error("--vla-instruction requires --vla-checkpoint")
if args.dagger_dataset and not args.vla_checkpoint:
    parser.error("--dagger-dataset requires --vla-checkpoint")
if policy_checkpoint and any((args.server_port, args.capture_path, args.wrist_capture_path, args.preview_wrist_camera, args.record_dataset, args.rgbd)):
    parser.error("A learned-policy checkpoint is a standalone rollout mode and cannot be combined with server/capture/record/RGB-D expert modes")

simulation_app = SimulationApp({"headless": args.headless, "hide_ui": False})

if args.headless:
    from isaacsim.core.experimental.utils.app import enable_extension

    simulation_app.set_setting("/app/window/drawMouse", True)
    enable_extension("omni.kit.livestream.app")

# The maintained Isaac Sim state-machine tutorial supplies the tested Franka,
# RMPflow, gripper and grasp-validation implementation used by this demo.
EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "isaacsim" / "standalone_examples" / "tutorials" / "state_machine"
if not EXAMPLE_DIR.is_dir():
    raise RuntimeError(f"Isaac Sim examples not found at {EXAMPLE_DIR}")
sys.path.insert(0, str(EXAMPLE_DIR))

import isaacsim.core.experimental.utils.app as app_utils
import numpy as np
from franka_stacking import (
    FrankaStackingSceneIO,
    Observation,
    PickPlaceController,
    PickPlaceDecision,
    PickPlacePhase,
    TaskConfig,
    make_pick_place_controller,
)
from isaacsim.core.experimental.objects import Cone, Cube, Cylinder, Sphere
from isaacsim.core.experimental.materials import RigidBodyMaterial
from isaacsim.core.experimental.prims import Articulation, GeomPrim, RigidPrim, XformPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.rendering_manager import ViewportManager
import omni.kit.app

import franka_stacking as stacking_module
from object_catalog import object_specs, semantic_vocabulary_for
from camera_extrinsics import DEFAULT_WRIST_CAMERA_EXTRINSICS, compose_pose
from vla_dataset import RawVLAEpisodeRecorder
from act_runtime import (
    ACT_GOAL_COLORS,
    ACTSidecarPolicy,
    PANDA_JOINT_NAMES,
    build_robot_state,
    language_goal_instruction,
    safety_filter_action,
)
from policy_diagnostics import PolicyRolloutTelemetry, save_policy_observation


PHYSICS_DT = 1.0 / 60.0
COLORS = ("red", "blue", "green", "yellow")
CATEGORIES = {
    "warm": ("red", "yellow"),
    "cool": ("blue", "green"),
    "primary": ("red", "blue", "yellow"),
    "all": COLORS,
}
COLOR_RGB = {
    "red": (0.8, 0.1, 0.1),
    "blue": (0.1, 0.1, 0.8),
    "green": (0.1, 0.7, 0.1),
    "yellow": (0.9, 0.8, 0.1),
}
DEFAULT_SEMANTIC_VOCABULARY = {
    "red": "a red cube",
    "blue": "a blue cube",
    "green": "a green cube",
    "yellow": "a yellow cube",
}
BOX_CENTER = np.array([0.55, 0.20])
TARGET_BOX_BASE_TOP_M = 0.012
TARGET_BOX_PACKING_OFFSETS = (
    np.array([-0.035, -0.035]),
    np.array([0.035, -0.035]),
    np.array([-0.035, 0.035]),
    np.array([0.035, 0.035]),
)
# A persistent service may observe repeatedly. Reusing the render product is
# essential: repeatedly creating a Camera at the same USD path leaks/competes
# for annotators in Kit and destabilizes long-running sessions.
_OVERHEAD_CAMERA = None
_OVERHEAD_EYE = np.array([0.50, -0.17, 1.60])
_OVERHEAD_TARGET = np.array([0.50, -0.17, 0.0])
# Fixed hand-eye calibration in the end-effector frame. The housing is on the
# side/upper rear of the hand, outside the finger closing region; the optical
# centre sits just in front of the housing and looks along the tool approach.
WRIST_CAMERA_EXTRINSICS = DEFAULT_WRIST_CAMERA_EXTRINSICS
_ACTIVE_VLA_RECORDER: RawVLAEpisodeRecorder | None = None


def wrist_mount_link(scene: FrankaStackingSceneIO) -> GeomPrim:
    """Return the rigid Panda hand link, never a moving finger link."""
    articulation = getattr(scene, "_articulation", None)
    if articulation is None:
        raise RuntimeError("Franka articulation is not initialized; cannot mount wrist camera")
    link_names = list(articulation.link_names)
    if "panda_hand" not in link_names:
        raise RuntimeError("Franka articulation has no panda_hand link for rigid camera mounting")
    path = str(articulation.link_paths[0][link_names.index("panda_hand")])
    return GeomPrim(paths=path)


def semantic_vocabulary() -> dict[str, str]:
    if not args.semantic_vocabulary:
        return semantic_vocabulary_for(args.object_set)
    path = Path(args.semantic_vocabulary).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != set(COLORS):
        raise ValueError(f"semantic vocabulary must describe exactly these scene IDs: {', '.join(COLORS)}")
    if not all(isinstance(value, str) and value.strip() for value in payload.values()):
        raise ValueError("semantic vocabulary descriptions must be non-empty strings")
    return {str(key): str(value) for key, value in payload.items()}


@dataclass
class PickPlaceState:
    """State shape consumed by the maintained tutorial's controller."""

    color: str
    destination: np.ndarray
    phase: PickPlacePhase
    phase_started: float
    settle_started: float | None = None
    pickup_position: np.ndarray | None = None
    cube_reference_xy: np.ndarray | None = None
    aborted: bool = False

    @property
    def holding(self) -> bool:
        return self.phase in {PickPlacePhase.LIFT, PickPlacePhase.TRANSPORT, PickPlacePhase.LOWER}


def resolve_targets(requests: list[str]) -> tuple[str, ...]:
    """Expand color/category selectors while retaining a stable pick order."""
    selected: list[str] = []
    for request in requests:
        name = request.lower()
        expanded = CATEGORIES.get(name, (name,))
        for color in expanded:
            if color not in COLORS:
                choices = ", ".join((*COLORS, *CATEGORIES))
                raise ValueError(f"Unknown target '{request}'. Choose one of: {choices}")
            if color not in selected:
                selected.append(color)
    if not selected:
        raise ValueError("Select at least one color or category")
    return tuple(selected)


def make_config(targets: tuple[str, ...], spawn_override: dict[str, np.ndarray] | None = None) -> TaskConfig:
    """Spawn four blocks, but reserve the target-box slots for requested ones."""
    cube_size = 0.05
    spawn_positions = {
        "red": np.array([0.45, -0.30, cube_size / 2]),
        "blue": np.array([0.55, -0.30, cube_size / 2]),
        "green": np.array([0.45, -0.45, cube_size / 2]),
        "yellow": np.array([0.55, -0.45, cube_size / 2]),
    }
    if spawn_override is not None:
        spawn_positions = {color: np.asarray(spawn_override[color], dtype=float) for color in COLORS}
    specs = object_specs(args.object_set)
    for color in COLORS:
        spawn_positions[color][2] = specs[color].dimensions_xyz[2] / 2.0
    # Non-targets retain harmless locations outside the goal test. Cubes use
    # vertical stacking; held-out curved/tapered objects use floor packing.
    stack_positions = {color: spawn_positions[color].copy() for color in COLORS}
    stack_height = 0.0
    for index, color in enumerate(targets):
        object_height = specs[color].dimensions_xyz[2]
        if args.object_set == "heldout":
            # A sphere, cone and tall cylinder do not form a physically stable
            # vertical stack. Pack the four held-out objects into collision-free
            # 2x2 floor slots inside the 18 cm box instead.
            stack_positions[color] = np.array([
                *(BOX_CENTER + TARGET_BOX_PACKING_OFFSETS[index]),
                TARGET_BOX_BASE_TOP_M + object_height / 2.0,
            ])
        else:
            stack_positions[color] = np.array([
                BOX_CENTER[0], BOX_CENTER[1], stack_height + object_height / 2.0
            ])
            stack_height += object_height
    return TaskConfig(
        cube_order=COLORS,
        spawn_positions=spawn_positions,
        stack_positions=stack_positions,
        cube_size=cube_size,
        # The stock tutorial's 10 cm pre-grasp puts a camera mounted 6 cm
        # below the wrist only ~1.5 cm above the cube top. At 22 cm the wrist
        # RGB-D frame includes the whole cube plus surrounding tabletop.
        above_height=0.22,
    )


class GeneralObjectSceneIO(FrankaStackingSceneIO):
    """Scene adapter that swaps fixed cubes for graspable primitive objects."""

    async def setup(self) -> None:
        assets_root = await stacking_module.get_assets_root_path_async()
        stacking_module.stage_utils.add_reference_to_stage(
            usd_path=assets_root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
            path=stacking_module._ROBOT_PATH,
            variants=[("Gripper", "AlternateFinger"), ("Mesh", "Performance")],
        )
        stacking_module.GroundPlane("/World/GroundPlane")
        stacking_module.DomeLight("/World/DomeLight").set_intensities(1000)
        specs = object_specs(args.object_set)
        for object_id in self._config.cube_order:
            spec = specs[object_id]
            path = stacking_module._CUBE_PATHS[object_id]
            half_yaw = spec.yaw_rad / 2.0
            orientation = [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)]
            common = {
                "paths": path,
                "positions": self._config.spawn_positions[object_id],
                "orientations": orientation,
                "colors": spec.color_rgb,
            }
            if spec.shape == "cuboid":
                Cube(**common, sizes=1.0, scales=spec.dimensions_xyz)
            elif spec.shape == "cylinder":
                Cylinder(**common, radii=spec.dimensions_xyz[0] / 2.0, heights=spec.dimensions_xyz[2], axes="Z")
            elif spec.shape == "sphere":
                Sphere(**common, radii=spec.dimensions_xyz[0] / 2.0)
            elif spec.shape == "cone":
                Cone(**common, radii=spec.dimensions_xyz[0] / 2.0, heights=spec.dimensions_xyz[2], axes="Z")
            else:
                raise ValueError(f"unsupported object shape: {spec.shape}")
            RigidPrim(paths=path)
            GeomPrim(paths=path, apply_collision_apis=True)

        # The alternate Franka fingers represent metal links, while physical
        # parallel grippers normally use rubber pads. A calibrated high-friction
        # material is required to lift tapered/curved objects such as a cone;
        # with Isaac's default contact material the fingers close correctly but
        # the object slides out during lift. Bind at the finger-link roots with
        # stronger-than-descendants inheritance so every collision mesh uses
        # the same pad contact model.
        gripper_pad = RigidBodyMaterial(
            "/World/PhysicsMaterials/FrankaRubberPads",
            static_frictions=[1.5],
            dynamic_frictions=[1.2],
            restitutions=[0.0],
        )
        gripper_pad.set_combine_modes(frictions=["max"], restitutions=["min"])
        for finger in ("panda_leftfinger", "panda_rightfinger"):
            XformPrim(paths=f"{stacking_module._ROBOT_PATH}/{finger}").apply_physics_materials(
                gripper_pad,
                weaker_than_descendants=[False],
            )

        await omni.kit.app.get_app().next_update_async()
        ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=[1.5, 0.5, 1.2], target=[0.5, 0.0, 0.2])
        self._articulation = Articulation(stacking_module._ROBOT_PATH)
        self._articulation.set_default_state(dof_positions=[stacking_module._DEFAULT_JOINTS])
        self._cubes = {
            object_id: GeomPrim(paths=stacking_module._CUBE_PATHS[object_id])
            for object_id in self._config.cube_order
        }
        await omni.kit.app.get_app().next_update_async()


def load_benchmark_scenario(path: Path) -> dict:
    """Validate a generated benchmark scenario before authoring the scene."""
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("Benchmark scenario schema_version must be 1")
    raw_positions = payload.get("spawn_positions")
    if not isinstance(raw_positions, dict) or set(raw_positions) != set(COLORS):
        raise ValueError(f"spawn_positions must contain exactly: {', '.join(COLORS)}")
    positions: dict[str, np.ndarray] = {}
    for color in COLORS:
        position = np.asarray(raw_positions[color], dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError(f"Invalid XYZ spawn position for {color}")
        if not (0.30 <= position[0] <= 0.70 and -0.60 <= position[1] <= -0.15 and 0.015 <= position[2] <= 0.04):
            raise ValueError(f"Spawn position for {color} is outside the benchmark workspace: {position.tolist()}")
        positions[color] = position
    for index, first in enumerate(COLORS):
        for second in COLORS[index + 1 :]:
            if np.linalg.norm(positions[first][:2] - positions[second][:2]) < 0.065:
                raise ValueError(f"Benchmark cubes {first} and {second} overlap or are too close")
    camera = payload.get("camera", {})
    eye = np.asarray(camera.get("eye", _OVERHEAD_EYE), dtype=float)
    target = np.asarray(camera.get("target", _OVERHEAD_TARGET), dtype=float)
    if eye.shape != (3,) or target.shape != (3,) or not np.isfinite(eye).all() or not np.isfinite(target).all():
        raise ValueError("Benchmark camera eye and target must be finite XYZ points")
    intensity = float(payload.get("dome_light_intensity", 1000.0))
    if not 100.0 <= intensity <= 5000.0:
        raise ValueError("dome_light_intensity must be within 100..5000")
    faults = payload.get("faults", {})
    if not isinstance(faults, dict):
        raise ValueError("scenario faults must be an object")
    bias = np.asarray(faults.get("initial_overhead_xy_bias_m", [0.0, 0.0]), dtype=float)
    if bias.shape != (2,) or not np.isfinite(bias).all() or np.linalg.norm(bias) > 0.05:
        raise ValueError("initial_overhead_xy_bias_m must be a finite XY offset within 50 mm")
    dropout = float(faults.get("wrist_dropout_probability", 0.0))
    noise = float(faults.get("wrist_noise_std_m", 0.0))
    if not 0.0 <= dropout <= 0.90 or not 0.0 <= noise <= 0.02:
        raise ValueError("wrist fault parameters exceed configured safety bounds")
    shift = faults.get("target_shift")
    if shift is not None:
        if not isinstance(shift, dict) or shift.get("color") not in COLORS:
            raise ValueError("target_shift requires a known color")
        delta = np.asarray(shift.get("xy_m"), dtype=float)
        # Manual demonstration scenarios may use a more visible disturbance
        # than the formal benchmark's generated 20--38 mm range. This moves a
        # simulated object only; the normal 30 mm visual-servo acceptance gate
        # remains unchanged, forcing large shifts through verified recovery.
        if delta.shape != (2,) or not np.isfinite(delta).all() or np.linalg.norm(delta) > 0.08:
            raise ValueError("target_shift.xy_m must be a finite offset within 80 mm")
    return {
        "spawn_positions": positions,
        "camera_eye": eye,
        "camera_target": target,
        "light_intensity": intensity,
        "faults": faults,
        "fault_seed": int(payload.get("seed", 0)),
    }


def apply_initial_overhead_fault(path: Path, faults: dict) -> None:
    """Inject a recorded perception fault into controller input only."""
    bias = np.asarray(faults.get("initial_overhead_xy_bias_m", [0.0, 0.0]), dtype=float)
    if not np.any(bias):
        return
    payload = json.loads(path.read_text())
    for pose in payload.get("poses", {}).values():
        pose[0] = float(pose[0] + bias[0])
        pose[1] = float(pose[1] + bias[1])
    payload["fault_injection"] = {
        "stage": "initial_overhead",
        "xy_bias_m": bias.tolist(),
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"[fault-injection] initial overhead XY bias={np.round(bias * 1000.0, 2).tolist()}mm", flush=True)


def inject_target_shift(
    scene: FrankaStackingSceneIO,
    observation: Observation,
    shift: dict,
) -> None:
    """Apply a benchmark-only external disturbance to one physical block."""
    color = str(shift["color"])
    delta = np.asarray(shift["xy_m"], dtype=float)
    current = np.asarray(observation.cube_positions[color], dtype=float)
    moved = current.copy()
    moved[:2] += delta
    cube = getattr(scene, "_cubes", {}).get(color)
    if cube is None:
        raise RuntimeError(f"fault injector cannot access physical cube {color}")
    cube.set_world_poses(positions=[moved])
    print(
        f"[fault-injection] external target shift color={color} "
        f"delta_mm={np.round(delta * 1000.0, 2).tolist()}",
        flush=True,
    )


class TargetSelectionFSM:
    """Reactive pick/place state machine, limited to requested scene labels."""

    _NEXT_PHASE = {
        PickPlacePhase.PRE_GRASP: PickPlacePhase.APPROACH,
        PickPlacePhase.APPROACH: PickPlacePhase.GRASP,
        PickPlacePhase.GRASP: PickPlacePhase.LIFT,
        PickPlacePhase.LIFT: PickPlacePhase.TRANSPORT,
        PickPlacePhase.TRANSPORT: PickPlacePhase.LOWER,
        PickPlacePhase.LOWER: PickPlacePhase.RELEASE,
        PickPlacePhase.RELEASE: PickPlacePhase.RETRACT,
    }
    _SHORT_PHASES = {
        PickPlacePhase.APPROACH,
        PickPlacePhase.LIFT,
        PickPlacePhase.TRANSPORT,
        PickPlacePhase.LOWER,
    }

    def __init__(self, config: TaskConfig, targets: tuple[str, ...]) -> None:
        self._config = config
        self._targets = targets
        self._active: PickPlaceState | None = None
        # RGB-D is a task-start snapshot.  Once a block has been physically
        # placed, do not let that old source-pose make us select it again.
        self._completed: set[str] = set()
        self.failure_reason: str | None = None
        self.failed_color: str | None = None
        self._grasp_failures: dict[str, int] = {}

    def reset(self) -> None:
        self._active = None
        self._completed.clear()
        self.failure_reason = None
        self.failed_color = None
        self._grasp_failures.clear()

    def forward(self, observation: Observation, *, target_reached: bool, gripper_reached: bool) -> PickPlaceDecision:
        if self.failure_reason is not None:
            return PickPlaceDecision(done=True)
        if self._active is None:
            self._active = self._select_next_cube(observation)
            if self._active is None:
                return PickPlaceDecision(done=True)
        if self._cube_was_dropped(observation):
            color = self._active.color
            self._grasp_failures[color] = self._grasp_failures.get(color, 0) + 1
            self.failure_reason = (
                f"{color} slipped or was not grasped; RGB-D pose is stale after contact "
                "and a fresh observation is required before retrying"
            )
            self.failed_color = color
            print(f"[recovery-required] {self.failure_reason}")
            self._active = None
            return PickPlaceDecision()
        self._restart_pickup_if_cube_moved(observation)
        self._advance_if_ready(observation, target_reached, gripper_reached)
        return self._decision()

    def _select_next_cube(self, observation: Observation) -> PickPlaceState | None:
        for color in self._targets:
            if color in self._completed:
                continue
            if not self._at_goal(observation.cube_positions[color], self._config.stack_positions[color]):
                print(f"[select] {color} -> target box")
                return PickPlaceState(
                    color=color,
                    destination=self._config.stack_positions[color].copy(),
                    phase=PickPlacePhase.PRE_GRASP,
                    phase_started=observation.time,
                )
        print("[complete] selected blocks are in the target box")
        return None

    def _cube_was_dropped(self, observation: Observation) -> bool:
        active = self._require_active()
        return active.holding and bool(
            np.linalg.norm(observation.cube_positions[active.color] - observation.end_effector_position)
            > self._config.drop_tolerance
        )

    def _restart_pickup_if_cube_moved(self, observation: Observation) -> None:
        active = self._require_active()
        if active.holding or active.pickup_position is not None:
            return
        current_xy = observation.cube_positions[active.color][:2]
        if active.cube_reference_xy is None:
            active.cube_reference_xy = current_xy.copy()
        elif np.linalg.norm(current_xy - active.cube_reference_xy) > self._config.cube_move_tolerance:
            active.phase = PickPlacePhase.PRE_GRASP
            active.phase_started = observation.time
            active.settle_started = None
            active.cube_reference_xy = current_xy.copy()

    def _advance_if_ready(self, observation: Observation, target_reached: bool, gripper_reached: bool) -> None:
        active = self._require_active()
        elapsed = max(0.0, observation.time - active.phase_started)
        if active.phase is PickPlacePhase.GRASP:
            if elapsed >= self._config.minimum_duration:
                self._advance(observation)
            return
        if active.phase is PickPlacePhase.RELEASE:
            if elapsed >= self._config.minimum_duration and gripper_reached:
                self._advance(observation)
            return
        if target_reached:
            if active.settle_started is None:
                active.settle_started = observation.time
        else:
            active.settle_started = None
        settled = active.settle_started is not None and observation.time - active.settle_started >= self._config.settle_duration
        timeout = self._config.short_phase_timeout if active.phase in self._SHORT_PHASES else self._config.phase_timeout
        if elapsed >= self._config.minimum_duration and settled:
            self._advance(observation, "settled")
        elif elapsed >= timeout:
            self._advance(observation, "timeout")

    def _advance(self, observation: Observation, reason: str = "ok") -> None:
        active = self._require_active()
        print(f"  [{active.color}] {active.phase.name.lower()} done ({reason})")
        if active.phase is PickPlacePhase.RETRACT:
            # This is a local simulator hard-rule verification, not an LLM
            # judgement. It prevents a stale pre-task RGB-D image from making
            # the policy pick an already placed cube at its old location.
            if self._at_goal(observation.cube_positions[active.color], self._config.stack_positions[active.color]):
                self._completed.add(active.color)
                print(f"  [{active.color}] placement verified; marking complete")
            else:
                self.failure_reason = (
                    f"{active.color} was not in target_box after release; "
                    "its visual pose may be stale and a fresh RGB-D observation is required"
                )
                self.failed_color = active.color
                print(f"  [recovery-required] {self.failure_reason}")
            self._active = None
            return
        active.phase = self._NEXT_PHASE[active.phase]
        active.phase_started = observation.time
        active.settle_started = None
        if active.phase is PickPlacePhase.GRASP:
            active.pickup_position = observation.cube_positions[active.color].copy()
        # Do not redirect a successfully grasped block back to its spawn pose.
        # During GRASP the policy intentionally consumes the task-start RGB-D
        # snapshot for pickup accuracy.  That snapshot still shows an earlier
        # placed block at its source position, so using it here to validate a
        # stack base can falsely decide that the base is absent and return the
        # new block to its original location.  Physics-based verification after
        # RELEASE remains the authoritative success/failure check.

    def _decision(self) -> PickPlaceDecision:
        active = self._active
        if active is None:
            return PickPlaceDecision()
        return PickPlaceDecision(
            phase=active.phase,
            color=active.color,
            destination=active.destination,
            pickup_position=active.pickup_position,
        )

    def _stack_below_is_valid(self, color: str, observation: Observation) -> bool:
        index = self._targets.index(color)
        return all(
            self._in_target_box(observation.cube_positions[lower])
            for lower in self._targets[:index]
        )

    def _in_target_box(self, position: np.ndarray) -> bool:
        """Accept a physically stable lower block anywhere inside the box.

        Requiring an exact pre-authored stack coordinate made a small PhysX
        settling displacement look like a missing base, so later blocks were
        returned to their original locations.  The box itself is the actual
        task constraint.
        """
        return bool(
            np.linalg.norm(position[:2] - BOX_CENTER) < 0.10
            and position[2] > self._config.cube_size * 0.35
        )

    def _at_goal(self, position: np.ndarray, goal: np.ndarray) -> bool:
        return bool(np.linalg.norm(position - goal) < self._config.goal_tolerance)

    def _require_active(self) -> PickPlaceState:
        if self._active is None:
            raise RuntimeError("No active pick/place state")
        return self._active


@dataclass
class SkillRequest:
    """A request from the language layer, fulfilled only on the simulation thread."""

    payload: dict
    response: dict | None = None
    done: threading.Event | None = None


class LocalSkillServer:
    """Small localhost bridge; it deliberately never exposes the simulator to a network."""

    def __init__(self, port: int) -> None:
        self.requests: queue.Queue[SkillRequest] = queue.Queue()
        self._port = port
        self._ready = threading.Event()
        self._error: Exception | None = None

    def start(self) -> None:
        threading.Thread(target=self._serve, daemon=True).start()
        if not self._ready.wait(timeout=5):
            raise RuntimeError("skill server socket did not start")
        if self._error is not None:
            raise RuntimeError(f"skill server socket failed: {self._error}")

    def _serve(self) -> None:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind(("127.0.0.1", self._port))
                server.listen()
                self._ready.set()
                while True:
                    connection, _ = server.accept()
                    threading.Thread(target=self._handle, args=(connection,), daemon=True).start()
        except Exception as error:
            self._error = error
            self._ready.set()
            traceback.print_exc()

    def _handle(self, connection: socket.socket) -> None:
        with connection:
            try:
                raw = connection.recv(65536)
                # A TCP connect-only readiness probe has no request payload.
                if not raw:
                    return
                request = SkillRequest(json.loads(raw.decode()), done=threading.Event())
                print(f"[skill-server] received {request.payload.get('tool')}", flush=True)
                self.requests.put(request)
                if not request.done.wait(timeout=300):
                    result = {"ok": False, "code": "MOTION_TIMEOUT", "detail": "simulation service did not reply"}
                else:
                    result = request.response
            except Exception as error:
                traceback.print_exc()
                result = {"ok": False, "code": "INVALID_REQUEST", "detail": str(error)}
            try:
                connection.sendall((json.dumps(result) + "\n").encode())
                print(f"[skill-server] replied {result.get('code')}", flush=True)
            except OSError:
                pass


def add_target_box() -> None:
    """Author an open, static, collision-enabled box around the placement slots."""
    wall = 0.012
    inner = 0.18
    height = 0.09
    pieces = {
        "base": ([BOX_CENTER[0], BOX_CENTER[1], wall / 2], [inner + 2 * wall, inner + 2 * wall, wall]),
        "front": ([BOX_CENTER[0], BOX_CENTER[1] - inner / 2 - wall / 2, height / 2], [inner + 2 * wall, wall, height]),
        "back": ([BOX_CENTER[0], BOX_CENTER[1] + inner / 2 + wall / 2, height / 2], [inner + 2 * wall, wall, height]),
        "left": ([BOX_CENTER[0] - inner / 2 - wall / 2, BOX_CENTER[1], height / 2], [wall, inner, height]),
        "right": ([BOX_CENTER[0] + inner / 2 + wall / 2, BOX_CENTER[1], height / 2], [wall, inner, height]),
    }
    for name, (position, scale) in pieces.items():
        path = f"/World/TargetBox/{name}"
        Cube(paths=path, positions=position, sizes=1.0, scales=scale, colors=(0.35, 0.35, 0.35))
        GeomPrim(paths=path, apply_collision_apis=True)


def prepare_overhead_camera(focus_xy: np.ndarray | None = None) -> object:
    """Create the one shared overhead camera before servicing requests.

    Kit can close when a render product is authored lazily while a persistent
    physics session is already handling an RPC.  The server therefore creates
    its sensor during startup, then every capture only reads this render
    product.
    """
    from isaacsim.core.experimental.utils.transform import look_at_quaternion
    from isaacsim.sensors.camera import Camera
    # A high, near-vertical view covers the complete 2x2 source layout
    # (y=-0.30/-0.45) as well as the target box, rather than only its front row.
    eye = _OVERHEAD_EYE.copy()
    target = _OVERHEAD_TARGET.copy()
    if focus_xy is not None:
        # A failed contact can leave the target at the edge of the randomized
        # task-start view. Re-centre using only the previous visual estimate;
        # the new pickup point still comes exclusively from the fresh RGB-D
        # frame (never from simulator object state).
        focus = np.asarray(focus_xy, dtype=float)
        if focus.shape != (2,) or not np.isfinite(focus).all():
            raise ValueError("overhead recovery focus must be a finite XY point")
        offset = focus - target[:2]
        eye[:2] += offset
        target[:2] += offset
    global _OVERHEAD_CAMERA
    if _OVERHEAD_CAMERA is None:
        _OVERHEAD_CAMERA = Camera(
            prim_path="/World/AgentOverheadCamera",
            frequency=20,
            resolution=(640, 640),
        )
        _OVERHEAD_CAMERA.initialize()
        _OVERHEAD_CAMERA.set_clipping_range(near_distance=0.01, far_distance=10.0)
        _OVERHEAD_CAMERA.add_distance_to_image_plane_to_frame()
    camera = _OVERHEAD_CAMERA
    # ``look_at_quaternion`` uses the USD camera convention (-Z forward),
    # whereas Camera's constructor assumes ``world`` axes by default.  Set
    # the pose explicitly to prevent an incorrectly horizontal view.
    camera.set_world_pose(
        position=eye,
        orientation=look_at_quaternion(eye=eye, target=target).numpy(),
        camera_axes="usd",
    )
    return camera


def capture_overhead_rgb(
    path: Path,
    pose_path: Path | None = None,
    *,
    focus_xy: np.ndarray | None = None,
) -> None:
    """Render a visual observation without exposing any object pose to the Agent."""
    from PIL import Image

    trace_path = path.with_name("rgbd_capture_trace.log")

    def trace(message: str) -> None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")
            handle.flush()

    trace("capture entered")
    camera = prepare_overhead_camera(focus_xy)
    if focus_xy is not None:
        trace(f"recovery focus xy={np.asarray(focus_xy, dtype=float).tolist()}")
    trace("persistent overhead camera ready")
    # Let PhysX settle and the render product receive several frames.
    for _ in range(30):
        simulation_app.update()
    trace("render warmup complete")
    rgba = camera.get_rgba()
    if rgba is None:
        raise RuntimeError("Overhead camera did not produce an RGB frame")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgba[:, :, :3], dtype=np.uint8)).save(path)
    trace(f"rgb saved shape={np.asarray(rgba).shape}")
    pose_path = pose_path or path.with_name("latest_vision_poses.json")
    try:
        from vision_localizer import (
            localize_coloured_blocks_with_quality,
            localize_open_vocabulary_with_quality,
            save_poses,
        )

        trace(f"camera frame keys={list(camera.get_current_frame())}")
        grasps = None
        if args.perception_backend == "generic":
            instance_dir = path.with_name(f"{path.stem}_instances")
            poses, qualities, grasps = localize_open_vocabulary_with_quality(
                camera,
                semantic_vocabulary(),
                instance_dir,
                semantic_backend=args.semantic_backend,
                semantic_model=args.semantic_model,
                min_semantic_confidence=args.semantic_min_confidence,
                wait_callback=simulation_app.update,
            )
            source = "generic_rgbd"
            trace(f"generic instances saved: {instance_dir}")
        else:
            poses, qualities = localize_coloured_blocks_with_quality(camera)
            source = "local_rgbd"
        if not poses:
            raise RuntimeError("RGB-D perception found no valid object instances")
        save_poses(poses, pose_path, qualities, source=source, grasps=grasps)
        trace(f"poses saved: {sorted(poses)}")
    except Exception as error:
        import traceback

        diagnostic = path.with_name("rgbd_capture_error.log")
        diagnostic.write_text(traceback.format_exc())
        raise RuntimeError(f"RGB-D localization failed; see {diagnostic}") from error
    print(f"Saved local RGB-D pose estimates: {pose_path}")
    print(f"Saved overhead RGB observation: {path}")


def load_rgbd_poses(path: Path, targets: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Read and validate local RGB-D estimates before they reach the controller.

    This boundary deliberately accepts only local camera output. It makes a
    missing or malformed observation a perception failure rather than silently
    replacing it with the simulator's object pose.
    """
    payload = json.loads(path.read_text())
    if payload.get("frame") != "world" or payload.get("source") not in {"local_rgbd", "generic_rgbd"}:
        raise RuntimeError(f"{path} is not a local world-frame RGB-D pose file")
    poses: dict[str, np.ndarray] = {}
    for name, value in payload.get("poses", {}).items():
        if name not in COLORS:
            continue
        pose = np.asarray(value, dtype=float)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise RuntimeError(f"RGB-D estimate for {name} is not a finite XYZ point")
        # This is a workspace envelope, not physics truth. It rejects a
        # corrupted calibration/depth frame before the arm can move toward it.
        if not (0.20 <= pose[0] <= 0.80 and -0.65 <= pose[1] <= 0.35 and -0.02 <= pose[2] <= 0.20):
            raise RuntimeError(f"RGB-D estimate for {name} is outside the configured workspace: {pose.tolist()}")
        poses[name] = pose
    missing = [color for color in targets if color not in poses]
    if missing:
        raise RuntimeError(
            "RGB-D localization did not detect requested target(s): " + ", ".join(missing) + ". Refusing simulator-pose fallback."
        )
    return poses


def load_grasp_descriptors(path: Path, targets: tuple[str, ...]) -> dict[str, dict]:
    """Validate category-independent grasp geometry before motion control."""
    payload = json.loads(path.read_text())
    descriptors = payload.get("grasps", {})
    if payload.get("source") != "generic_rgbd":
        return {}
    result: dict[str, dict] = {}
    for object_id in targets:
        descriptor = descriptors.get(object_id)
        if not isinstance(descriptor, dict) or descriptor.get("feasible") is not True:
            raise RuntimeError(f"No feasible shape-aware grasp for {object_id}; refusing fixed-grasp fallback")
        yaw = descriptor.get("yaw_rad")
        width = descriptor.get("grasp_width_m")
        dimensions = np.asarray(descriptor.get("dimensions_xyz"), dtype=float)
        if (
            not isinstance(yaw, (int, float))
            or not np.isfinite(yaw)
            or not isinstance(width, (int, float))
            or not 0.010 <= float(width) <= 0.075
            or dimensions.shape != (3,)
            or not np.isfinite(dimensions).all()
        ):
            raise RuntimeError(f"Malformed shape-aware grasp descriptor for {object_id}")
        result[object_id] = descriptor
    return result


class ShapeAwarePickPlaceController(PickPlaceController):
    """Top-down controller whose tool yaw comes from RGB-D footprint PCA."""

    def __init__(self, *, grasp_descriptors: dict[str, dict], **kwargs) -> None:
        super().__init__(**kwargs)
        self._grasp_descriptors = grasp_descriptors
        self._grasp_object_id: str | None = None

    def _target_for_decision(self, observation, decision):
        self._grasp_object_id = decision.color
        return super()._target_for_decision(observation, decision)

    def _setpoint(self, position: np.ndarray):
        yaw = 0.0
        if self._grasp_object_id in self._grasp_descriptors:
            yaw = float(self._grasp_descriptors[self._grasp_object_id]["yaw_rad"])
        # q_z(yaw) * q_down: local-Y jaw closure follows the footprint's
        # minor axis while the tool remains vertically downward.
        orientation = [0.0, -np.sin(yaw / 2.0), np.cos(yaw / 2.0), 0.0]
        return stacking_module.mg.RobotState(
            sites=stacking_module.mg.SpatialState.from_name(
                spatial_space=self._site_space,
                positions=(
                    [self._tool_frame],
                    stacking_module.wp.array([position.tolist()], dtype=stacking_module.wp.float32),
                ),
                orientations=(
                    [self._tool_frame],
                    stacking_module.wp.array([orientation], dtype=stacking_module.wp.float32),
                ),
            )
        )


def make_runtime_controller(
    config: TaskConfig,
    policy,
    resources,
    grasp_descriptors: dict[str, dict] | None = None,
) -> PickPlaceController:
    if not grasp_descriptors:
        return make_pick_place_controller(config, policy, resources)
    return ShapeAwarePickPlaceController(
        config=config,
        policy=policy,
        arm_controller=resources.arm_controller,
        joint_space=resources.joint_space,
        site_space=resources.site_space,
        tool_frame=resources.tool_frame,
        grasp_descriptors=grasp_descriptors,
    )


def commanded_joint_positions(scene: FrankaStackingSceneIO, desired) -> np.ndarray:
    """Expand a possibly sparse controller command to the full Franka DOF vector."""
    articulation = getattr(scene, "_articulation", None)
    if articulation is None:
        raise RuntimeError("Franka articulation is unavailable while recording")
    command = np.asarray(articulation.get_dof_positions().numpy(), dtype=float).reshape(-1).copy()
    joints = None if desired is None else desired.joints
    positions = None if joints is None else joints.positions
    if positions is None:
        return command
    values = np.asarray(positions.numpy(), dtype=float).reshape(-1)
    indices = joints.position_indices
    if indices is None:
        if values.shape != command.shape:
            raise RuntimeError("Dense controller command does not match Franka DOF count")
        return values
    resolved = np.asarray(indices.numpy(), dtype=int).reshape(-1)
    if resolved.shape != values.shape:
        raise RuntimeError("Sparse controller command indices do not match values")
    command[resolved] = values
    return command


def capture_wrist_rgb(path: Path, scene: FrankaStackingSceneIO) -> None:
    """Capture an RGB-D frame from a calibrated wrist-following camera.

    This is deliberately a separate diagnostic command before enabling wrist
    measurements in closed-loop control.
    """
    from PIL import Image
    from isaacsim.sensors.camera import Camera

    trace_path = path.with_name("wrist_capture_trace.log")

    def trace(message: str) -> None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")
            handle.flush()

    trace("wrist capture entered")
    end_effector = wrist_mount_link(scene)
    mount_path = "/World/WristRGBDCamera"
    wrist_positions, wrist_orientations = end_effector.get_world_poses()
    wrist_position = np.asarray(wrist_positions.numpy()[0], dtype=float)
    wrist_pose = np.asarray(wrist_orientations.numpy()[0], dtype=float)
    eye, wrist_orientation = compose_pose(
        wrist_position,
        wrist_pose,
        np.asarray(WRIST_CAMERA_EXTRINSICS.optical_translation_m),
        np.asarray(WRIST_CAMERA_EXTRINSICS.camera_rotation_wxyz),
    )
    trace(f"wrist world position: {wrist_position.tolist()}")
    trace(f"fixed hand-eye camera position: {eye.tolist()}")
    camera = Camera(
        prim_path=mount_path,
        frequency=20,
        resolution=(480, 480),
    )
    trace("camera created")
    camera.set_world_pose(position=eye, orientation=wrist_orientation, camera_axes="usd")
    trace("camera pose set before initialization")
    camera.initialize()
    trace("camera initialized")
    # The wrist is only centimetres from a block; the USD default near clip is
    # too large and otherwise clips away the entire tabletop scene.
    camera.set_clipping_range(near_distance=0.01, far_distance=10.0)
    trace("wrist clipping range set to 0.01m..10m")
    # Camera.initialize may author/reset its USD xform, so apply the calibrated
    # wrist pose afterwards rather than before initialization.
    camera.set_world_pose(
        position=eye,
        orientation=wrist_orientation,
        camera_axes="usd",
    )
    actual_position, actual_orientation = camera.get_world_pose(camera_axes="usd")
    trace(f"camera aligned to fixed wrist extrinsics: pos={actual_position.tolist()} quat={actual_orientation.tolist()}")
    camera.add_distance_to_image_plane_to_frame()
    trace("depth annotator attached")
    for _ in range(30):
        simulation_app.update()
    trace("render warmup complete")
    rgba = camera.get_rgba()
    if rgba is None:
        raise RuntimeError("Wrist camera did not produce an RGB frame")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgba[:, :, :3], dtype=np.uint8)).save(path)
    trace(f"rgb saved shape={np.asarray(rgba).shape}")
    from vision_localizer import localize_coloured_blocks_with_quality, save_poses

    poses, qualities = localize_coloured_blocks_with_quality(camera)
    pose_path = path.with_name("latest_wrist_poses.json")
    save_poses(poses, pose_path, qualities)
    trace(f"poses saved: {sorted(poses)}")
    print(f"Saved wrist RGB-D observation: {path}")
    print(f"Saved wrist RGB-D pose estimates: {pose_path}")


class VisibleWristCameraMount:
    """Persistent RGB-D sensor rigidly mounted in the end-effector frame."""

    D455_ASSET = "/Isaac/Sensors/RealSense/D455/rsd455.usd"
    SENSOR_PATH = "/World/FrankaWristRGBDSensor"

    def __init__(
        self,
        scene: FrankaStackingSceneIO,
        *,
        visible_model: bool = True,
    ) -> None:
        from isaacsim.sensors.camera import Camera
        import omni.usd
        from pxr import UsdGeom

        self._scene = scene
        self._hand = wrist_mount_link(scene)
        self.root_path = f"{self._hand.paths[0]}/FrankaWristCamera"
        # Keep the renderer sensor at a world path. Isaac's camera unprojection
        # reads USD world transforms, while articulated link motion is held in
        # Fabric; parenting the Camera directly below panda_hand therefore
        # renders correctly but produces stale world points. The visible model
        # remains a true hand child, and update() synchronizes this sensor from
        # the same fixed extrinsics immediately before each rendered frame.
        self.camera_path = self.SENSOR_PATH
        self.model_path = f"{self.root_path}/RealSenseD455"
        stage = omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(stage, self.root_path)
        self._root = XformPrim(paths=self.root_path, reset_xform_op_properties=True)
        self._root.set_local_poses(translations=[[0.0, 0.0, 0.0]], orientations=[[1.0, 0.0, 0.0, 0.0]])
        self.camera = Camera(prim_path=self.camera_path, frequency=20, resolution=(480, 480))
        self.camera.initialize()
        self.camera.set_clipping_range(near_distance=0.01, far_distance=10.0)
        self.camera.add_distance_to_image_plane_to_frame()
        self._eye = np.zeros(3, dtype=float)
        self._model: XformPrim | None = None
        self._model_error: str | None = None
        try:
            if not visible_model:
                raise RuntimeError("visual housing disabled for persistent service")
            from isaacsim.core.experimental.utils.stage import add_reference_to_stage
            from isaacsim.storage.native import get_assets_root_path

            assets_root = get_assets_root_path()
            if not assets_root:
                raise RuntimeError("Isaac Sim assets root is unavailable")
            add_reference_to_stage(assets_root + self.D455_ASSET, self.model_path)
            # The D455 asset is also usable as a free-standing prop and ships
            # with physics APIs. Here it is a wrist-mounted *visual* asset, so
            # disable all descendant rigid bodies and collisions before the
            # timeline advances; otherwise PhysX makes it fall to the floor.
            from pxr import Usd, UsdPhysics

            model_root = stage.GetPrimAtPath(self.model_path)
            for prim in Usd.PrimRange(model_root):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Set(False)
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
            # The vendor USD root is authored without editable xformOps. Give
            # the referenced instance standard transform ops before following
            # Franka's wrist each simulation frame.
            self._model = XformPrim(paths=self.model_path, reset_xform_op_properties=True)
            self._model.set_local_poses(
                translations=[WRIST_CAMERA_EXTRINSICS.body_translation_m],
                orientations=[WRIST_CAMERA_EXTRINSICS.camera_rotation_wxyz],
            )
        except Exception as error:
            # Retain a simple visible fallback when the optional Nucleus asset
            # server is offline. The sensing pipeline still remains functional.
            self._model_error = str(error)
            self._housing = Cube(
                paths=f"{self.root_path}/RealSenseD455FallbackBody",
                positions=[0.0, 0.0, -1.0], sizes=1.0,
                scales=[0.090, 0.023, 0.020], colors=(0.08, 0.08, 0.08),
            )
            self._lens = Cube(
                paths=f"{self.root_path}/RealSenseD455FallbackLens",
                positions=[0.0, 0.0, -1.0], sizes=1.0,
                scales=[0.014, 0.026, 0.006], colors=(0.02, 0.02, 0.02),
            )
            self._housing.set_local_poses(
                translations=[WRIST_CAMERA_EXTRINSICS.body_translation_m],
                orientations=[WRIST_CAMERA_EXTRINSICS.camera_rotation_wxyz],
            )
            self._lens.set_local_poses(
                translations=[[0.060, 0.0, 0.049]],
                orientations=[WRIST_CAMERA_EXTRINSICS.camera_rotation_wxyz],
            )
        self.update()

    def update(self) -> np.ndarray:
        """Synchronize the world sensor from the fixed panda_hand extrinsics."""
        wrist_positions, wrist_orientations = self._hand.get_world_poses()
        wrist = np.asarray(wrist_positions.numpy()[0], dtype=float)
        wrist_orientation = np.asarray(wrist_orientations.numpy()[0], dtype=float)
        eye, orientation = compose_pose(
            wrist,
            wrist_orientation,
            np.asarray(WRIST_CAMERA_EXTRINSICS.optical_translation_m),
            np.asarray(WRIST_CAMERA_EXTRINSICS.camera_rotation_wxyz),
        )
        self.camera.set_world_pose(position=eye, orientation=orientation, camera_axes="usd")
        self._eye = eye
        return eye.copy()

    @property
    def eye(self) -> np.ndarray:
        return self._eye.copy()

    @property
    def display_model(self) -> str:
        return "official RealSense D455 USD" if self._model is not None else f"fallback housing ({self._model_error})"


class WristRGBDVisualServo:
    """Periodic, filtered wrist RGB-D feedback during the approach phase."""

    def __init__(
        self,
        scene: FrankaStackingSceneIO,
        mount: VisibleWristCameraMount,
        artifact_dir: Path | None = None,
        *,
        gate_enabled: bool = True,
        rate_hz: float = 10.0,
        alpha: float = 0.70,
        max_step_m: float = 0.008,
        min_range_m: float = 0.19,
        max_range_m: float = 0.30,
        max_updates_per_color: int | None = None,
        fault_config: dict | None = None,
        fault_seed: int = 0,
        perception_backend: str = "color",
    ) -> None:
        del scene
        self._mount = mount
        self._camera = mount.camera
        self._artifact_dir = artifact_dir or Path(__file__).resolve().parent / "observations"
        self._gate_enabled = gate_enabled
        self._sample_period = 1.0 / rate_hz
        self._alpha = alpha
        self._max_step_m = max_step_m
        self._min_range_m = min_range_m
        self._max_range_m = max_range_m
        self._max_updates_per_color = max_updates_per_color
        self._fault_config = fault_config or {}
        self._fault_rng = np.random.default_rng(fault_seed)
        self._perception_backend = perception_backend
        self._tracking_color: str | None = None
        self._anchor_positions: dict[str, np.ndarray] = {}
        self._aim_frames = 0
        self._next_sample_time = 0.0
        self._sample_counts: dict[str, int] = {}
        self._accepted_windows: dict[str, list[np.ndarray]] = {}
        self._consecutive_misses = 0

    def active_for(self, color: str) -> bool:
        """Whether this mode permits another accepted correction."""
        return (
            self._max_updates_per_color is None
            or len(self._accepted_windows.get(color, [])) < self._max_updates_per_color
        )

    def allow_retry(self, color: str) -> None:
        """Reset camera settling and sampling state after explicit recovery."""
        if self._tracking_color == color:
            self._tracking_color = None
        self._accepted_windows.pop(color, None)
        self._anchor_positions.pop(color, None)

    def track(self, color: str, target_hint: np.ndarray, time_s: float) -> np.ndarray | None:
        """Return the next bounded visual target, or None between observations."""
        from vision_localizer import (
            assess_refinement,
            bounded_servo_update,
            localize_coloured_blocks_with_quality,
            localize_nearest_instance_with_quality,
            save_poses,
        )

        if not self.active_for(color):
            return None
        if color != self._tracking_color:
            self._tracking_color = color
            # Keep a fixed reference for the whole approach. Comparing each
            # observation only with the preceding servo command permits a
            # sequence of individually small errors to drift arbitrarily far.
            self._anchor_positions[color] = np.asarray(target_hint, dtype=float).copy()
            self._aim_frames = 0
            self._next_sample_time = time_s
            self._consecutive_misses = 0
        self._aim_frames += 1
        # update() ran immediately before this rendered frame; use that exact
        # pose for range gating instead of changing the sensor after capture.
        eye = self._mount.eye
        view_range = float(np.linalg.norm(eye - np.asarray(target_hint, dtype=float)))
        # The camera can track the block throughout the motion, but depth and
        # segmentation cease to be reliable when the cube fills the frame near
        # grasp. Freeze the final trusted command outside this calibrated
        # observation window instead of feeding close-range artifacts back.
        if not self._min_range_m <= view_range <= self._max_range_m:
            return None
        # The first rendered frame after changing camera pose is stale. Let the
        # normal simulation loop render once instead of advancing physics from
        # inside the perception callback.
        if self._aim_frames < 2 or time_s + 1e-9 < self._next_sample_time:
            return None
        self._next_sample_time = time_s + self._sample_period
        sample_index = self._sample_counts.get(color, 0) + 1
        self._sample_counts[color] = sample_index
        dropout_probability = float(self._fault_config.get("wrist_dropout_probability", 0.0))
        if dropout_probability and self._fault_rng.random() < dropout_probability:
            self._consecutive_misses += 1
            print(
                f"[fault-injection] wrist frame dropped color={color} sample={sample_index}",
                flush=True,
            )
            print(
                f"[visual-servo] {color} rejected: injected frame dropout "
                f"(miss {self._consecutive_misses}); holding last safe target"
            )
            return None
        if self._perception_backend == "generic":
            estimate, quality = localize_nearest_instance_with_quality(self._camera, target_hint)
            poses = {color: estimate} if estimate is not None else {}
            qualities = {color: quality} if quality is not None else {}
            pose_source = "generic_rgbd"
        else:
            poses, qualities = localize_coloured_blocks_with_quality(self._camera)
            pose_source = "local_rgbd"
        from PIL import Image

        debug_dir = self._artifact_dir
        debug_dir.mkdir(parents=True, exist_ok=True)
        stem = f"wrist_{color}_servo_{sample_index:03d}"
        rgba = self._camera.get_rgba()
        if rgba is not None:
            image = Image.fromarray(np.asarray(rgba[:, :, :3], dtype=np.uint8))
            image.save(debug_dir / f"{stem}.png")
            image.save(debug_dir / f"wrist_{color}_pregrasp.png")
        save_poses(poses, debug_dir / f"{stem}_poses.json", qualities, source=pose_source)
        # Preserve the legacy latest-sample artifact used by older reports.
        save_poses(poses, debug_dir / f"wrist_{color}_pregrasp_poses.json", qualities, source=pose_source)
        estimate = poses.get(color)
        quality = qualities.get(color)
        if estimate is None or quality is None:
            self._consecutive_misses += 1
            print(
                f"[visual-servo] {color} rejected: not visible "
                f"(miss {self._consecutive_misses}); holding last safe target"
            )
            return None
        measurement = np.asarray(estimate, dtype=float)
        noise_std = float(self._fault_config.get("wrist_noise_std_m", 0.0))
        if noise_std:
            noise = self._fault_rng.normal(0.0, noise_std, size=3)
            measurement += noise
            poses[color] = measurement.tolist()
            save_poses(poses, debug_dir / f"{stem}_poses.json", qualities, source=pose_source)
            save_poses(poses, debug_dir / f"wrist_{color}_pregrasp_poses.json", qualities, source=pose_source)
            print(
                f"[fault-injection] wrist noise color={color} sample={sample_index} "
                f"delta_mm={np.round(noise * 1000.0, 2).tolist()}",
                flush=True,
            )
        current = np.asarray(target_hint, dtype=float)
        anchor = self._anchor_positions.setdefault(color, current.copy())
        accepted, reason = (True, "gate disabled for ablation")
        if self._gate_enabled:
            initial_bias = np.asarray(
                self._fault_config.get("initial_overhead_xy_bias_m", [0.0, 0.0]),
                dtype=float,
            )
            target_shift = self._fault_config.get("target_shift")
            allows_deliberate_displacement = (
                float(np.linalg.norm(initial_bias)) > 1e-9 or target_shift is not None
            )
            # Normal operation is anchored to the overhead estimate. Fault
            # scenarios that deliberately bias/move the object retain the
            # wider high-confidence recovery envelope used by the benchmark.
            max_anchor_offset = 0.030 if allows_deliberate_displacement else 0.012
            accepted, reason = assess_refinement(
                anchor,
                measurement,
                quality,
                max_xy_offset_m=0.012,
                max_high_confidence_xy_offset_m=max_anchor_offset,
            )
        if not accepted:
            self._consecutive_misses += 1
            print(f"[visual-servo] {color} rejected: {reason}; holding last safe target")
            return None
        self._consecutive_misses = 0
        window = self._accepted_windows.setdefault(color, [])
        window.append(measurement)
        if len(window) > 3:
            del window[0]
        robust_measurement = np.median(np.stack(window), axis=0)
        command, innovation, applied = bounded_servo_update(
            current,
            robust_measurement,
            alpha=self._alpha,
            max_step_m=self._max_step_m,
        )
        trace = {
            "time_s": float(time_s),
            "color": color,
            "sample": sample_index,
            "prior_command": current.tolist(),
            "overhead_anchor": anchor.tolist(),
            "measurement": measurement.tolist(),
            "median_measurement": robust_measurement.tolist(),
            "command": command.tolist(),
            "innovation_mm": innovation * 1000.0,
            "applied_step_mm": applied * 1000.0,
            "confidence": quality.confidence,
            "view_range_m": view_range,
            "accepted": True,
        }
        with (debug_dir / "visual_servo_trace.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace) + "\n")
        print(
            f"[visual-servo] {color} accepted: sample={sample_index} "
            f"innovation={innovation * 1000.0:.2f}mm step={applied * 1000.0:.2f}mm "
            f"command={np.round(command, 4).tolist()} ({reason})"
        )
        return command


class RecoveryRetreatPolicy:
    """Move the open gripper to a camera-clear pose before re-observation."""

    def __init__(self, config: TaskConfig, color: str) -> None:
        self._config = config
        self._color = color
        self._safe_position = np.array([0.30, 0.0, 0.50])

    def reset(self) -> None:
        pass

    def forward(
        self,
        observation: Observation,
        *,
        target_reached: bool,
        gripper_reached: bool,
    ) -> PickPlaceDecision:
        del observation, gripper_reached
        if target_reached:
            return PickPlaceDecision(done=True)
        # PRE_GRASP adds above_height to pickup_position. Cancel that offset
        # so the maintained controller receives the explicit safe pose.
        source = self._safe_position - np.array([0.0, 0.0, self._config.above_height])
        return PickPlaceDecision(
            phase=PickPlacePhase.PRE_GRASP,
            color=self._color,
            destination=self._safe_position,
            pickup_position=source,
        )


def main() -> None:
    global _ACTIVE_VLA_RECORDER
    # The persistent server always spawns every known object. Individual skills
    # select one object at a time, preserving the physics state between calls.
    targets = COLORS if args.server_port else resolve_targets(args.target)
    if args.vla_instruction and len(targets) != 1:
        raise ValueError("--vla-instruction supports exactly one resolved --target")
    if args.dagger_dataset and len(targets) != 1:
        raise ValueError("DAgger collection currently requires exactly one resolved --target")
    if args.dagger_dataset and args.object_set == "heldout" and args.perception_backend != "generic":
        raise ValueError("Held-out DAgger collection requires --perception-backend generic")
    scenario = load_benchmark_scenario(Path(args.scenario)) if args.scenario else None
    config = make_config(targets, None if scenario is None else scenario["spawn_positions"])
    if scenario is not None:
        global _OVERHEAD_EYE, _OVERHEAD_TARGET
        _OVERHEAD_EYE = scenario["camera_eye"]
        _OVERHEAD_TARGET = scenario["camera_target"]
    vision_poses: dict[str, np.ndarray] = {}
    vision_grasps: dict[str, dict] = {}
    if args.vision_poses:
        vision_poses = load_rgbd_poses(Path(args.vision_poses), targets)
        vision_grasps = load_grasp_descriptors(Path(args.vision_poses), targets)
    print(f"Requested: {', '.join(targets)}. Target box: {BOX_CENTER.tolist()}")

    SimulationManager.setup_simulation(dt=PHYSICS_DT, device="cpu")
    scene = (
        FrankaStackingSceneIO(config)
        if args.object_set == "cubes"
        else GeneralObjectSceneIO(config)
    )
    policy = TargetSelectionFSM(config, targets)
    simulation_app.run_coroutine(scene.setup())
    simulation_app.update()
    if scenario is not None:
        from isaacsim.core.experimental.objects import DomeLight

        DomeLight("/World/DomeLight").set_intensities(scenario["light_intensity"])
        print(
            "Benchmark scenario: "
            f"light={scenario['light_intensity']:.1f}, camera_eye={np.round(_OVERHEAD_EYE, 4).tolist()}"
        )
    add_target_box()
    simulation_app.update()

    if args.capture_path:
        capture_trace = Path(args.capture_path).expanduser().resolve().with_name("rgbd_capture_trace.log")
        capture_trace.parent.mkdir(parents=True, exist_ok=True)
        capture_trace.write_text(f"capture requested: {args.capture_path}\n", encoding="utf-8")
        # Use the same timeline/scene initialization sequence as the working
        # state-machine tutorial.  Initializing a render camera before these
        # handles exist can make Kit close silently on some Isaac Sim 6 builds.
        with capture_trace.open("a", encoding="utf-8") as trace:
            trace.write("starting timeline for capture\n")
            trace.flush()
        app_utils.play()
        simulation_app.update()
        with capture_trace.open("a", encoding="utf-8") as trace:
            trace.write("timeline started; initializing scene\n")
            trace.flush()
        scene.initialize()
        simulation_app.update()
        with capture_trace.open("a", encoding="utf-8") as trace:
            trace.write("scene initialized; creating camera\n")
            trace.flush()
        capture_overhead_rgb(Path(args.capture_path).expanduser().resolve())
        return

    if args.wrist_capture_path:
        wrist_path = Path(args.wrist_capture_path).expanduser().resolve()
        wrist_trace = wrist_path.with_name("wrist_capture_trace.log")
        wrist_error = wrist_path.with_name("wrist_capture_error.log")
        wrist_trace.parent.mkdir(parents=True, exist_ok=True)
        wrist_trace.write_text(f"wrist capture requested: {wrist_path}\n", encoding="utf-8")
        wrist_error.unlink(missing_ok=True)
        try:
            app_utils.play()
            wrist_trace.open("a", encoding="utf-8").write("timeline started\n")
            simulation_app.update()
            scene.initialize()
            wrist_trace.open("a", encoding="utf-8").write("scene initialized\n")
            simulation_app.update()
            capture_wrist_rgb(wrist_path, scene)
        except Exception:
            wrist_error.write_text(traceback.format_exc(), encoding="utf-8")
            raise
        return

    if args.preview_wrist_camera:
        # A dedicated non-autonomous preview is more useful than trying to
        # inspect the short-lived Isaac processes launched by agent_main.py.
        preview_log = Path(__file__).resolve().parent / "observations" / "realsense_preview.log"
        preview_log.parent.mkdir(parents=True, exist_ok=True)

        def preview_trace(message: str) -> None:
            print(f"[realsense-preview] {message}", flush=True)
            with preview_log.open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")

        preview_log.write_text("preview requested\n", encoding="utf-8")
        try:
            preview_trace("starting timeline")
            app_utils.play()
            simulation_app.update()
            preview_trace("initializing Franka scene")
            scene.initialize()
            simulation_app.update()
            preview_trace("creating RealSense D455 mount")
            preview_mount = VisibleWristCameraMount(scene)
            preview_trace(f"ready: {preview_mount.display_model}")
            print(
                "RealSense preview ready. In Stage expand "
                f"{preview_mount.model_path}. "
                "Close the Isaac Sim window to exit.",
                flush=True,
            )
            # The preview robot is static. Re-authoring a referenced vendor
            # asset every frame triggers expensive USD recomposition and can
            # make the desktop window report "Not responding".
            while simulation_app.is_running():
                simulation_app.update()
        except Exception:
            preview_trace("failed:\n" + traceback.format_exc())
            raise
        return

    server = LocalSkillServer(args.server_port) if args.server_port else None

    if args.headless and not args.test and not server and not args.rgbd:
        print("Headless: press Play in the livestream UI to begin.")
        initialized = False
        controller: PickPlaceController | None = None
    else:
        app_utils.play()
        simulation_app.update()
        scene.initialize()
        # Scene initialization may reset the timeline. A persistent headless
        # service must resume only after its physics handles exist.
        if server:
            app_utils.play()
            simulation_app.update()
        controller = (
            None
            if (server or (policy_checkpoint and not args.dagger_dataset))
            else make_runtime_controller(config, policy, scene.controller_resources(), vision_grasps)
        )
        initialized = True

    if args.rgbd:
        if server:
            raise RuntimeError("--rgbd is a one-shot task option; the persistent server captures before each pick")
        if not initialized:
            raise RuntimeError("--rgbd requires a running simulation; use --test with --headless")
        image_path = (
            Path(args.rgbd_output).expanduser().resolve()
            if args.rgbd_output
            else Path(__file__).resolve().parent / "observations" / "latest_overhead.png"
        )
        pose_path = image_path.with_name("latest_vision_poses.json")
        capture_overhead_rgb(image_path)
        if scenario is not None:
            apply_initial_overhead_fault(pose_path, scenario["faults"])
        vision_poses = load_rgbd_poses(pose_path, targets)
        vision_grasps = load_grasp_descriptors(pose_path, targets)
        controller = make_runtime_controller(config, policy, scene.controller_resources(), vision_grasps)
        print(f"RGB-D pickup poses accepted for: {', '.join(sorted(vision_poses))}")

    if server:
        # Pre-create the render product before accepting any RPC. See
        # prepare_overhead_camera() for why runtime camera creation is unsafe
        # in this long-lived Kit session.
        prepare_overhead_camera()
        print("Persistent overhead RGB-D camera ready", flush=True)

    # The generic persistent service is the Hybrid VLA execution boundary and
    # therefore keeps both overhead and wrist RGB-D views alive. The lighter
    # colour-only service retains the historical overhead-only path.
    wrist_mount: VisibleWristCameraMount | None = None
    if not server or args.perception_backend == "generic":
        try:
            wrist_mount = VisibleWristCameraMount(scene, visible_model=not bool(server))
            print(
                "Visible wrist RGB-D camera mounted: "
                f"{wrist_mount.model_path} ({wrist_mount.display_model})"
            )
        except Exception as error:
            print(f"[wrist-rgbd] mount unavailable: {error}")

    wrist_servo: WristRGBDVisualServo | None = None
    if (
        (vision_poses or server)
        and not policy_checkpoint
        and wrist_mount is not None
        and args.wrist_mode != "off"
    ):
        try:
            artifact_dir = Path(args.artifact_dir).expanduser().resolve() if args.artifact_dir else None
            wrist_servo = WristRGBDVisualServo(
                scene,
                wrist_mount,
                artifact_dir,
                gate_enabled=not args.disable_wrist_gate,
                rate_hz=args.visual_servo_rate_hz,
                alpha=args.visual_servo_alpha,
                max_step_m=args.visual_servo_max_step_mm / 1000.0,
                max_updates_per_color=1 if args.wrist_mode == "single" else None,
                fault_config=None if scenario is None else scenario["faults"],
                fault_seed=0 if scenario is None else scenario["fault_seed"],
                perception_backend=args.perception_backend,
            )
            print(
                f"Wrist RGB-D mode={args.wrist_mode} ready: "
                f"{args.visual_servo_rate_hz:g}Hz, alpha={args.visual_servo_alpha:g}, "
                f"max_step={args.visual_servo_max_step_mm:g}mm"
            )
        except Exception as error:
            print(f"[visual-servo] unavailable; using overhead RGB-D only: {error}")

    act_runtime: ACTSidecarPolicy | None = None
    act_overhead_camera = None
    act_hand = None
    runtime_artifact_dir = (
        Path(args.artifact_dir).expanduser().resolve()
        if args.artifact_dir
        else Path("observations").resolve()
    )
    if policy_checkpoint:
        if not initialized or wrist_mount is None:
            raise RuntimeError("ACT rollout requires an initialized scene and calibrated wrist camera")
        articulation = getattr(scene, "_articulation", None)
        if articulation is None:
            raise RuntimeError("ACT rollout requires an initialized Franka articulation")
        joint_names = tuple(articulation.dof_names)
        if joint_names != PANDA_JOINT_NAMES:
            raise RuntimeError(
                f"ACT joint order mismatch: expected {PANDA_JOINT_NAMES}, got {joint_names}"
            )
        act_overhead_camera = prepare_overhead_camera()
        act_hand = wrist_mount_link(scene)
        act_runtime = ACTSidecarPolicy(
            policy_checkpoint,
            device="cuda",
            control_hz=args.act_control_hz,
            action_steps=args.act_action_steps,
            max_arm_step=args.act_max_arm_step,
            inference_seed=args.policy_seed,
            adaptive_reobserve=args.act_adaptive_reobserve,
            precision_action_steps=args.act_precision_action_steps,
        )
        print(
            f"Learned-policy rollout ready: type={act_runtime.policy_type}, "
            f"checkpoint={act_runtime.checkpoint}, rate={args.act_control_hz:g}Hz, "
            f"reobserve_every={args.act_action_steps} actions, "
            f"adaptive={args.act_adaptive_reobserve}",
            flush=True,
        )

    dagger_recorder: RawVLAEpisodeRecorder | None = None
    if args.dagger_dataset:
        if act_runtime is None or not act_runtime.language_conditioned:
            raise RuntimeError("DAgger collection requires a language-conditioned policy")
        if controller is None or act_overhead_camera is None or act_hand is None or wrist_mount is None:
            raise RuntimeError("DAgger collection requires both learned and shadow-expert controllers")
        articulation = getattr(scene, "_articulation")
        dagger_instruction = args.vla_instruction or language_goal_instruction(targets[0])
        dagger_recorder = RawVLAEpisodeRecorder(
            args.dagger_dataset,
            fps=args.act_control_hz,
            instruction=dagger_instruction,
            targets=targets,
            object_set=args.object_set,
            joint_names=list(articulation.dof_names),
            dataset_kind="dagger_correction",
            metadata_extra={
                "dagger_round": args.dagger_round,
                "beta": args.dagger_beta,
                "intervention_threshold": args.dagger_intervention_threshold,
                "student_checkpoint": str(act_runtime.checkpoint),
                "student_policy_type": act_runtime.policy_type,
                "teacher_source": "generic_rgbd_shape_aware_shadow_expert",
                "label_action": "teacher_action",
            },
        )
        _ACTIVE_VLA_RECORDER = dagger_recorder
        # Build the teacher from locally observed geometry rather than giving
        # it simulator object poses. This also supplies the PCA-derived grasp
        # yaw required by held-out irregular objects.
        dagger_teacher_image = dagger_recorder.episode_dir / "teacher_initial_overhead.png"
        dagger_teacher_poses = dagger_teacher_image.with_name("teacher_initial_vision_poses.json")
        capture_overhead_rgb(dagger_teacher_image, dagger_teacher_poses)
        vision_poses = load_rgbd_poses(dagger_teacher_poses, targets)
        vision_grasps = load_grasp_descriptors(dagger_teacher_poses, targets)
        policy = TargetSelectionFSM(config, targets)
        controller = make_runtime_controller(
            config, policy, scene.controller_resources(), vision_grasps
        )
        print(
            f"DAgger recorder ready: {dagger_recorder.episode_dir} "
            f"(round={args.dagger_round}, beta={args.dagger_beta:g}, teacher=shadow expert)",
            flush=True,
        )

    recorder: RawVLAEpisodeRecorder | None = None
    record_overhead_camera = None
    record_hand = None
    recording_camera_warning = False
    if args.record_dataset:
        if not initialized or controller is None:
            raise RuntimeError("VLA recording requires an initialized expert controller")
        if wrist_mount is None:
            raise RuntimeError("VLA recording requires the calibrated wrist camera")
        articulation = getattr(scene, "_articulation", None)
        if articulation is None:
            raise RuntimeError("VLA recording requires an initialized Franka articulation")
        record_overhead_camera = prepare_overhead_camera()
        record_hand = wrist_mount_link(scene)
        instruction = args.record_instruction or (
            "Place " + ", ".join(targets) + " into the target box"
        )
        recorder = RawVLAEpisodeRecorder(
            args.record_dataset,
            fps=args.record_fps,
            instruction=instruction,
            targets=targets,
            object_set=args.object_set,
            joint_names=list(articulation.dof_names),
            episode_index=args.record_episode_index,
        )
        _ACTIVE_VLA_RECORDER = recorder
        print(
            f"VLA recorder ready: {recorder.episode_dir} "
            f"({args.record_fps:g}Hz, overhead RGB-D + wrist RGB-D)",
            flush=True,
        )

    def apply_expert_step(
        active_controller: PickPlaceController,
        estimated_state,
        current_observation: Observation,
        time_s: float,
    ) -> None:
        """Run the expert once and atomically pair its observation with its command."""
        nonlocal recording_camera_warning
        desired = active_controller.forward(
            estimated_state, None, time_s, observation=current_observation
        )
        if recorder is not None and recorder.due(time_s):
            assert record_overhead_camera is not None and wrist_mount is not None and record_hand is not None
            try:
                from vision_localizer import _read_rgbd

                overhead_rgb, overhead_depth = _read_rgbd(record_overhead_camera)
                wrist_rgb, wrist_depth = _read_rgbd(wrist_mount.camera)
                articulation = getattr(scene, "_articulation")
                joint_position = np.asarray(
                    articulation.get_dof_positions().numpy(), dtype=float
                ).reshape(-1)
                joint_velocity = np.asarray(
                    articulation.get_dof_velocities().numpy(), dtype=float
                ).reshape(-1)
                hand_positions, hand_orientations = record_hand.get_world_poses()
                policy_state = getattr(getattr(active_controller, "_policy", None), "_active", None)
                phase = None
                object_id = None
                if policy_state is not None:
                    phase_value = getattr(policy_state, "phase", None)
                    phase = getattr(phase_value, "name", str(phase_value)) if phase_value is not None else None
                    object_id = getattr(policy_state, "color", None)
                recorder.add_frame(
                    time_s=time_s,
                    overhead_rgb=overhead_rgb,
                    overhead_depth_m=overhead_depth,
                    wrist_rgb=wrist_rgb,
                    wrist_depth_m=wrist_depth,
                    joint_position=joint_position,
                    joint_velocity=joint_velocity,
                    end_effector_position=np.asarray(hand_positions.numpy()[0], dtype=float),
                    end_effector_quaternion_wxyz=np.asarray(hand_orientations.numpy()[0], dtype=float),
                    gripper_width=2.0 * float(current_observation.gripper_position),
                    joint_position_target=commanded_joint_positions(scene, desired),
                    eef_position_target=getattr(active_controller, "_target", None),
                    gripper_open_target=getattr(active_controller, "_gripper_open", None),
                    phase=phase,
                    object_id=object_id,
                    object_positions={
                        name: cube.get_world_poses()[0].numpy()[0]
                        for name, cube in getattr(scene, "_cubes", {}).items()
                    },
                    done=bool(active_controller.done),
                )
            except RuntimeError as error:
                # Render products can need one extra frame immediately after
                # creation. Do not lose an otherwise valid expert episode.
                if not recording_camera_warning:
                    print(f"[vla-recorder] waiting for camera frames: {error}", flush=True)
                    recording_camera_warning = True
        scene.apply(desired)

    # Do not accept TCP requests until the articulation, physics tensors and
    # scene handles are ready.  Advertising the socket earlier lets the client
    # submit a camera request during Kit startup, which can close the app.
    if server:
        server.start()
        print(f"SKILL_SERVER_READY {args.server_port}", flush=True)

    # A persistent service has just built a fresh scene. Resetting the
    # articulation again in its first simulation tick races Kit's initial
    # physics/tensor setup and can terminate the app as the first RPC arrives.
    # Individual move requests reset their newly-created controller instead.
    needs_reset = not bool(server)
    frame_count = 0
    active_request: SkillRequest | None = None
    recovery_attempts = 0
    recovery_stage: str | None = None
    recovery_failed_color: str | None = None
    target_shift_applied = False
    act_started_at: float | None = None
    act_goal_since: float | None = None
    act_goal_index = 0
    act_current_goal = targets[0]
    act_last_logged_inference = -1
    act_last_saved_inference = -1
    act_telemetry: PolicyRolloutTelemetry | None = None

    def reply(request: SkillRequest, body: dict) -> None:
        request.response = body
        assert request.done is not None
        request.done.set()

    def state(observation: Observation) -> dict:
        objects = []
        in_box: list[tuple[float, str]] = []
        for color in COLORS:
            position = observation.cube_positions[color]
            inside = bool(np.linalg.norm(position[:2] - BOX_CENTER) < 0.10 and position[2] > config.cube_size * 0.35)
            objects.append({"id": color, "position": [round(float(value), 4) for value in position], "location": "target_box" if inside else "table"})
            if inside:
                in_box.append((float(position[2]), color))
        in_box.sort()
        return {"objects": objects, "target_box": {"contents_bottom_to_top": [color for _, color in in_box]}, "robot": {"holding": None, "state": "busy" if active_request else "idle"}}

    # A headless/livestream Kit session may report ``is_running() == False``
    # when it has no desktop viewport, even though physics and RPC requests
    # are usable.  The persistent agent service therefore owns its lifetime;
    # the parent tool terminates it explicitly after the task ends.
    while server is not None or simulation_app.is_running():
        # Author the fixed hand-eye sensor pose before Kit renders this frame.
        # This keeps RGB/depth and the transform used for unprojection aligned.
        if wrist_mount is not None:
            wrist_mount.update()
        simulation_app.update()
        # In headless Kit the initial play request can be deferred by a few
        # frames. Keep requesting it while the persistent skill server waits.
        if server and not app_utils.is_playing():
            app_utils.play()
        if app_utils.is_playing() and SimulationManager.is_simulating():
            if not initialized:
                scene.initialize()
                controller = make_runtime_controller(config, policy, scene.controller_resources(), vision_grasps)
                initialized = True
            time = SimulationManager.get_simulation_time()
            if needs_reset:
                scene.reset()
                estimated, observation = scene.read(time)
                if act_runtime is not None:
                    act_runtime.reset(time)
                    act_started_at = time
                    act_goal_since = None
                    act_telemetry = PolicyRolloutTelemetry(
                        runtime_artifact_dir,
                        {color: observation.cube_positions[color] for color in targets},
                    )
                    if dagger_recorder is not None:
                        if controller is None or not controller.reset(estimated, None, time):
                            raise RuntimeError("DAgger shadow expert reset failed")
                    scene.apply(None)
                elif controller is not None:
                    if not controller.reset(estimated, None, time):
                        raise RuntimeError("Controller reset failed")
                    apply_expert_step(controller, estimated, observation, time)
                needs_reset = False
                continue
            estimated, observation = scene.read(time)
            if act_runtime is not None:
                assert act_overhead_camera is not None and wrist_mount is not None and act_hand is not None
                assert act_started_at is not None
                assert act_telemetry is not None
                act_telemetry.update(
                    observation.end_effector_position,
                    {color: observation.cube_positions[color] for color in targets},
                )
                desired = None
                teacher_desired = None
                if dagger_recorder is not None:
                    assert controller is not None
                    teacher_observation = observation
                    teacher_policy = getattr(controller, "_policy", None)
                    teacher_active = getattr(teacher_policy, "_active", None)
                    use_teacher_vision = teacher_active is None or getattr(
                        teacher_active, "phase", None
                    ) in {
                        PickPlacePhase.PRE_GRASP,
                        PickPlacePhase.APPROACH,
                        PickPlacePhase.GRASP,
                    }
                    if use_teacher_vision and vision_poses:
                        perceived_positions = dict(observation.cube_positions)
                        perceived_positions.update(vision_poses)
                        teacher_observation = Observation(
                            sample_id=observation.sample_id,
                            time=observation.time,
                            end_effector_position=observation.end_effector_position,
                            gripper_position=observation.gripper_position,
                            cube_positions=perceived_positions,
                        )
                    # Shadow-query the expert from the exact physical state
                    # visited by the student/mixed policy. The command is not
                    # applied until the DAgger mixer below selects it.
                    teacher_desired = controller.forward(
                        estimated, None, time, observation=teacher_observation
                    )
                # Let both render products refresh after the scene reset before
                # the first learned-policy observation.
                if time - act_started_at >= 0.5 and act_runtime.due(time):
                    from vision_localizer import _read_rgbd

                    overhead_rgb, overhead_depth = _read_rgbd(act_overhead_camera)
                    wrist_rgb, wrist_depth = _read_rgbd(wrist_mount.camera)
                    articulation = getattr(scene, "_articulation")
                    joint_position = np.asarray(
                        articulation.get_dof_positions().numpy(), dtype=float
                    ).reshape(-1)
                    joint_velocity = np.asarray(
                        articulation.get_dof_velocities().numpy(), dtype=float
                    ).reshape(-1)
                    hand_positions, hand_orientations = act_hand.get_world_poses()
                    policy_state = build_robot_state(
                        joint_position,
                        joint_velocity,
                        np.asarray(hand_positions.numpy()[0], dtype=float),
                        np.asarray(hand_orientations.numpy()[0], dtype=float),
                        2.0 * float(observation.gripper_position),
                    )
                    if act_runtime.state_dim == 30:
                        goal_one_hot = np.zeros(len(ACT_GOAL_COLORS), dtype=np.float32)
                        goal_one_hot[ACT_GOAL_COLORS.index(act_current_goal)] = 1.0
                        policy_state = np.concatenate((policy_state, goal_one_hot))
                    elif act_runtime.state_dim != 26:
                        raise RuntimeError(
                            f"Unsupported policy state dimension: {act_runtime.state_dim}"
                        )
                    instruction = None
                    if act_runtime.language_conditioned:
                        instruction = args.vla_instruction or language_goal_instruction(
                            act_current_goal
                        )
                    action = act_runtime.command(
                        time,
                        overhead_rgb=overhead_rgb,
                        wrist_rgb=wrist_rgb,
                        state=policy_state,
                        current_joint_position=joint_position,
                        instruction=instruction,
                    )
                    if action is not None:
                        act_telemetry.record_command(
                            time_s=time - act_started_at,
                            goal=act_current_goal,
                            inference_count=act_runtime.inference_count,
                            end_effector_position=np.asarray(
                                hand_positions.numpy()[0], dtype=float
                            ),
                            gripper_width_m=2.0 * float(observation.gripper_position),
                            diagnostics=act_runtime.last_command_diagnostics,
                        )
                        if (
                            args.act_telemetry_images_every > 0
                            and act_runtime.inference_count != act_last_saved_inference
                            and (
                                act_runtime.inference_count == 1
                                or act_runtime.inference_count
                                % args.act_telemetry_images_every == 0
                            )
                        ):
                            save_policy_observation(
                                runtime_artifact_dir,
                                act_runtime.inference_count,
                                overhead_rgb=overhead_rgb,
                                overhead_depth_m=overhead_depth,
                                wrist_rgb=wrist_rgb,
                                wrist_depth_m=wrist_depth,
                            )
                            act_last_saved_inference = act_runtime.inference_count
                        executed_action = action
                        intervention_reason = None
                        expert_intervened = False
                        teacher_action = None
                        if dagger_recorder is not None:
                            assert controller is not None
                            teacher_action = commanded_joint_positions(scene, teacher_desired)
                            disagreement = float(
                                np.linalg.norm(action[:7] - teacher_action[:7])
                            )
                            if disagreement > args.dagger_intervention_threshold:
                                executed_action = teacher_action
                                expert_intervened = True
                                intervention_reason = "action_divergence"
                            else:
                                executed_action = (
                                    args.dagger_beta * teacher_action
                                    + (1.0 - args.dagger_beta) * action
                                )
                                expert_intervened = args.dagger_beta > 0.0
                                intervention_reason = (
                                    "scheduled_beta" if expert_intervened else None
                                )
                            executed_action = safety_filter_action(
                                executed_action,
                                joint_position,
                                max_arm_step=args.act_max_arm_step,
                            )
                            # Avoid half-open gripper targets when blending two
                            # discrete grasp phases.
                            if args.dagger_beta >= 0.5 or intervention_reason == "action_divergence":
                                executed_action[7:] = teacher_action[7:]

                            teacher_policy_state = getattr(
                                getattr(controller, "_policy", None), "_active", None
                            )
                            teacher_phase = None
                            teacher_object = act_current_goal
                            if teacher_policy_state is not None:
                                phase_value = getattr(teacher_policy_state, "phase", None)
                                teacher_phase = (
                                    getattr(phase_value, "name", str(phase_value))
                                    if phase_value is not None
                                    else None
                                )
                                teacher_object = getattr(
                                    teacher_policy_state, "color", act_current_goal
                                )
                            dagger_recorder.add_frame(
                                time_s=time,
                                overhead_rgb=overhead_rgb,
                                overhead_depth_m=overhead_depth,
                                wrist_rgb=wrist_rgb,
                                wrist_depth_m=wrist_depth,
                                joint_position=joint_position,
                                joint_velocity=joint_velocity,
                                end_effector_position=np.asarray(
                                    hand_positions.numpy()[0], dtype=float
                                ),
                                end_effector_quaternion_wxyz=np.asarray(
                                    hand_orientations.numpy()[0], dtype=float
                                ),
                                gripper_width=2.0 * float(observation.gripper_position),
                                joint_position_target=teacher_action,
                                student_action=action,
                                executed_action=executed_action,
                                beta=args.dagger_beta,
                                expert_intervened=expert_intervened,
                                intervention_reason=intervention_reason,
                                label_valid=teacher_phase is not None,
                                eef_position_target=getattr(controller, "_target", None),
                                gripper_open_target=getattr(controller, "_gripper_open", None),
                                phase=teacher_phase,
                                object_id=teacher_object,
                                object_positions={
                                    name: cube.get_world_poses()[0].numpy()[0]
                                    for name, cube in getattr(scene, "_cubes", {}).items()
                                },
                                done=bool(controller.done),
                            )
                        desired = stacking_module.mg.RobotState(
                            joints=stacking_module.mg.JointState.from_name(
                                robot_joint_space=list(PANDA_JOINT_NAMES),
                                positions=(
                                    list(PANDA_JOINT_NAMES),
                                    stacking_module.wp.array(
                                        executed_action.tolist(), dtype=stacking_module.wp.float32
                                    ),
                                ),
                            )
                        )
                        if (
                            act_runtime.inference_count != act_last_logged_inference
                            and (act_runtime.inference_count == 1 or act_runtime.inference_count % 10 == 0)
                        ):
                            print(
                                f"[policy:{act_runtime.policy_type}] t={time - act_started_at:.1f}s "
                                f"goal={act_current_goal} instruction={instruction!r} "
                                f"inferences={act_runtime.inference_count}"
                                + (
                                    f" dagger_beta={args.dagger_beta:g} intervention={intervention_reason}"
                                    if dagger_recorder is not None
                                    else ""
                                ),
                                flush=True,
                            )
                            act_last_logged_inference = act_runtime.inference_count
                scene.apply(desired)
                frame_count += 1

                placed = {
                    color: bool(
                        np.linalg.norm(observation.cube_positions[color][:2] - BOX_CENTER) < 0.10
                        and observation.cube_positions[color][2] > TARGET_BOX_BASE_TOP_M
                    )
                    for color in targets
                }
                goal_conditioned = act_runtime.state_dim == 30 or act_runtime.language_conditioned
                goal_reached = placed[act_current_goal] if goal_conditioned else all(placed.values())
                if goal_reached:
                    act_goal_since = time if act_goal_since is None else act_goal_since
                    if time - act_goal_since >= 0.5:
                        if goal_conditioned and act_goal_index + 1 < len(targets):
                            print(
                                f"[policy:{act_runtime.policy_type}] verified goal={act_current_goal}; placed="
                                f"{[color for color, inside in placed.items() if inside]}",
                                flush=True,
                            )
                            act_goal_index += 1
                            act_current_goal = targets[act_goal_index]
                            act_goal_since = None
                            act_last_logged_inference = -1
                            act_runtime.reset(time)
                            continue
                        print(
                            f"[policy:{act_runtime.policy_type}] closed-loop placement verified after "
                            f"{time - act_started_at:.1f}s and {act_runtime.inference_count} observations",
                            flush=True,
                        )
                        print("Success: requested blocks placed in target box.")
                        telemetry_summary = act_telemetry.finalize(
                            success=True,
                            placed=placed,
                            runtime=act_runtime,
                            elapsed_s=time - act_started_at,
                        )
                        print(
                            f"[policy-diagnostics] clipped_rate="
                            f"{telemetry_summary['clipped_command_rate']:.3f} "
                            f"summary={act_telemetry.summary_path}",
                            flush=True,
                        )
                        if dagger_recorder is not None:
                            dagger_recorder.finalize(success=True)
                        break
                else:
                    act_goal_since = None
                if (
                    dagger_recorder is not None
                    and controller is not None
                    and controller.done
                    and not goal_reached
                ):
                    reason = getattr(getattr(controller, "_policy", None), "failure_reason", None)
                    detail = "shadow expert terminated before verified placement"
                    if reason:
                        detail += f": {reason}"
                    dagger_recorder.finalize(success=False, failure_reason=detail)
                    raise RuntimeError(f"DAgger collection stopped: {detail}")
                if time - act_started_at >= args.act_timeout:
                    telemetry_summary = act_telemetry.finalize(
                        success=False,
                        placed=placed,
                        runtime=act_runtime,
                        elapsed_s=time - act_started_at,
                    )
                    target_stage = telemetry_summary["targets"][act_current_goal][
                        "failure_stage"
                    ]
                    print(
                        f"[policy-diagnostics] failure_stage={target_stage} "
                        f"clipped_rate={telemetry_summary['clipped_command_rate']:.3f} "
                        f"summary={act_telemetry.summary_path}",
                        flush=True,
                    )
                    if dagger_recorder is not None:
                        dagger_recorder.finalize(
                            success=False,
                            failure_reason=f"rollout timeout after {args.act_timeout:g}s",
                        )
                    raise RuntimeError(
                        f"Learned-policy rollout timed out after {args.act_timeout:g}s; "
                        f"goal={act_current_goal}, inferences={act_runtime.inference_count}, "
                        f"placed={[color for color, inside in placed.items() if inside]}, "
                        f"positions={{{', '.join(f'{color}: {np.round(observation.cube_positions[color], 3).tolist()}' for color in targets)}}}"
                    )
                continue
            task_policy = policy if isinstance(policy, TargetSelectionFSM) else None
            shift_fault = None if scenario is None else scenario["faults"].get("target_shift")
            if (
                not target_shift_applied
                and shift_fault is not None
                and task_policy is not None
                and task_policy._active is not None
                and task_policy._active.color == shift_fault["color"]
                and task_policy._active.phase is PickPlacePhase.APPROACH
            ):
                inject_target_shift(scene, observation, shift_fault)
                target_shift_applied = True
            servo_active = bool(
                wrist_servo is not None
                and task_policy is not None
                and task_policy._active is not None
                and wrist_servo.active_for(task_policy._active.color)
                and task_policy._active.phase in {
                    PickPlacePhase.PRE_GRASP,
                    PickPlacePhase.APPROACH,
                }
            )
            # During approach, the fixed camera emits bounded corrections at
            # the configured sensor rate; it no longer swivels toward targets.
            if servo_active:
                assert wrist_servo is not None and task_policy is not None and task_policy._active is not None
                color = task_policy._active.color
                if color in vision_poses:
                    corrected = wrist_servo.track(color, vision_poses[color], time)
                    if corrected is not None:
                        vision_poses[color] = corrected
            # Only use RGB-D estimates while approaching/grasping. Once held,
            # safety/drop checks use measured simulator state; final success is
            # always evaluated from physics, never from this estimate.
            use_vision = task_policy is not None and (
                task_policy._active is None
                or task_policy._active.phase in {
                    PickPlacePhase.PRE_GRASP, PickPlacePhase.APPROACH, PickPlacePhase.GRASP
                }
            )
            if use_vision and vision_poses:
                perceived_positions = dict(observation.cube_positions)
                perceived_positions.update(vision_poses)
                observation = Observation(
                    sample_id=observation.sample_id,
                    time=observation.time,
                    end_effector_position=observation.end_effector_position,
                    gripper_position=observation.gripper_position,
                    cube_positions=perceived_positions,
                )
            if server and active_request is None and controller is None:
                try:
                    request = server.requests.get_nowait()
                except queue.Empty:
                    request = None
                if request:
                    operation = request.payload.get("tool")
                    print(f"[skill-server] executing {operation}", flush=True)
                    if operation == "get_scene_state":
                        reply(request, {"ok": True, "code": "OK", "detail": "state read", "data": state(observation)})
                    elif operation == "capture_rgbd":
                        capture_path = request.payload.get("capture_path")
                        if not isinstance(capture_path, str) or not capture_path:
                            reply(request, {"ok": False, "code": "INVALID_ARGUMENT", "detail": "capture_path is required", "data": {}})
                        elif active_request is not None:
                            reply(request, {"ok": False, "code": "ROBOT_BUSY", "detail": "cannot capture while a motion is active", "data": {}})
                        else:
                            try:
                                capture_file = Path(capture_path).expanduser().resolve()
                                capture_overhead_rgb(capture_file)
                                payload = json.loads(capture_file.with_name("latest_vision_poses.json").read_text())
                                vision_poses = {
                                    name: np.asarray(value, dtype=float)
                                    for name, value in payload.get("poses", {}).items()
                                    if name in COLORS
                                }
                                vision_grasps = {
                                    name: value
                                    for name, value in payload.get("grasps", {}).items()
                                    if name in COLORS and isinstance(value, dict)
                                }
                                reply(request, {
                                    "ok": True,
                                    "code": "OK",
                                    "detail": "current RGB-D observation captured",
                                    "data": {"detected": sorted(vision_poses), "image_path": str(capture_file)},
                                })
                            except Exception as error:
                                traceback.print_exc()
                                reply(request, {"ok": False, "code": "PERCEPTION_FAILED", "detail": str(error), "data": {}})
                    elif operation == "capture_wrist_rgbd":
                        capture_path = request.payload.get("capture_path")
                        if not isinstance(capture_path, str) or not capture_path:
                            reply(request, {"ok": False, "code": "INVALID_ARGUMENT", "detail": "capture_path is required", "data": {}})
                        elif wrist_mount is None:
                            reply(request, {"ok": False, "code": "WRIST_CAMERA_UNAVAILABLE", "detail": "persistent wrist camera is not enabled", "data": {}})
                        else:
                            try:
                                from PIL import Image
                                from vision_localizer import _read_rgbd

                                capture_file = Path(capture_path).expanduser().resolve()
                                capture_file.parent.mkdir(parents=True, exist_ok=True)
                                wrist_rgb, wrist_depth = _read_rgbd(wrist_mount.camera)
                                Image.fromarray(np.asarray(wrist_rgb[..., :3], dtype=np.uint8)).save(capture_file)
                                depth_path = capture_file.with_name(capture_file.stem + "_depth.npy")
                                np.save(depth_path, np.asarray(wrist_depth, dtype=np.float32))
                                reply(request, {
                                    "ok": True,
                                    "code": "OK",
                                    "detail": "current wrist RGB-D observation captured",
                                    "data": {
                                        "image_path": str(capture_file),
                                        "depth_path": str(depth_path),
                                    },
                                })
                            except Exception as error:
                                traceback.print_exc()
                                reply(request, {"ok": False, "code": "PERCEPTION_FAILED", "detail": str(error), "data": {}})
                    elif operation == "reset_scene":
                        needs_reset = True
                        reply(request, {"ok": True, "code": "OK", "detail": "scene reset queued", "data": {}})
                    elif operation == "move_object_to_box":
                        color, level = request.payload.get("object_id"), request.payload.get("stack_level")
                        if color not in COLORS or not isinstance(level, int) or not 0 <= level < len(COLORS):
                            reply(request, {"ok": False, "code": "INVALID_ARGUMENT", "detail": "object_id must be a known color and stack_level must be 0..3", "data": {}})
                        elif state(observation)["objects"][COLORS.index(color)]["location"] == "target_box":
                            reply(request, {"ok": True, "code": "ALREADY_AT_GOAL", "detail": f"{color} is already in target_box", "data": state(observation)})
                        elif len(state(observation)["target_box"]["contents_bottom_to_top"]) != level:
                            reply(request, {"ok": False, "code": "STACK_PRECONDITION_FAILED", "detail": f"stack_level {level} requires exactly {level} verified blocks already in target_box", "data": state(observation)})
                        else:
                            # TaskConfig freezes its mappings by design. Build
                            # a per-request immutable configuration instead of
                            # mutating the shared scene configuration.
                            stack_positions = {
                                name: position.copy() for name, position in config.stack_positions.items()
                            }
                            if args.object_set == "heldout":
                                object_height = object_specs(args.object_set)[color].dimensions_xyz[2]
                                slot_xy = BOX_CENTER + TARGET_BOX_PACKING_OFFSETS[level]
                                stack_positions[color] = np.array([
                                    slot_xy[0], slot_xy[1],
                                    TARGET_BOX_BASE_TOP_M + object_height / 2.0,
                                ])
                                print(
                                    f"[skill-server] packing {color} into box slot {level}: "
                                    f"{np.round(stack_positions[color], 4).tolist()}",
                                    flush=True,
                                )
                            elif level:
                                # Stack relative to the verified physical
                                # centre of the current top block. PhysX can
                                # settle the first cube several centimetres
                                # away from the nominal box centre; a fixed
                                # second-layer XY target then misses it.
                                lower_color = state(observation)["target_box"]["contents_bottom_to_top"][-1]
                                lower_position = observation.cube_positions[lower_color]
                                stack_positions[color] = np.array(
                                    [lower_position[0], lower_position[1], lower_position[2] + config.cube_size]
                                )
                                print(
                                    f"[skill-server] aligning {color} above {lower_color}: "
                                    f"{np.round(stack_positions[color], 4).tolist()}",
                                    flush=True,
                                )
                            else:
                                stack_positions[color] = np.array(
                                    [BOX_CENTER[0], BOX_CENTER[1], config.cube_size / 2]
                                )
                            config = replace(config, stack_positions=stack_positions)
                            policy = TargetSelectionFSM(config, (color,))
                            print(f"[skill-server] building controller for {color} level={level}", flush=True)
                            controller = make_runtime_controller(config, policy, scene.controller_resources(), vision_grasps)
                            print(f"[skill-server] resetting controller for {color}", flush=True)
                            if not controller.reset(estimated, None, time):
                                reply(request, {"ok": False, "code": "CONTROLLER_RESET_FAILED", "detail": "controller reset failed", "data": {}})
                                controller = None
                            else:
                                print(f"[skill-server] controller active for {color}", flush=True)
                                active_request = request
            if controller is not None:
                if active_request is not None and frame_count == 0:
                    print(f"[skill-server] first controller forward for {active_request.payload['object_id']}", flush=True)
                apply_expert_step(controller, estimated, observation, time)
            else:
                # SceneIO requires every read() to be paired with apply(). In
                # the persistent service, camera/state RPCs do not create a
                # controller, but they still consume an observation. Submit an
                # empty command so the next service frame can safely read the
                # current physical state instead of terminating Kit.
                scene.apply(None)
            frame_count += 1
            if controller is not None and controller.done:
                if active_request is not None:
                    _, final_observation = scene.read(time)
                    color = active_request.payload["object_id"]
                    final_state = state(final_observation)
                    # Pair the verification read before returning to the idle
                    # RPC loop; otherwise the next request cannot read state.
                    scene.apply(None)
                    placed = next(item for item in final_state["objects"] if item["id"] == color)["location"] == "target_box"
                    policy_failure = getattr(policy, "failure_reason", None)
                    if placed:
                        failure_code = "OK"
                        failure_phase = "verify"
                    elif policy_failure and (
                        "slipped" in policy_failure or "not grasped" in policy_failure
                    ):
                        failure_code = "GRASP_FAILED"
                        failure_phase = "grasp"
                    else:
                        failure_code = "PLACEMENT_FAILED"
                        failure_phase = "place"
                    reply(active_request, {
                        "ok": placed,
                        "code": failure_code,
                        "detail": (
                            f"simulator verified {color} in target_box"
                            if placed else (policy_failure or f"simulator did not verify {color} in target_box")
                        ),
                        "data": {
                            "state": final_state,
                            "execution_feedback": {
                                "object_id": color,
                                "phase": failure_phase,
                                "outcome": "succeeded" if placed else "failed",
                                "failure_code": "" if placed else failure_code,
                                "verification": "in_target_box" if placed else "not_in_target_box",
                            },
                        },
                    })
                    active_request = None
                    controller = None
                    continue
                if recovery_stage == "retreat":
                    assert recovery_failed_color is not None
                    failed_color = recovery_failed_color
                    print("[recovery] safe retreat complete; capturing fresh overhead RGB-D", flush=True)
                    recovery_dir = (
                        Path(args.artifact_dir).expanduser().resolve()
                        if args.artifact_dir
                        else Path(__file__).resolve().parent / "observations"
                    )
                    recovery_image = recovery_dir / f"recovery_{recovery_attempts:02d}_overhead.png"
                    recovery_pose_path = recovery_dir / f"recovery_{recovery_attempts:02d}_vision_poses.json"
                    stale_visual_pose = vision_poses.get(failed_color)
                    recovery_focus = None if stale_visual_pose is None else stale_visual_pose[:2]
                    capture_overhead_rgb(
                        recovery_image,
                        recovery_pose_path,
                        focus_xy=recovery_focus,
                    )

                    # Physics is consulted only to exclude targets already
                    # verified in the box. New pickup positions come solely
                    # from the post-contact RGB-D observation.
                    recovery_time = SimulationManager.get_simulation_time()
                    recovery_estimated, recovery_observation = scene.read(recovery_time)
                    remaining = tuple(
                        color
                        for color in targets
                        if not (
                            np.linalg.norm(recovery_observation.cube_positions[color][:2] - BOX_CENTER) < 0.10
                            and recovery_observation.cube_positions[color][2] > config.cube_size * 0.35
                        )
                    )
                    if not remaining:
                        scene.apply(None)
                        print(f"[recovery] succeeded after {recovery_attempts} retry; all targets already verified", flush=True)
                        print("Success: requested blocks placed in target box.")
                        if recorder is not None:
                            recorder.finalize(success=True)
                        break
                    fresh_poses = load_rgbd_poses(recovery_pose_path, remaining)
                    fresh_grasps = load_grasp_descriptors(recovery_pose_path, remaining)
                    vision_poses.update(fresh_poses)
                    vision_grasps.update(fresh_grasps)
                    policy = TargetSelectionFSM(config, remaining)
                    controller = make_runtime_controller(config, policy, scene.controller_resources(), vision_grasps)
                    if not controller.reset(recovery_estimated, None, recovery_time):
                        scene.apply(None)
                        raise RuntimeError("Recovery controller reset failed")
                    if wrist_servo is not None:
                        wrist_servo.allow_retry(failed_color)
                    perceived_positions = dict(recovery_observation.cube_positions)
                    perceived_positions.update({color: vision_poses[color] for color in remaining})
                    recovery_perception = Observation(
                        sample_id=recovery_observation.sample_id,
                        time=recovery_observation.time,
                        end_effector_position=recovery_observation.end_effector_position,
                        gripper_position=recovery_observation.gripper_position,
                        cube_positions=perceived_positions,
                    )
                    apply_expert_step(controller, recovery_estimated, recovery_perception, recovery_time)
                    recovery_stage = None
                    print(
                        f"[recovery] retry started for {failed_color}; remaining={list(remaining)}; "
                        f"detected={sorted(fresh_poses)}",
                        flush=True,
                    )
                    continue
                if getattr(policy, "failure_reason", None) is not None:
                    reason = policy.failure_reason
                    failed_color = policy.failed_color
                    if recovery_attempts >= args.max_recovery_attempts or failed_color is None:
                        raise RuntimeError(f"RECOVERY_EXHAUSTED: {reason}")
                    recovery_attempts += 1
                    recovery_failed_color = failed_color
                    print(
                        f"[recovery] attempt {recovery_attempts}/{args.max_recovery_attempts}: "
                        f"retreat and fresh RGB-D after {reason}",
                        flush=True,
                    )
                    retreat_time = SimulationManager.get_simulation_time()
                    retreat_estimated, retreat_observation = scene.read(retreat_time)
                    policy = RecoveryRetreatPolicy(config, failed_color)
                    controller = make_runtime_controller(config, policy, scene.controller_resources(), vision_grasps)
                    if not controller.reset(retreat_estimated, None, retreat_time):
                        scene.apply(None)
                        raise RuntimeError("Recovery retreat controller reset failed")
                    apply_expert_step(controller, retreat_estimated, retreat_observation, retreat_time)
                    recovery_stage = "retreat"
                    continue
                if recovery_attempts:
                    print(f"[recovery] succeeded after {recovery_attempts} retry", flush=True)
                print("Success: requested blocks placed in target box.")
                if recorder is not None:
                    recorder.finalize(success=True)
                break
            if args.test and frame_count >= 2400 * (1 + args.max_recovery_attempts):
                raise RuntimeError("Timed out before placing requested blocks")
        elif app_utils.is_stopped():
            needs_reset = True
    if recorder is not None and not recorder.finalized:
        recorder.finalize(success=False, failure_reason="simulation stopped before task verification")
    if dagger_recorder is not None and not dagger_recorder.finalized:
        dagger_recorder.finalize(
            success=False, failure_reason="simulation stopped before task verification"
        )


if __name__ == "__main__":
    failed = False
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
        if _ACTIVE_VLA_RECORDER is not None:
            _ACTIVE_VLA_RECORDER.finalize(success=False, failure_reason="interrupted by user")
    except Exception:
        # Kit can otherwise close before its normal stderr log is flushed;
        # retain the Python failure that caused a persistent service to exit.
        failure_detail = traceback.format_exc()
        print(failure_detail, file=sys.stderr)
        if _ACTIVE_VLA_RECORDER is not None:
            _ACTIVE_VLA_RECORDER.finalize(success=False, failure_reason=failure_detail[-2000:])
        failed = True
    finally:
        simulation_app.close()
    if failed:
        raise SystemExit(1)
