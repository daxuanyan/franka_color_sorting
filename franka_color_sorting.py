"""Sort requested colored blocks into a physical target box with a Franka Panda.

Run from the Isaac Sim root, for example:
    ./python.sh /home/robot/franka_color_sorting/franka_color_sorting.py --target red

The task policy uses the scene's color/category labels.  This intentionally
keeps object selection separate from robot control; replace ``resolve_targets``
with a camera detector when moving to visual perception.
"""

from __future__ import annotations

import argparse
import json
import queue
import socket
import sys
import threading
import traceback
from dataclasses import dataclass
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
args, _ = parser.parse_known_args()

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
from isaacsim.core.experimental.objects import Cube
from isaacsim.core.experimental.prims import GeomPrim
from isaacsim.core.simulation_manager import SimulationManager


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
BOX_CENTER = np.array([0.55, 0.20])


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


def make_config(targets: tuple[str, ...]) -> TaskConfig:
    """Spawn four blocks, but reserve the target-box slots for requested ones."""
    cube_size = 0.05
    spawn_positions = {
        "red": np.array([0.45, -0.30, cube_size / 2]),
        "blue": np.array([0.55, -0.30, cube_size / 2]),
        "green": np.array([0.45, -0.45, cube_size / 2]),
        "yellow": np.array([0.55, -0.45, cube_size / 2]),
    }
    # Non-targets retain harmless locations outside the goal test.  Selected
    # blocks are placed one above another inside the box.
    stack_positions = {color: spawn_positions[color].copy() for color in COLORS}
    for index, color in enumerate(targets):
        stack_positions[color] = np.array([BOX_CENTER[0], BOX_CENTER[1], cube_size / 2 + index * cube_size])
    return TaskConfig(
        cube_order=COLORS,
        spawn_positions=spawn_positions,
        stack_positions=stack_positions,
        cube_size=cube_size,
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

    def reset(self) -> None:
        self._active = None

    def forward(self, observation: Observation, *, target_reached: bool, gripper_reached: bool) -> PickPlaceDecision:
        if self._active is None:
            self._active = self._select_next_cube(observation)
            if self._active is None:
                return PickPlaceDecision(done=True)
        if self._cube_was_dropped(observation):
            print(f"[drop] {self._active.color} fell out of gripper; retrying")
            self._active = None
            return PickPlaceDecision()
        self._restart_pickup_if_cube_moved(observation)
        self._advance_if_ready(observation, target_reached, gripper_reached)
        return self._decision()

    def _select_next_cube(self, observation: Observation) -> PickPlaceState | None:
        for color in self._targets:
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
            self._active = None
            return
        active.phase = self._NEXT_PHASE[active.phase]
        active.phase_started = observation.time
        active.settle_started = None
        if active.phase is PickPlacePhase.GRASP:
            active.pickup_position = observation.cube_positions[active.color].copy()
        if active.holding and not active.aborted and not self._stack_below_is_valid(active.color, observation):
            active.destination = self._config.spawn_positions[active.color].copy()
            active.aborted = True

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
            self._at_goal(observation.cube_positions[lower], self._config.stack_positions[lower])
            for lower in self._targets[:index]
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


def capture_overhead_rgb(path: Path) -> None:
    """Render a visual observation without exposing any object pose to the Agent."""
    from PIL import Image
    from isaacsim.core.experimental.utils.transform import look_at_quaternion
    from isaacsim.sensors.camera import Camera

    eye = np.array([0.52, -0.08, 1.35])
    camera = Camera(
        prim_path="/World/AgentOverheadCamera",
        position=eye,
        orientation=look_at_quaternion(eye=eye, target=np.array([0.50, -0.10, 0.0])).numpy(),
        frequency=20,
        resolution=(640, 640),
    )
    app_utils.play()
    camera.initialize()
    # Let PhysX settle and the render product receive several frames.
    for _ in range(12):
        simulation_app.update()
    rgba = camera.get_rgba()
    if rgba is None:
        raise RuntimeError("Overhead camera did not produce an RGB frame")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgba[:, :, :3], dtype=np.uint8)).save(path)
    print(f"Saved overhead RGB observation: {path}")


def main() -> None:
    # The persistent server always spawns every known object. Individual skills
    # select one object at a time, preserving the physics state between calls.
    targets = COLORS if args.server_port else resolve_targets(args.target)
    config = make_config(targets)
    print(f"Requested: {', '.join(targets)}. Target box: {BOX_CENTER.tolist()}")

    SimulationManager.setup_simulation(dt=PHYSICS_DT, device="cpu")
    scene = FrankaStackingSceneIO(config)
    policy = TargetSelectionFSM(config, targets)
    simulation_app.run_coroutine(scene.setup())
    simulation_app.update()
    add_target_box()
    simulation_app.update()

    if args.capture_path:
        capture_overhead_rgb(Path(args.capture_path).expanduser().resolve())
        return

    server = LocalSkillServer(args.server_port) if args.server_port else None
    if server:
        server.start()
        print(f"SKILL_SERVER_READY {args.server_port}", flush=True)

    if args.headless and not args.test and not server:
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
        controller = None if server else make_pick_place_controller(config, policy, scene.controller_resources())
        initialized = True

    needs_reset = True
    frame_count = 0
    active_request: SkillRequest | None = None

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

    while simulation_app.is_running():
        simulation_app.update()
        # In headless Kit the initial play request can be deferred by a few
        # frames. Keep requesting it while the persistent skill server waits.
        if server and not app_utils.is_playing():
            app_utils.play()
        if app_utils.is_playing() and SimulationManager.is_simulating():
            if not initialized:
                scene.initialize()
                controller = make_pick_place_controller(config, policy, scene.controller_resources())
                initialized = True
            time = SimulationManager.get_simulation_time()
            if needs_reset:
                scene.reset()
                estimated, observation = scene.read(time)
                if controller is not None:
                    if not controller.reset(estimated, None, time):
                        raise RuntimeError("Controller reset failed")
                    scene.apply(controller.forward(estimated, None, time, observation=observation))
                needs_reset = False
                continue
            estimated, observation = scene.read(time)
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
                            config.stack_positions[color] = np.array([BOX_CENTER[0], BOX_CENTER[1], config.cube_size / 2 + level * config.cube_size])
                            policy = TargetSelectionFSM(config, (color,))
                            controller = make_pick_place_controller(config, policy, scene.controller_resources())
                            if not controller.reset(estimated, None, time):
                                reply(request, {"ok": False, "code": "CONTROLLER_RESET_FAILED", "detail": "controller reset failed", "data": {}})
                                controller = None
                            else:
                                active_request = request
            if controller is not None:
                scene.apply(controller.forward(estimated, None, time, observation=observation))
            frame_count += 1
            if controller is not None and controller.done:
                if active_request is not None:
                    _, final_observation = scene.read(time)
                    color = active_request.payload["object_id"]
                    final_state = state(final_observation)
                    placed = next(item for item in final_state["objects"] if item["id"] == color)["location"] == "target_box"
                    reply(active_request, {"ok": placed, "code": "OK" if placed else "PLACEMENT_FAILED", "detail": f"simulator {'verified' if placed else 'did not verify'} {color} in target_box", "data": final_state})
                    active_request = None
                    controller = None
                    continue
                print("Success: requested blocks placed in target box.")
                break
            if args.test and frame_count >= 2400:
                raise RuntimeError("Timed out before placing requested blocks")
        elif app_utils.is_stopped():
            needs_reset = True


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        simulation_app.close()
