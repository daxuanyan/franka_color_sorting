"""Run a natural-language Franka sorting task through local tools and an optional LLM."""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import subprocess
from dataclasses import dataclass

from llm_agent.planner import (
    HYBRID_VLA_SYSTEM_PROMPT,
    OfflinePlanner,
    instruction_rejection_reason,
    planner_from_environment,
)
from llm_agent.recovery import (
    CloudRecoverySupervisor,
    RecoveryDecision,
    RuleRecoverySupervisor,
    feedback_from_tool_result,
)
from llm_agent.contracts import ToolResult
from llm_agent.tools import FrankaTools, HYBRID_VLA_TOOL_DEFINITIONS, TOOL_DEFINITIONS


@dataclass(frozen=True)
class VLAObservation:
    overhead_image_url: str
    wrist_image_url: str | None
    pose_path: Path

    def input_content(self, instruction: str) -> list[dict]:
        content: list[dict] = [
            {"type": "input_text", "text": instruction},
            {"type": "input_text", "text": "Overhead RGB observation:"},
            {"type": "input_image", "image_url": self.overhead_image_url, "detail": "low"},
        ]
        if self.wrist_image_url is not None:
            content.extend([
                {"type": "input_text", "text": "Wrist RGB observation:"},
                {"type": "input_image", "image_url": self.wrist_image_url, "detail": "low"},
            ])
        return content


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"'))


def capture_scene_image(root: Path, tools: FrankaTools | None = None) -> tuple[str, Path]:
    """Capture an RGB observation from Isaac Sim and return it as an API data URL."""
    image_path = root / "observations" / "latest_overhead.png"
    pose_path = image_path.with_name("latest_vision_poses.json")
    diagnostic = image_path.with_name("rgbd_capture_error.log")
    trace = image_path.with_name("rgbd_capture_trace.log")
    # A previous RGB image must never make a failed new capture look valid.
    for stale in (image_path, pose_path, diagnostic, trace):
        if stale.exists():
            stale.unlink()
    if tools is not None and tools.persistent_sim:
        captured = tools.capture_rgbd(image_path)
        tail = captured.detail + "\n" + json.dumps(captured.data, ensure_ascii=False)
        capture_failed = not captured.ok
    else:
        command = [str(root.parent / "isaacsim" / "python.sh"), str(root / "franka_color_sorting.py"), "--capture-path", str(image_path)]
        if tools is not None:
            command.extend(tools.perception_cli_args())
        completed = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=120, check=False)
        tail = (completed.stdout + "\n" + completed.stderr)[-3000:]
        capture_failed = completed.returncode != 0
    if capture_failed or not image_path.exists():
        extra = trace.read_text(errors="replace") if trace.exists() else tail
        raise RuntimeError(f"Camera capture failed. Detail:\n{extra}")
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    if not pose_path.exists():
        if diagnostic.exists():
            extra = diagnostic.read_text(errors="replace")
        elif trace.exists():
            extra = trace.read_text(errors="replace")
        else:
            extra = tail
        raise RuntimeError(f"Camera capture completed but did not produce RGB-D pose estimates. Detail:\n{extra}")
    return f"data:image/png;base64,{encoded}", pose_path


def capture_vla_observation(root: Path, tools: FrankaTools) -> VLAObservation:
    """Capture synchronized-enough overhead and wrist evidence for high-level VLA decisions."""
    overhead_url, pose_path = capture_scene_image(root, tools)
    wrist_url = None
    if tools.persistent_sim:
        wrist_path = root / "observations" / "latest_wrist_vla.png"
        wrist_path.unlink(missing_ok=True)
        captured = tools.capture_wrist_rgbd(wrist_path)
        if not captured.ok or not wrist_path.is_file():
            raise RuntimeError(f"Wrist VLA observation failed: {captured.detail}")
        wrist_url = "data:image/png;base64," + base64.b64encode(
            wrist_path.read_bytes()
        ).decode("ascii")
    return VLAObservation(overhead_url, wrist_url, pose_path)


def supervised_recovery(
    root: Path,
    tools: FrankaTools,
    call_name: str,
    call_arguments: dict,
    failed_result: ToolResult,
    *,
    mode: str,
    no_vision: bool,
) -> ToolResult:
    """Run one policy-checked observe/decide/retry cycle after plan failure."""
    if call_name not in {"execute_sort_plan", "execute_hybrid_vla_plan"} or mode == "off":
        return failed_result
    budget = call_arguments.get("max_retries", 0)
    if type(budget) is not int or budget <= 0:
        return failed_result
    try:
        def make_supervisor():
            if mode == "rule":
                return RuleRecoverySupervisor()
            recovery_planner = planner_from_environment()
            if recovery_planner is None:
                raise RuntimeError("LLM recovery requested but no cloud planner is configured")
            return CloudRecoverySupervisor(recovery_planner)

        feedback = feedback_from_tool_result(call_arguments, failed_result, recovery_budget=budget)
        decisions: list[RecoveryDecision] = []
        supervisor = make_supervisor()
        decision = supervisor.decide(feedback)
        decisions.append(decision)
        print(json.dumps({"recovery_feedback": feedback.to_dict(), "decision": decision.to_arguments()}, ensure_ascii=False, indent=2))
        if decision.action == "reobserve":
            if no_vision:
                raise RuntimeError("reobserve selected but vision is disabled")
            observation = capture_vla_observation(root, tools)
            tools.set_vision_pose_path(observation.pose_path)
            feedback = feedback.after_reobservation()
            # Use a fresh cloud conversation: each recovery decision is a
            # standalone classification, not an unresolved tool-call chain.
            supervisor = make_supervisor()
            if isinstance(supervisor, CloudRecoverySupervisor):
                image_urls = [observation.overhead_image_url]
                if observation.wrist_image_url is not None:
                    image_urls.append(observation.wrist_image_url)
                decision = supervisor.decide(feedback, image_urls=image_urls)
            else:
                decision = supervisor.decide(feedback)
            decisions.append(decision)
            print(json.dumps({"recovery_feedback": feedback.to_dict(), "decision": decision.to_arguments()}, ensure_ascii=False, indent=2))
        if decision.action != "retry":
            data = dict(failed_result.data)
            data["recovery_supervision"] = [item.to_arguments() for item in decisions]
            return ToolResult(False, failed_result.code, failed_result.detail, data)
        retry_result = tools.call(call_name, call_arguments)
        data = dict(retry_result.data)
        data["recovery_supervision"] = [item.to_arguments() for item in decisions]
        return ToolResult(retry_result.ok, retry_result.code, retry_result.detail, data)
    except Exception as error:
        # A malformed model decision or unavailable observation is a safe stop.
        data = dict(failed_result.data)
        data["recovery_supervision_error"] = str(error)
        print(f"Recovery supervisor stopped safely: {error}")
        return ToolResult(False, failed_result.code, failed_result.detail, data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instruction", help="Example: 把暖色方块堆进盒子")
    parser.add_argument("--dry-run", action="store_true", help="Validate planning and commands without opening Isaac Sim")
    parser.add_argument("--offline", action="store_true", help="Use deterministic local language parsing, even when an API key exists")
    parser.add_argument("--no-vision", action="store_true", help="Do not capture/send the overhead RGB observation to the cloud planner")
    parser.add_argument("--persistent-sim", action="store_true", help="Keep one Isaac Sim process alive for observation, execution, and replanning.")
    parser.add_argument(
        "--precise",
        action="store_true",
        help=(
            "Use the accuracy-first hierarchical VLA preset: persistent scene, "
            "generic RGB-D perception, CLIP association, and held-out irregular objects."
        ),
    )
    parser.add_argument("--perception-backend", choices=("color", "generic"), default="color")
    parser.add_argument("--semantic-backend", choices=("clip", "smolvlm", "prototype"), default="clip")
    parser.add_argument("--semantic-vocabulary", type=Path, help="JSON mapping scene IDs to visual descriptions.")
    parser.add_argument("--semantic-model", help="Optional semantic model override.")
    parser.add_argument("--semantic-min-confidence", type=float, default=0.50)
    parser.add_argument("--object-set", choices=("cubes", "seen", "heldout"), default="cubes")
    parser.add_argument(
        "--recovery-supervisor",
        choices=("auto", "off", "rule", "llm"),
        default="auto",
        help="Post-failure supervisor. auto uses LLM with a cloud planner and rules offline.",
    )
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument(
        "--trace-output", type=Path,
        help="Write the visual inputs, structured skill calls, and verified outcomes as JSON.",
    )
    args = parser.parse_args()
    if args.precise:
        # Keep language planning separate from metric execution: every pick is
        # re-observed in one persistent scene and verified from simulator physics.
        args.persistent_sim = not args.dry_run
        args.perception_backend = "generic"
        args.semantic_backend = "clip"
        args.object_set = "heldout"
    root = Path(__file__).resolve().parent
    load_dotenv(root / ".env")
    tools = FrankaTools(
        root,
        dry_run=args.dry_run,
        persistent_sim=args.persistent_sim,
        perception_backend=args.perception_backend,
        semantic_backend=args.semantic_backend,
        semantic_vocabulary=args.semantic_vocabulary,
        semantic_model=args.semantic_model,
        semantic_min_confidence=args.semantic_min_confidence,
        object_set=args.object_set,
    )
    planner = None if args.offline else planner_from_environment()
    recovery_mode = args.recovery_supervisor
    if recovery_mode == "auto":
        recovery_mode = "rule" if planner is None else "llm"
    trace_path = args.trace_output
    if trace_path is None and args.precise:
        trace_path = root / "observations" / "hybrid_vla_trace.json"
    trace: dict = {
        "schema_version": 1,
        "instruction": args.instruction,
        "mode": "hybrid_vla" if args.precise else "language_agent",
        "planner": "offline" if planner is None else f"cloud:{planner.model}",
        "visual_inputs": [],
        "tool_events": [],
    }

    try:
        rejection = instruction_rejection_reason(args.instruction)
        if rejection:
            print(f"Request rejected before planning: {rejection}")
            return
        if planner is None:
            print("Planner: offline (configure .env for cloud tool calling)")
            for call in OfflinePlanner().plan(args.instruction, hybrid=args.precise):
                result = tools.call(call.name, call.arguments)
                if not result.ok:
                    result = supervised_recovery(
                        root, tools, call.name, call.arguments, result,
                        mode=recovery_mode, no_vision=args.no_vision,
                    )
                trace["tool_events"].append({
                    "tool": call.name,
                    "arguments": call.arguments,
                    "result": result.to_dict(),
                })
                print(json.dumps({"tool": call.name, "result": result.to_dict()}, ensure_ascii=False, indent=2))
                if not result.ok:
                    raise SystemExit(1)
            return

        print(f"Planner: cloud Responses API ({planner.model})")
        if args.no_vision:
            user_input = [{"role": "user", "content": args.instruction}]
        else:
            observation = capture_vla_observation(root, tools)
            tools.set_vision_pose_path(observation.pose_path)
            trace["visual_inputs"] = [
                {"view": "overhead", "path": str(root / "observations" / "latest_overhead.png")},
                *([{"view": "wrist", "path": str(root / "observations" / "latest_wrist_vla.png")}]
                  if observation.wrist_image_url is not None else []),
            ]
            user_input = [{"role": "user", "content": [
                *observation.input_content(args.instruction),
            ]}]
            print(
                "Vision: overhead"
                + (" + wrist" if observation.wrist_image_url is not None else "")
                + " RGB observations captured and sent to the planner"
            )
        active_tools = HYBRID_VLA_TOOL_DEFINITIONS if args.precise else TOOL_DEFINITIONS
        active_prompt = HYBRID_VLA_SYSTEM_PROMPT if args.precise else None
        response = planner.create(
            user_input, instructions=active_prompt, tools=active_tools
        )
        for _ in range(args.max_turns):
            calls = planner.tool_calls(response)
            if not calls:
                print(planner.text(response))
                return
            outputs = []
            for call in calls:
                result = tools.call(call.name, call.arguments)
                if not result.ok:
                    result = supervised_recovery(
                        root, tools, call.name, call.arguments, result,
                        mode=recovery_mode, no_vision=args.no_vision,
                    )
                trace["tool_events"].append({
                    "tool": call.name,
                    "arguments": call.arguments,
                    "result": result.to_dict(),
                })
                print(json.dumps({"tool": call.name, "result": result.to_dict()}, ensure_ascii=False, indent=2))
                outputs.append({"type": "function_call_output", "call_id": call.call_id, "output": json.dumps(result.to_dict(), ensure_ascii=False)})
            response = planner.create(
                outputs,
                previous_response_id=response["id"],
                instructions=active_prompt,
                tools=active_tools,
            )
        raise SystemExit("Agent exceeded max turns without a final answer")
    finally:
        tools.close()
        if trace_path is not None:
            resolved_trace = trace_path.expanduser().resolve()
            resolved_trace.parent.mkdir(parents=True, exist_ok=True)
            resolved_trace.write_text(
                json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"Hybrid VLA trace: {resolved_trace}")


if __name__ == "__main__":
    main()
