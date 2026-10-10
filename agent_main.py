"""Run a natural-language Franka sorting task through local tools and an optional LLM."""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import subprocess

from llm_agent.planner import OfflinePlanner, planner_from_environment
from llm_agent.tools import FrankaTools


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"'))


def capture_scene_image(root: Path) -> str:
    """Capture an RGB observation from Isaac Sim and return it as an API data URL."""
    image_path = root / "observations" / "latest_overhead.png"
    command = [str(root.parent / "isaacsim" / "python.sh"), str(root / "franka_color_sorting.py"), "--capture-path", str(image_path)]
    completed = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=120, check=False)
    if completed.returncode != 0 or not image_path.exists():
        tail = (completed.stdout + "\n" + completed.stderr)[-2000:]
        raise RuntimeError(f"Camera capture failed. Isaac Sim output:\n{tail}")
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instruction", help="Example: 把暖色方块堆进盒子")
    parser.add_argument("--dry-run", action="store_true", help="Validate planning and commands without opening Isaac Sim")
    parser.add_argument("--offline", action="store_true", help="Use deterministic local language parsing, even when an API key exists")
    parser.add_argument("--no-vision", action="store_true", help="Do not capture/send the overhead RGB observation to the cloud planner")
    parser.add_argument("--max-turns", type=int, default=8)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    load_dotenv(root / ".env")
    tools = FrankaTools(root, dry_run=args.dry_run)
    planner = None if args.offline else planner_from_environment()

    try:
        if planner is None:
            print("Planner: offline (configure .env for cloud tool calling)")
            for call in OfflinePlanner().plan(args.instruction):
                result = tools.call(call.name, call.arguments)
                print(json.dumps({"tool": call.name, "result": result.to_dict()}, ensure_ascii=False, indent=2))
                if not result.ok:
                    raise SystemExit(1)
            return

        print(f"Planner: cloud Responses API ({planner.model})")
        if args.no_vision:
            user_input = [{"role": "user", "content": args.instruction}]
        else:
            image_url = capture_scene_image(root)
            user_input = [{"role": "user", "content": [
                {"type": "input_text", "text": args.instruction},
                {"type": "input_image", "image_url": image_url, "detail": "low"},
            ]}]
            print("Vision: overhead RGB observation captured and sent to the planner")
        response = planner.create(user_input)
        for _ in range(args.max_turns):
            calls = planner.tool_calls(response)
            if not calls:
                print(planner.text(response))
                return
            outputs = []
            for call in calls:
                result = tools.call(call.name, call.arguments)
                print(json.dumps({"tool": call.name, "result": result.to_dict()}, ensure_ascii=False, indent=2))
                outputs.append({"type": "function_call_output", "call_id": call.call_id, "output": json.dumps(result.to_dict(), ensure_ascii=False)})
            response = planner.create(outputs, previous_response_id=response["id"])
        raise SystemExit("Agent exceeded max turns without a final answer")
    finally:
        tools.close()


if __name__ == "__main__":
    main()
