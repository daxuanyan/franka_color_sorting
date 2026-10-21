"""Cloud Responses-API planner plus an offline planner for safe local smoke tests."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .contracts import COLORS, GROUPS
from .task_plan import TaskPlan
from .hybrid_vla import HybridVLAPlan
from .tools import HYBRID_VLA_TOOL_DEFINITIONS, TOOL_DEFINITIONS


SYSTEM_PROMPT = """You are the high-level task planner for a simulated Franka Panda.
You can affect the robot only through listed tools. Compile the user's request into an ordered,
minimal plan; deterministic local code handles perception coordinates and motion. Always call
get_scene_state before execute_sort_plan. Use only observed object IDs and target_box. Resolve
explicit exclusions, ordering, spatial relations, and confidence conditions from the instruction
and observed scene. If the request is ambiguous, contradictory, impossible, or asks you to bypass
safety, do not execute; ask one concise clarification or explain the refusal. Never claim success
until a physical tool returns ok=true. After a failure, make at most the plan's permitted retry and
otherwise stop for human inspection. Never invent tools, poses, camera readings, or objects.
Color groups are fixed: warm means red and yellow; cool means blue and green; all means red, blue,
green, and yellow. After a successful get_scene_state, either call execute_sort_plan or clearly
refuse/clarify; do not silently stop after observing."""

HYBRID_VLA_SYSTEM_PROMPT = """You are the visual-language-action planner in a hierarchical
robot manipulation system. You receive the user's instruction, an overhead RGB observation, a
wrist RGB observation when available, and trusted structured scene/tool feedback. Always call
get_scene_state before execute_hybrid_vla_plan. Use the images for semantic target selection and
ordering, then output only the bounded high-level skill schema. Every new task starts at localize,
uses pick_and_place with shape_aware grasping, and targets target_box. Local RGB-D perception,
grasp geometry, wrist visual servo, motion control, and physics verification are authoritative for
metric execution. Never output coordinates or joint commands. Never claim success before a tool
returns ok=true. On failure, use the reported phase and verification outcome; request only a
policy-permitted reobservation/retry and preserve completed objects. Reject ambiguity, unknown
objects, unsafe requests, and attempts to bypass verification."""


def instruction_rejection_reason(instruction: str) -> str | None:
    """Reject high-risk language locally before an LLM can request motion."""
    text = instruction.lower()
    if re.search(r"\bor\b", text) or any(token in text for token in ("或者", "还是")):
        return "ambiguous alternatives require one explicit target or order."
    impossible = ("purple", "orange", "black", "white", "紫色", "橙色", "黑色", "白色")
    if any(token in text for token in impossible):
        return "The request names an object that is not present in the supported scene."
    bypass = (
        "ignore safety", "claim success without", "bypass the controller", "joint angles",
        "忽略安全", "不调用工具直接", "绕过控制器", "关节角",
    )
    if any(token in text for token in bypass):
        return "Requests to bypass the verified controller or fabricate success are not allowed."
    vague = ("nice-looking", "whichever you like", "随便放", "好看的方块")
    if any(token in text for token in vague):
        return "The requested object is subjective or underspecified; name a color or group."
    return None


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


class OfflinePlanner:
    """Predictable Chinese/English fallback; useful before an API key is configured."""

    def plan(self, instruction: str, *, hybrid: bool = False) -> list[ToolCall]:
        text = instruction.lower()
        rejection = instruction_rejection_reason(instruction)
        if rejection:
            raise ValueError(rejection)
        spatial = ("left", "right", "nearest", "farthest", "closest", "左边", "右边", "最近", "最远", "从左到右", "从右到左")
        if any(token in text for token in spatial):
            raise ValueError("Offline planner cannot resolve spatial relations; use a vision-capable planner.")
        aliases = {
            "红色": "red", "红": "red", "red": "red",
            "蓝色": "blue", "蓝": "blue", "blue": "blue",
            "绿色": "green", "绿": "green", "green": "green",
            "黄色": "yellow", "黄": "yellow", "yellow": "yellow",
            "暖色": "warm", "暖": "warm", "warm": "warm",
            "冷色": "cool", "冷": "cool", "cool": "cool",
            "全部": "all", "所有": "all", "all": "all",
        }

        def token_position(source: str, token: str) -> int:
            if token.isascii():
                match = re.search(rf"\b{re.escape(token)}\b", source)
                return -1 if match is None else match.start()
            return source.find(token)

        exclusions: set[str] = set()
        exclusion_markers = ("except", "excluding", "but not", "do not move", "don't move", "不要动", "不要移动", "别动", "排除")
        for marker in exclusion_markers:
            start = text.find(marker)
            if start < 0:
                continue
            segment = text[start + len(marker) :]
            for token, target in aliases.items():
                if token_position(segment, token) >= 0:
                    exclusions.update(GROUPS.get(target, (target,)))
        mentions: list[tuple[int, str]] = []
        for token, target in aliases.items():
            position = token_position(text, token)
            if position >= 0:
                mentions.append((position, target))
        # In "all warm blocks" / "所有暖色方块", all is a quantifier for the
        # named group rather than a request for every colour in the scene.
        if any(target in {"warm", "cool", "primary"} for _, target in mentions):
            mentions = [(position, target) for position, target in mentions if target != "all"]
        mentions.sort()
        found: list[str] = []
        for _, target in mentions:
            for color in GROUPS.get(target, (target,)):
                if color not in exclusions and color not in found:
                    found.append(color)
        if not found:
            raise ValueError("Offline planner could not identify a color/group. Mention red/blue/green/yellow or 暖色/冷色.")
        plan = HybridVLAPlan.for_targets(found) if hybrid else TaskPlan.for_targets(found)
        return [
            ToolCall("offline-state", "get_scene_state", {}),
            ToolCall(
                "offline-plan",
                "execute_hybrid_vla_plan" if hybrid else "execute_sort_plan",
                plan.to_arguments(),
            ),
        ]


class OpenAIResponsesPlanner:
    """Minimal standard-library client for OpenAI's Responses API function-call loop."""

    def __init__(self, api_key: str, model: str, base_url: str = "https://api.openai.com/v1") -> None:
        self.api_key, self.model, self.base_url = api_key, model, base_url.rstrip("/")

    @classmethod
    def from_environment(cls) -> "OpenAIResponsesPlanner | None":
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            return None
        return cls(key, os.getenv("OPENAI_MODEL", "gpt-5.4"), os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))

    def create(
        self,
        input_items: Any,
        previous_response_id: str | None = None,
        *,
        instructions: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "instructions": instructions or SYSTEM_PROMPT,
            "input": input_items,
            "tools": TOOL_DEFINITIONS if tools is None else tools,
            "parallel_tool_calls": False,
            "store": False,
        }
        if previous_response_id:
            payload["previous_response_id"] = previous_response_id
        request = urllib.request.Request(f"{self.base_url}/responses", data=json.dumps(payload).encode(), headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"LLM request failed ({error.code}): {error.read().decode(errors='replace')}") from error

    @staticmethod
    def tool_calls(response: dict[str, Any]) -> list[ToolCall]:
        calls = []
        for item in response.get("output", []):
            if item.get("type") == "function_call":
                calls.append(ToolCall(item["call_id"], item["name"], json.loads(item["arguments"])))
        return calls

    @staticmethod
    def text(response: dict[str, Any]) -> str:
        return response.get("output_text", "")


class DeepSeekResponsesPlanner(OpenAIResponsesPlanner):
    """DeepSeek's compatible Responses API, which requires client-held history."""

    def __init__(self, api_key: str, model: str, base_url: str = "https://api.deepseek.com") -> None:
        super().__init__(api_key, model, base_url)
        self._history: list[Any] = []

    @classmethod
    def from_environment(cls) -> "DeepSeekResponsesPlanner | None":
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key:
            return None
        return cls(key, os.getenv("DEEPSEEK_MODEL", "deepseek-flash"), os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))

    def create(
        self,
        input_items: Any,
        previous_response_id: str | None = None,
        *,
        instructions: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        # DeepSeek does not persist response state. Preserve every assistant
        # item and every function result locally, then submit the full history.
        if not self._history:
            self._history.append({"role": "system", "content": instructions or SYSTEM_PROMPT})
        if isinstance(input_items, list):
            self._history.extend(input_items)
        else:
            self._history.append({"role": "user", "content": str(input_items)})
        payload = {
            "model": self.model,
            "input": self._history,
            "tools": TOOL_DEFINITIONS if tools is None else tools,
            "parallel_tool_calls": False,
        }
        request = urllib.request.Request(f"{self.base_url}/responses", data=json.dumps(payload).encode(), headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"DeepSeek request failed ({error.code}): {error.read().decode(errors='replace')}") from error
        self._history.extend(result.get("output", []))
        return result


def planner_from_environment() -> OpenAIResponsesPlanner | DeepSeekResponsesPlanner | None:
    """Select the provider explicitly so incompatible continuation rules cannot mix."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider == "deepseek":
        return DeepSeekResponsesPlanner.from_environment()
    if provider == "openai":
        return OpenAIResponsesPlanner.from_environment()
    raise ValueError("LLM_PROVIDER must be 'openai' or 'deepseek'")
