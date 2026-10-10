"""Cloud Responses-API planner plus an offline planner for safe local smoke tests."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .contracts import COLORS, GROUPS
from .tools import TOOL_DEFINITIONS


SYSTEM_PROMPT = """You are the high-level planner for a simulated Franka Panda.
You can only affect the robot through listed tools. A user image may show the scene:
infer object colors and layout from that image; do not claim any unseen pose. Always call
get_scene_state before a physical action. Use only available object IDs and target_box. Never claim success:
state success only after a tool result has ok=true. If a physical tool fails, inspect its
structured error and either make at most one safe retry or explain why human intervention
is needed. Do not invent tools, poses, camera readings, or objects."""


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


class OfflinePlanner:
    """Predictable Chinese/English fallback; useful before an API key is configured."""

    def plan(self, instruction: str) -> list[ToolCall]:
        text = instruction.lower()
        found: list[str] = []
        aliases = {"红": "red", "red": "red", "蓝": "blue", "blue": "blue", "绿": "green", "green": "green", "黄": "yellow", "yellow": "yellow", "暖": "warm", "warm": "warm", "冷": "cool", "cool": "cool", "全部": "all", "all": "all"}
        for token, target in aliases.items():
            if token in text and target not in found:
                found.append(target)
        if not found:
            raise ValueError("Offline planner could not identify a color/group. Mention red/blue/green/yellow or 暖色/冷色.")
        return [ToolCall("offline-state", "get_scene_state", {}), ToolCall("offline-sort", "sort_to_target_box", {"targets": found})]


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

    def create(self, input_items: Any, previous_response_id: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model, "instructions": SYSTEM_PROMPT, "input": input_items, "tools": TOOL_DEFINITIONS, "parallel_tool_calls": False, "store": False}
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

    def create(self, input_items: Any, previous_response_id: str | None = None) -> dict[str, Any]:
        # DeepSeek does not persist response state. Preserve every assistant
        # item and every function result locally, then submit the full history.
        if not self._history:
            self._history.append({"role": "system", "content": SYSTEM_PROMPT})
        if isinstance(input_items, list):
            self._history.extend(input_items)
        else:
            self._history.append({"role": "user", "content": str(input_items)})
        payload = {"model": self.model, "input": self._history, "tools": TOOL_DEFINITIONS, "parallel_tool_calls": False}
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
