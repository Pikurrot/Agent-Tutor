from __future__ import annotations

import json
from typing import Any, Optional

from langchain_core.callbacks.base import BaseCallbackHandler


def extract_thought_from_log(log: str) -> str:
    if "Action:" in log:
        return log.split("Action:")[0].strip()
    return log.strip()


def normalize_action_input(tool_input: Any) -> Any:
    if isinstance(tool_input, dict):
        return tool_input
    if isinstance(tool_input, str):
        stripped = tool_input.strip()
        if stripped.startswith("{"):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass
        return tool_input
    return str(tool_input)


def metadata_from_action(tool: str, tool_input: Any) -> tuple[Optional[str], Optional[int]]:
    document: Optional[str] = None
    slide_number: Optional[int] = None
    normalized = normalize_action_input(tool_input)

    if tool == "Search_Document_Context" and isinstance(normalized, dict):
        document = normalized.get("document_name")
        if document is not None:
            document = str(document)
    elif tool == "Retrieve_Slide_Context" and isinstance(normalized, dict):
        document = normalized.get("document_name")
        if document is not None:
            document = str(document)
        raw_slide = normalized.get("slide_number")
        if raw_slide is not None:
            slide_number = int(raw_slide)
    return document, slide_number


class AgentTraceCallback(BaseCallbackHandler):
    """Collect ReAct action steps and final text."""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.steps: list[dict[str, Any]] = []
        self.raw_output: str = ""
        self.notes: list[str] = []

    @property
    def chain_length(self) -> int:
        return len(self.steps)

    def on_agent_action(self, action, **kwargs):
        thought = extract_thought_from_log(action.log)
        tool_input = normalize_action_input(action.tool_input)
        document, slide_number = metadata_from_action(action.tool, action.tool_input)
        self.steps.append(
            {
                "step": len(self.steps) + 1,
                "thought": thought,
                "action": action.tool,
                "action_input": tool_input,
                "document": document,
                "slide_number": slide_number,
            }
        )

    def on_agent_finish(self, finish, **kwargs):
        # Works with both AgentFinish and dictionary-style payloads.
        output = ""
        if hasattr(finish, "return_values"):
            output = str(getattr(finish, "return_values", {}).get("output", ""))
        elif isinstance(finish, dict):
            output = str(finish.get("output", ""))
        self.raw_output = output

    def to_trace_dict(
        self,
        *,
        used_fallback: bool = False,
        parse_retry_count: int = 0,
        parse_failed: bool = False,
        notes: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "steps": self.steps,
            "raw_output": self.raw_output,
            "chain_length": self.chain_length,
            "used_fallback": bool(used_fallback),
            "parse_retry_count": int(parse_retry_count),
            "parse_failed": bool(parse_failed),
            "notes": list(notes or self.notes),
        }
