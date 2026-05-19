from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from tutor.modules.models.base import BaseModel


LastInteractionDict = dict


@dataclass
class ConversationMemory:
    summary: str = ""
    last_interaction: Optional[LastInteractionDict] = None

    @classmethod
    def empty(cls) -> "ConversationMemory":
        return cls(summary="", last_interaction=None)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "ConversationMemory":
        if not data:
            return cls.empty()
        last = data.get("last_interaction")
        if last is not None and not (isinstance(last, dict) and "user" in last and "assistant" in last):
            last = None
        return cls(summary=str(data.get("summary") or ""), last_interaction=last)

    def to_dict(self) -> dict:
        return {"summary": self.summary, "last_interaction": self.last_interaction}

    def is_empty(self) -> bool:
        return not self.summary and self.last_interaction is None


_SUMMARY_PROMPT = """You are maintaining a running summary of a conversation between a user and an AI tutor.

Update the summary by integrating the new exchange below. Follow these rules strictly:
- Preserve key facts, topics, and any specific documents/concepts the user has been studying.
- Keep it concise: short bullet points or a brief paragraph.
- Drop trivial pleasantries and redundant detail.
- Do NOT invent information that is not in the existing summary or the new exchange.
- Output ONLY the updated summary text. No preface, no labels, no quoting.

Existing summary:
<<<
{previous_summary}
>>>

New exchange to fold in:
USER: {user_msg}
ASSISTANT: {assistant_msg}

Updated summary:
"""


def summarize_context(
    previous_summary: str,
    last_interaction: LastInteractionDict,
    model: BaseModel,
    max_new_tokens: int = 512,
) -> str:
    prev = (previous_summary or "").strip() or "(empty)"
    user_msg = (last_interaction.get("user") or "").strip()
    assistant_msg = (last_interaction.get("assistant") or "").strip()

    prompt = _SUMMARY_PROMPT.format(
        previous_summary=prev,
        user_msg=user_msg,
        assistant_msg=assistant_msg,
    )

    new_summary = model.generate(prompt, max_new_tokens=max_new_tokens)
    return (new_summary or "").strip()


def format_memory_for_prompt(memory: Optional[ConversationMemory]) -> str:
    if memory is None or memory.is_empty():
        return ""

    parts: list[str] = []
    if memory.summary.strip():
        parts.append(
            "Conversation summary so far:\n"
            f"{memory.summary.strip()}"
        )
    if memory.last_interaction is not None:
        user_msg = (memory.last_interaction.get("user") or "").strip()
        assistant_msg = (memory.last_interaction.get("assistant") or "").strip()
        if user_msg or assistant_msg:
            parts.append(
                "Most recent exchange (verbatim):\n"
                f"USER: {user_msg}\n"
                f"ASSISTANT: {assistant_msg}"
            )

    if not parts:
        return ""

    return "\n\n" + "\n\n".join(parts) + "\n"


def roll_memory(
    memory: Optional[ConversationMemory],
    new_user: str,
    new_assistant: str,
    model: BaseModel,
    summary_max_new_tokens: int = 512,
) -> ConversationMemory:
    """Slide the memory window forward by one turn.

    Folds the OLD ``last_interaction`` into ``summary`` (no-op when there is
    no prior interaction) and sets ``last_interaction`` to the just-finished
    exchange.
    """
    mem = memory if memory is not None else ConversationMemory.empty()

    if mem.last_interaction is not None:
        new_summary = summarize_context(
            mem.summary,
            mem.last_interaction,
            model,
            max_new_tokens=summary_max_new_tokens,
        )
    else:
        new_summary = mem.summary

    return ConversationMemory(
        summary=new_summary,
        last_interaction={"user": new_user, "assistant": new_assistant},
    )
