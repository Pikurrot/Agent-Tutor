from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from tutor.utils.misc import parse_json_response


TeachingStatus = Literal["exploring", "narrowing", "close", "resolved"]

TOPIC_SHIFT_PREFIX = "[TOPIC_SHIFT:"
ADVANCE_SCAFFOLD_SIGNAL = "[ADVANCE_SCAFFOLD]"

_TOPIC_SHIFT_RE = re.compile(r"\[TOPIC_SHIFT:\s*(.*?)\]", re.IGNORECASE | re.DOTALL)
_ADVANCE_SCAFFOLD_RE = re.compile(r"\[ADVANCE_SCAFFOLD\]", re.IGNORECASE)


@dataclass
class TeachingAnchor:
    """Private tutor notes about the current student doubt.

    Built by the Answer Agent from retrieved course material. Never quoted
    verbatim to the student; consumed only by the Pedagogic Agent prompt.
    """

    original_question: str = ""
    target_explanation: str = ""
    key_facts: list[str] = field(default_factory=list)
    misconceptions: list[str] = field(default_factory=list)
    scaffold_questions: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "TeachingAnchor":
        if not data:
            return cls()
        return cls(
            original_question=str(data.get("original_question") or ""),
            target_explanation=str(data.get("target_explanation") or ""),
            key_facts=_coerce_str_list(data.get("key_facts")),
            misconceptions=_coerce_str_list(data.get("misconceptions")),
            scaffold_questions=_coerce_str_list(data.get("scaffold_questions")),
            citations=_coerce_str_list(data.get("citations")),
        )

    def to_dict(self) -> dict:
        return {
            "original_question": self.original_question,
            "target_explanation": self.target_explanation,
            "key_facts": list(self.key_facts),
            "misconceptions": list(self.misconceptions),
            "scaffold_questions": list(self.scaffold_questions),
            "citations": list(self.citations),
        }

    def is_empty(self) -> bool:
        return not (
            self.target_explanation
            or self.key_facts
            or self.misconceptions
            or self.scaffold_questions
            or self.citations
        )

    def merge_with(self, other: "TeachingAnchor") -> "TeachingAnchor":
        """Fold a freshly-built anchor on top of the current one.

        Used when the orchestrator rebuilds the anchor via the Answer Agent:
        we want to keep existing scaffolding but enrich facts and citations.
        """
        if other.is_empty():
            return self
        merged_explanation = self.target_explanation
        if other.target_explanation and other.target_explanation != self.target_explanation:
            if merged_explanation:
                merged_explanation = (
                    merged_explanation.rstrip() + "\n\n" + other.target_explanation.strip()
                )
            else:
                merged_explanation = other.target_explanation
        return TeachingAnchor(
            original_question=other.original_question or self.original_question,
            target_explanation=merged_explanation,
            key_facts=_merge_unique(self.key_facts, other.key_facts),
            misconceptions=_merge_unique(self.misconceptions, other.misconceptions),
            scaffold_questions=_merge_unique(self.scaffold_questions, other.scaffold_questions),
            citations=_merge_unique(self.citations, other.citations),
        )


@dataclass
class TeachingSession:
    """Per-conversation tutor state. Round-tripped through the API.

    The server is the source of truth for the anchor; the client only carries
    the serialized form between turns.
    """

    anchor: Optional[TeachingAnchor] = None
    status: TeachingStatus = "exploring"
    hints_used: int = 0
    current_scaffold_index: int = 0

    @classmethod
    def empty(cls) -> "TeachingSession":
        return cls()

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "TeachingSession":
        if not data:
            return cls.empty()
        anchor_data = data.get("anchor")
        anchor = TeachingAnchor.from_dict(anchor_data) if anchor_data else None
        if anchor is not None and anchor.is_empty():
            anchor = None
        status = data.get("status") or "exploring"
        if status not in ("exploring", "narrowing", "close", "resolved"):
            status = "exploring"
        hints_used = data.get("hints_used", 0)
        try:
            hints_used = int(hints_used)
        except (TypeError, ValueError):
            hints_used = 0
        scaffold_index = data.get("current_scaffold_index", 0)
        try:
            scaffold_index = max(0, int(scaffold_index))
        except (TypeError, ValueError):
            scaffold_index = 0
        return cls(
            anchor=anchor,
            status=status,
            hints_used=hints_used,
            current_scaffold_index=scaffold_index,
        )

    def to_dict(self) -> dict:
        return {
            "anchor": self.anchor.to_dict() if self.anchor is not None else None,
            "status": self.status,
            "hints_used": int(self.hints_used),
            "current_scaffold_index": int(self.current_scaffold_index),
        }

    def has_anchor(self) -> bool:
        return self.anchor is not None and not self.anchor.is_empty()


def _coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value)]


def _merge_unique(base: list[str], extra: list[str]) -> list[str]:
    out = list(base)
    seen = {item.strip().lower() for item in out}
    for item in extra:
        key = item.strip().lower()
        if key and key not in seen:
            out.append(item)
            seen.add(key)
    return out


def format_anchor_for_pedagogic_prompt(anchor: Optional[TeachingAnchor]) -> str:
    """Build the TUTOR-ONLY anchor block injected into the Pedagogic Agent prompt.

    The model is repeatedly told (here and in the system prompt) that this block
    is private and must not be quoted verbatim.
    """
    if anchor is None or anchor.is_empty():
        return (
            "\n[TUTOR-ONLY ANCHOR]\n"
            "No teaching anchor available yet. Ask one focused opening question "
            "and avoid giving any factual hint.\n[/TUTOR-ONLY ANCHOR]\n"
        )

    lines: list[str] = ["", "[TUTOR-ONLY ANCHOR — DO NOT REVEAL OR QUOTE VERBATIM]"]
    if anchor.original_question:
        lines.append(f"Original student question: {anchor.original_question}")
    if anchor.target_explanation:
        lines.append("")
        lines.append("Target explanation (private):")
        lines.append(anchor.target_explanation.strip())
    if anchor.key_facts:
        lines.append("")
        lines.append("Key facts the student should reach:")
        for fact in anchor.key_facts:
            lines.append(f"- {fact}")
    if anchor.misconceptions:
        lines.append("")
        lines.append("Common misconceptions to probe:")
        for m in anchor.misconceptions:
            lines.append(f"- {m}")
    if anchor.scaffold_questions:
        lines.append("")
        lines.append("Scaffold questions (use in order, escalate as needed):")
        for i, q in enumerate(anchor.scaffold_questions, start=1):
            lines.append(f"{i}. {q}")
    if anchor.citations:
        lines.append("")
        lines.append("Citations (course material backing the anchor):")
        for c in anchor.citations:
            lines.append(f"- {c}")
    lines.append("[/TUTOR-ONLY ANCHOR]")
    lines.append("")
    return "\n".join(lines)


_REQUIRED_ANCHOR_KEYS = (
    "target_explanation",
    "key_facts",
    "misconceptions",
    "scaffold_questions",
    "citations",
)


def parse_anchor_from_json(raw: str, original_question: str = "") -> TeachingAnchor:
    """Parse the Answer Agent's final JSON output into a TeachingAnchor.

    Tolerates fenced ```json code blocks and minor formatting noise via
    ``parse_json_response``. Missing fields default to empty.
    """
    parsed = parse_json_response(raw)
    if not isinstance(parsed, dict):
        raise ValueError(
            f"Expected a JSON object for the teaching anchor, got {type(parsed).__name__}"
        )
    data = {k: parsed.get(k) for k in _REQUIRED_ANCHOR_KEYS}
    data["original_question"] = parsed.get("original_question") or original_question
    return TeachingAnchor.from_dict(data)


def active_scaffold_question(
    anchor: Optional[TeachingAnchor], index: int
) -> Optional[str]:
    """Return the scaffold question the tutor should focus on this turn.

    Clamps the index to the available range so an over-advanced session still
    points at the last (most specific) scaffold question.
    """
    if anchor is None or not anchor.scaffold_questions:
        return None
    safe_index = min(max(0, int(index)), len(anchor.scaffold_questions) - 1)
    return anchor.scaffold_questions[safe_index]


def extract_pedagogic_signals(text: str) -> tuple[str, Optional[str], bool]:
    """Strip control signals from a Pedagogic Agent reply.

    Returns ``(clean_text, topic_shift_query_or_None, advance_flag)``. The
    signals are tutor-internal and must never reach the student, so the orchestrator
    calls this before displaying or summarizing the move.
    """
    if not text:
        return "", None, False

    topic_shift_query: Optional[str] = None
    shift_match = _TOPIC_SHIFT_RE.search(text)
    if shift_match is not None:
        candidate = shift_match.group(1).strip()
        topic_shift_query = candidate or None

    advance_flag = bool(_ADVANCE_SCAFFOLD_RE.search(text))

    clean = _TOPIC_SHIFT_RE.sub("", text)
    clean = _ADVANCE_SCAFFOLD_RE.sub("", clean)
    clean = clean.strip()

    return clean, topic_shift_query, advance_flag


