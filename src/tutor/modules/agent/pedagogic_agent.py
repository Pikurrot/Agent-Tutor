from __future__ import annotations

from typing import Optional

from tutor.modules.agent.pedagogy import (
    TeachingSession,
    active_scaffold_question,
    format_anchor_for_pedagogic_prompt,
)
from tutor.modules.agent.summarizer import ConversationMemory, format_memory_for_prompt
from tutor.modules.models.base import BaseModel
from tutor.modules.retrieval.RAG import RAGModule, SlideRetrieverTool


_SOCRATIC_RULES = """You are a Socratic tutor for a university course. The TUTOR-ONLY ANCHOR above (target_explanation, key_facts, misconceptions, scaffold_questions) is PRIVATE. Your goal is to make the student REASON their way to the answer, not to explain the answer to them.

ACTIVE STEP: focus this turn on guiding the student through the single scaffold question labeled "ACTIVE SCAFFOLD QUESTION" below. Do NOT explain or hint at concepts that belong to LATER scaffold questions or to the full target_explanation.

Hard limits:
- Never state the target_explanation, and never reveal more than ONE key_fact in a single turn.
- On a broad opening question, do NOT lecture. Instead, open the ACTIVE scaffold question as a concrete thought experiment the student can attempt (a small example, a "what happens if..." setup).
- A "mini-explanation" may ONLY clarify a prerequisite the student is visibly missing (e.g. a definition they explicitly ask about) — never the core idea the active scaffold question targets.
- If the student is wrong, correct minimally and let them retry; if they are close, affirm and nudge.
- Reveal the full answer ONLY if the student explicitly gives up ("I don't know", "just tell me", "I give up") or has clearly stalled after several turns; then connect it to what they already reasoned ("reveal with credit").

Signals (append on their OWN line at the very end; they are stripped before the student sees them):
- If the student has correctly answered the ACTIVE scaffold question, append [ADVANCE_SCAFFOLD].
- If the student shifts to a clearly different topic the anchor does not cover, append [TOPIC_SHIFT: <short description of the new topic>] and ask nothing else this turn.

Style: warm, conversational, plain prose, roughly 30-100 words (up to ~180 only for a reveal-with-credit), usually ending with one prompt that invites the student to take the next reasoning step. Do not announce the move you are making; just make it. Do not use bullet lists unless they genuinely help."""


def _build_pedagogic_prompt(
    session: TeachingSession,
    memory_block: str,
    user_prompt: str,
) -> str:
    """Assemble the direct chat-completion prompt for one tutoring turn.

    Replaces the previous ReAct template. No Thought/Action/Final Answer
    scaffolding, so the model cannot trip the ReAct parser.
    """
    anchor_block = format_anchor_for_pedagogic_prompt(session.anchor)
    active_question = active_scaffold_question(session.anchor, session.current_scaffold_index)

    parts: list[str] = [
        _SOCRATIC_RULES,
        "",
        anchor_block.strip(),
    ]

    if active_question:
        parts.extend(
            [
                "",
                f"ACTIVE SCAFFOLD QUESTION (step {session.current_scaffold_index + 1}): {active_question}",
            ]
        )
    else:
        parts.extend(
            [
                "",
                "ACTIVE SCAFFOLD QUESTION: (none available — guide with one focused opening question).",
            ]
        )

    if memory_block.strip():
        parts.extend(["", memory_block.strip()])

    parts.extend(
        [
            "",
            "Student message:",
            user_prompt.strip(),
            "",
            "Write the tutor's next move directly to the student (plain conversational text). "
            "Append any control signal on its own final line.",
        ]
    )
    return "\n".join(parts)


def run_pedagogic_agent(
    qwen_model: BaseModel,
    rag_module: RAGModule,
    config: dict,
    prompt: str,
    session: TeachingSession,
    *,
    memory: Optional[ConversationMemory] = None,
    slide_manager: Optional[SlideRetrieverTool] = None,
    callbacks: Optional[list] = None,
    debug: bool = False,
) -> tuple[str, SlideRetrieverTool, Optional[dict]]:
    """Run the Pedagogic Agent for one student turn as a direct chat completion.

    Returns ``(raw_move, slide_manager, trace)``. The raw move may contain
    control signals ([ADVANCE_SCAFFOLD] / [TOPIC_SHIFT: ...]); the orchestrator
    is responsible for extracting and stripping them before display.

    The slide manager is created/reused but the Pedagogic Agent itself does no
    retrieval — slides only accumulate when the orchestrator rebuilds the anchor
    via the Answer Agent on a topic shift.
    """
    slide_tool = slide_manager if slide_manager is not None else SlideRetrieverTool(rag_module)

    agent_cfg = config.get("agent_config", {}) or {}
    pedagogic_cfg = config.get("pedagogic_config", {}) or {}
    move_max_new_tokens = int(pedagogic_cfg.get("move_max_new_tokens", 256))
    memory_enabled = bool((agent_cfg.get("memory", {}) or {}).get("enabled", True))

    memory_block = (
        format_memory_for_prompt(memory) if memory_enabled and memory is not None else ""
    )

    direct_prompt = _build_pedagogic_prompt(session, memory_block, prompt)
    move = str(
        qwen_model.generate(direct_prompt, max_new_tokens=move_max_new_tokens)
    ).strip()

    trace: Optional[dict] = None
    if debug:
        active_question = active_scaffold_question(
            session.anchor, session.current_scaffold_index
        )
        notes = [
            "Direct chat completion (no ReAct).",
            f"Active scaffold index: {session.current_scaffold_index}",
            f"Active scaffold question: {active_question or '(none)'}",
        ]
        trace = {
            "agent_name": "PedagogicAgent",
            "steps": [],
            "raw_output": move,
            "chain_length": 0,
            "used_fallback": False,
            "parse_retry_count": 0,
            "parse_failed": False,
            "notes": notes,
        }

    return move, slide_tool, trace
