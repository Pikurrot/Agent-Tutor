from __future__ import annotations

from typing import Optional

from tutor.modules.agent.pedagogy import (
    TeachingAnchor,
    format_anchor_for_pedagogic_prompt,
    parse_anchor_from_json,
)
from tutor.modules.models.base import BaseModel
from tutor.modules.retrieval.RAG import RAGModule, SlideRetrieverTool


_JSON_INSTRUCTION = (
    "Output a single JSON object with EXACTLY these keys:\n"
    '  "target_explanation" (string, 3-8 sentences): the full correct answer, grounded in the retrieved context.\n'
    '  "key_facts" (list of short strings): the core concepts the student must reach.\n'
    '  "misconceptions" (list of short strings): common student mistakes about this topic.\n'
    '  "scaffold_questions" (list of strings): 2-4 Socratic questions in order of increasing specificity that guide a student toward the target_explanation.\n'
    '  "citations" (list of strings, format "<document_name> · slide <N>"): supporting course slides.\n'
    "Output ONLY the JSON object. No prose, no markdown fences, no commentary before or after."
)


def _build_direct_anchor_prompt(
    rag_module: RAGModule,
    question: str,
    context_block: str,
    *,
    anchor_block: str = "",
) -> str:
    docs_list = "\n".join(
        [f'- "{name}"' for name in rag_module.retriever.documents_names]
    )
    parts = [
        "You are the Answer Agent for a Socratic tutoring system. Your job is to build a "
        "private TeachingAnchor (a structured lesson plan) for the question below, "
        "grounded in the course material provided.\n",
        "Course documents:\n" + docs_list,
    ]
    if anchor_block.strip():
        parts.extend(["", anchor_block.strip()])
    parts.extend(
        [
            "",
            "Retrieved course context:",
            context_block.strip() if context_block.strip() else "(no slides retrieved)",
            "",
            "Student question:",
            question.strip(),
            "",
            _JSON_INSTRUCTION,
        ]
    )
    return "\n".join(parts)


def _strict_json_retry_suffix() -> str:
    return (
        "\n\nYour previous response was not valid JSON. Output a single JSON object with keys "
        "target_explanation, key_facts, misconceptions, scaffold_questions, citations. "
        "No markdown fences, no extra text."
    )


def run_answer_agent(
    qwen_model: BaseModel,
    rag_module: RAGModule,
    config: dict,
    question: str,
    *,
    prior_anchor: Optional[TeachingAnchor] = None,
    slide_manager: Optional[SlideRetrieverTool] = None,
    callbacks: Optional[list] = None,
    debug: bool = False,
) -> tuple[TeachingAnchor, SlideRetrieverTool, Optional[dict]]:
    """Build a TeachingAnchor via one RAG retrieval and one direct Qwen JSON call.

    Replaces the previous ReAct loop (which often hit iteration limits and poisoned
    ``target_explanation``). On total parse failure returns an empty anchor.
    """
    del callbacks  # reserved for API compatibility; direct path has no LangChain executor

    agent_cfg = config.get("agent_config", {}) or {}
    pedagogic_cfg = config.get("pedagogic_config", {}) or {}
    anchor_max_new_tokens = int(
        pedagogic_cfg.get("anchor_max_new_tokens", agent_cfg.get("max_new_tokens", 1024))
    )

    slide_tool = slide_manager if slide_manager is not None else SlideRetrieverTool(rag_module)

    retrieved_data, _metadata = rag_module.retrieve(question)
    context_block = slide_tool.prepare_retrieved_data(retrieved_data)

    anchor_block = ""
    if prior_anchor is not None and prior_anchor.is_valid():
        anchor_block = (
            "You are refining an existing anchor. Preserve correct facts, "
            "expand or correct as needed, and include all relevant slides as citations.\n"
            + format_anchor_for_pedagogic_prompt(prior_anchor)
        )

    base_prompt = _build_direct_anchor_prompt(
        rag_module, question, context_block, anchor_block=anchor_block
    )

    parse_retry_count = 0
    parse_failed = False
    notes: list[str] = ["Direct RAG retrieval + single chat completion (no ReAct)."]
    raw_output = ""

    def _generate(prompt: str) -> str:
        return str(
            qwen_model.generate(prompt, max_new_tokens=anchor_max_new_tokens)
        ).strip()

    def _try_parse(raw: str) -> TeachingAnchor:
        anchor = parse_anchor_from_json(raw, original_question=question)
        if not anchor.is_valid():
            raise ValueError("Parsed anchor failed validity checks")
        return anchor

    anchor: TeachingAnchor
    try:
        raw_output = _generate(base_prompt)
        anchor = _try_parse(raw_output)
    except Exception:
        parse_retry_count = 1
        try:
            raw_output = _generate(base_prompt + _strict_json_retry_suffix())
            anchor = _try_parse(raw_output)
            notes.append("Recovered after one JSON parse retry.")
        except Exception:
            parse_failed = True
            notes.append("Parse failed; returning empty anchor (no poison fallback).")
            anchor = TeachingAnchor(original_question=question)

    if prior_anchor is not None and prior_anchor.is_valid() and anchor.is_valid():
        anchor = prior_anchor.merge_with(anchor)

    trace: Optional[dict] = None
    if debug:
        trace = {
            "agent_name": "AnswerAgent",
            "steps": [],
            "raw_output": raw_output,
            "chain_length": 0,
            "used_fallback": False,
            "parse_retry_count": parse_retry_count,
            "parse_failed": parse_failed,
            "notes": notes,
            "anchor_valid": anchor.is_valid(),
        }

    return anchor, slide_tool, trace
