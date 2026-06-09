from __future__ import annotations

from typing import Any, Optional

from tutor.modules.agent.answer_agent import run_answer_agent
from tutor.modules.agent.pedagogic_agent import run_pedagogic_agent
from tutor.modules.agent.pedagogy import TeachingSession, extract_pedagogic_signals
from tutor.modules.agent.summarizer import ConversationMemory, roll_memory
from tutor.modules.models.base import BaseModel
from tutor.modules.retrieval.RAG import RAGModule, SlideRetrieverTool


def run_tutor_turn(
    qwen_model: BaseModel,
    rag_module: RAGModule,
    config: dict,
    prompt: str,
    *,
    memory: Optional[ConversationMemory] = None,
    session: Optional[TeachingSession] = None,
    callbacks: Optional[list] = None,
    debug: bool = False,
) -> tuple[
    str,
    list[dict[str, Any]],
    ConversationMemory,
    TeachingSession,
    Optional[dict[str, Any]],
]:
    """Run a single tutoring turn.

    Steps:
    1. If the session has no anchor, bootstrap one by invoking the Answer Agent.
    2. Invoke the Pedagogic Agent (direct chat completion) for the active
       scaffold step.
    3. Handle control signals it may emit:
       - ``[TOPIC_SHIFT: X]`` -> rebuild the anchor via the Answer Agent, reset
         the scaffold index, and re-run the Pedagogic Agent once.
       - ``[ADVANCE_SCAFFOLD]`` -> advance to the next scaffold question.
    4. Roll the public conversation memory using only the signal-stripped,
       student-visible tutoring move (the private anchor never enters the summary).

    Returns ``(tutoring_move, retrieved_slides_for_ui, new_memory, session, debug_data)``.
    """
    mem = memory if memory is not None else ConversationMemory.empty()
    sess = session if session is not None else TeachingSession.empty()

    slide_manager = SlideRetrieverTool(rag_module)

    if sess.anchor is not None and not sess.anchor.is_valid():
        sess.anchor = None

    if not sess.has_anchor():
        anchor, _, answer_trace = run_answer_agent(
            qwen_model,
            rag_module,
            config,
            question=prompt,
            prior_anchor=None,
            slide_manager=slide_manager,
            callbacks=callbacks,
            debug=debug,
        )
        sess.anchor = anchor
        sess.current_scaffold_index = 0
    else:
        answer_trace = None

    raw_move, _, pedagogic_trace = run_pedagogic_agent(
        qwen_model,
        rag_module,
        config,
        prompt,
        sess,
        memory=mem,
        slide_manager=slide_manager,
        callbacks=callbacks,
        debug=debug,
    )

    clean_move, shift_query, advance = extract_pedagogic_signals(raw_move)

    if shift_query:
        # Student moved to a new topic the anchor does not cover: rebuild it,
        # reset scaffolding, and let the Pedagogic Agent take one fresh pass.
        new_anchor, _, answer_trace = run_answer_agent(
            qwen_model,
            rag_module,
            config,
            question=shift_query,
            prior_anchor=None,
            slide_manager=slide_manager,
            callbacks=callbacks,
            debug=debug,
        )
        sess.anchor = new_anchor
        sess.current_scaffold_index = 0

        raw_move, _, pedagogic_trace = run_pedagogic_agent(
            qwen_model,
            rag_module,
            config,
            prompt,
            sess,
            memory=mem,
            slide_manager=slide_manager,
            callbacks=callbacks,
            debug=debug,
        )
        # Ignore any further shift signal this turn to avoid loops; still honor advance.
        clean_move, _, advance = extract_pedagogic_signals(raw_move)

    if advance and sess.anchor is not None and sess.anchor.scaffold_questions:
        last_index = len(sess.anchor.scaffold_questions) - 1
        sess.current_scaffold_index = min(sess.current_scaffold_index + 1, last_index)

    new_memory = _roll_dialogue_memory(config, mem, prompt, clean_move, qwen_model)

    sess.hints_used = int(sess.hints_used) + 1

    debug_data: Optional[dict[str, Any]] = None
    if debug:
        debug_data = {
            "teaching_anchor_snapshot": sess.anchor.to_dict() if sess.anchor is not None else None,
            "answer_agent_trace": answer_trace,
            "pedagogic_agent_trace": pedagogic_trace,
            "session_status": sess.status,
            "hints_used": int(sess.hints_used),
            "current_scaffold_index": int(sess.current_scaffold_index),
        }

    return clean_move, list(slide_manager.retrieved_slides), new_memory, sess, debug_data


def _roll_dialogue_memory(
    config: dict,
    memory: ConversationMemory,
    user_prompt: str,
    assistant_move: str,
    qwen_model: BaseModel,
) -> ConversationMemory:
    agent_cfg = config.get("agent_config", {}) or {}
    memory_cfg = agent_cfg.get("memory", {}) or {}
    if not bool(memory_cfg.get("enabled", True)):
        return memory
    return roll_memory(
        memory,
        new_user=user_prompt,
        new_assistant=assistant_move,
        model=qwen_model,
        summary_max_new_tokens=int(memory_cfg.get("summary_max_new_tokens", 512)),
    )
