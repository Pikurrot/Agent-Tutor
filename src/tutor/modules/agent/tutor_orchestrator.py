from __future__ import annotations

from typing import Any, Optional

from tutor.modules.agent.answer_agent import run_answer_agent
from tutor.modules.agent.pedagogic_agent import run_pedagogic_agent
from tutor.modules.agent.pedagogy import TeachingSession
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
    2. Invoke the Pedagogic Agent (it may call Query_Expert_Agent mid-turn to
       refresh the anchor on topic shifts).
    3. Roll the public conversation memory using only the student-visible
       tutoring move (the private anchor never enters the summary).

    Returns ``(tutoring_move, retrieved_slides_for_ui, new_memory, session)``.
    """
    mem = memory if memory is not None else ConversationMemory.empty()
    sess = session if session is not None else TeachingSession.empty()

    slide_manager = SlideRetrieverTool(rag_module)

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
    else:
        answer_trace = None

    move, _, pedagogic_trace = run_pedagogic_agent(
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

    new_memory = _roll_dialogue_memory(config, mem, prompt, move, qwen_model)

    sess.hints_used = int(sess.hints_used) + 1

    debug_data: Optional[dict[str, Any]] = None
    if debug:
        debug_data = {
            "teaching_anchor_snapshot": sess.anchor.to_dict() if sess.anchor is not None else None,
            "answer_agent_trace": answer_trace,
            "pedagogic_agent_trace": pedagogic_trace,
            "session_status": sess.status,
            "hints_used": int(sess.hints_used),
        }

    return move, list(slide_manager.retrieved_slides), new_memory, sess, debug_data


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
