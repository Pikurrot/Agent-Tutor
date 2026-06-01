from __future__ import annotations

from typing import Optional

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

from tutor.modules.agent.agent_tracing import AgentTraceCallback
from tutor.modules.agent.pedagogy import (
    TeachingAnchor,
    parse_anchor_from_json,
)
from tutor.modules.models.base import BaseModel
from tutor.modules.models.qwen import LangChainQwen
from tutor.modules.retrieval.RAG import RAGModule, SlideRetrieverTool


_JSON_INSTRUCTION = (
    'Final Answer must be a single JSON object with EXACTLY these keys:\n'
    '  "target_explanation" (string, 3-8 sentences): the full correct answer, grounded in the retrieved context.\n'
    '  "key_facts" (list of short strings): the core concepts the student must reach.\n'
    '  "misconceptions" (list of short strings): common student mistakes about this topic.\n'
    '  "scaffold_questions" (list of strings): 2-4 Socratic questions in order of increasing specificity that guide a student toward the target_explanation.\n'
    '  "citations" (list of strings, format "<document_name> · slide <N>"): supporting course slides.\n'
    'Output ONLY the JSON object. No prose, no markdown fences, no commentary before or after.'
)


def _build_answer_agent_template(rag_module: RAGModule, anchor_block: str = "") -> str:
    """Build the ReAct template used by the Answer Agent.

    Same skeleton as ``build_rag_agent`` in ``agent.py`` but the final step
    produces a structured ``TeachingAnchor`` JSON instead of a free-form answer.
    """
    docs_list = "\n".join([f"- \"{name}\"" for name in rag_module.retriever.documents_names])
    safe_anchor = anchor_block.replace("{", "{{").replace("}", "}}") if anchor_block else ""

    return (
        "You are the Answer Agent for a Socratic tutoring system. Your job is to "
        "build a private TeachingAnchor (a structured lesson plan) for the question below, "
        "grounded in the course material.\n\n"
        "Retrieval instructions:\n"
        "- When the question involves more than one concept, split retrieval into multiple steps, "
        "making one search query after the other.\n"
        "- When the retrieved context does not contain the expected information, rephrase the query.\n"
        "- Always ground your final answer in the retrieved context; do not invent facts.\n\n"
        "You have access to the following documents:\n"
        + docs_list
        + "\n\nAlways use one of the available tools before producing the Final Answer.\n"
        "You have access to the following tools:\n\n{tools}"
        + safe_anchor
        + "\n\nUse the following format:\n\n"
        "Question: the input question you must build the anchor for\n"
        "Thought: you should always think about what to do\n"
        "Action: the action to take, should be one of [{tool_names}]\n"
        "Action Input: the input to the action\n"
        "Observation: the result of the action\n"
        "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
        "Thought: I now have enough information to build the teaching anchor\n"
        "Final Answer: <JSON object as described below>\n\n"
        + _JSON_INSTRUCTION
        + "\n\nBegin!\n\n"
        "Question: {input}\n"
        "Thought:{agent_scratchpad}"
    )


def build_answer_agent(
    qwen_model: BaseModel,
    rag_module: RAGModule,
    config: dict,
    *,
    slide_manager: Optional[SlideRetrieverTool] = None,
    anchor_block: str = "",
):
    """Construct the Answer Agent executor.

    If ``slide_manager`` is provided we reuse it so retrieved slides collected
    when the orchestrator rebuilds the anchor bubble up to the UI.
    """
    agent_cfg = config.get("agent_config", {}) or {}
    pedagogic_cfg = config.get("pedagogic_config", {}) or {}
    answer_max_new_tokens = int(
        pedagogic_cfg.get("anchor_max_new_tokens", agent_cfg.get("max_new_tokens", 1024))
    )
    max_iterations = int(pedagogic_cfg.get("answer_max_iterations", agent_cfg.get("max_iterations", 4)))

    slide_tool_manager = slide_manager if slide_manager is not None else SlideRetrieverTool(rag_module)

    llm = LangChainQwen(
        qwen_model=qwen_model,
        slide_manager=slide_tool_manager,
        agent_max_new_tokens=answer_max_new_tokens,
    )

    tools = [
        slide_tool_manager.get_tool("Search_All_Course_Context"),
        slide_tool_manager.get_tool("Search_Document_Context"),
        slide_tool_manager.get_tool("Retrieve_Slide_Context"),
    ]

    template = _build_answer_agent_template(rag_module, anchor_block=anchor_block)
    prompt = PromptTemplate.from_template(template)

    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=max_iterations,
    )
    return executor, slide_tool_manager


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
    """Run the Answer Agent for one question and return a TeachingAnchor.

    Strategy:
    1. Build the agent (optionally biased toward refining a prior anchor).
    2. Invoke once and parse the JSON Final Answer.
    3. On parse failure, retry once with a strict reminder prompt before giving up.
    4. If ``prior_anchor`` is provided, merge the new anchor on top of it.
    """
    anchor_block = ""
    if prior_anchor is not None and not prior_anchor.is_empty():
        from tutor.modules.agent.pedagogy import format_anchor_for_pedagogic_prompt

        anchor_block = (
            "\n\nYou are refining an existing anchor. Preserve correct facts, "
            "expand or correct as needed, and include all relevant slides as citations.\n"
            + format_anchor_for_pedagogic_prompt(prior_anchor)
        )

    executor, slide_tool = build_answer_agent(
        qwen_model,
        rag_module,
        config,
        slide_manager=slide_manager,
        anchor_block=anchor_block,
    )

    trace_cb = AgentTraceCallback(agent_name="AnswerAgent") if debug else None
    exec_callbacks = list(callbacks or [])
    if trace_cb is not None:
        exec_callbacks.append(trace_cb)

    response = executor.invoke(
        {"input": question},
        config={"callbacks": exec_callbacks} if exec_callbacks else None,
    )
    raw_output = response.get("output", "")
    parse_retry_count = 0
    parse_failed = False
    notes: list[str] = []

    try:
        anchor = parse_anchor_from_json(raw_output, original_question=question)
    except Exception:
        parse_retry_count = 1
        retry_prompt = (
            f"{question}\n\n"
            "Your previous Final Answer was not valid JSON. Output a single JSON object "
            "with keys target_explanation, key_facts, misconceptions, scaffold_questions, "
            "citations. No markdown fences, no extra text."
        )
        retry_response = executor.invoke(
            {"input": retry_prompt},
            config={"callbacks": exec_callbacks} if exec_callbacks else None,
        )
        try:
            anchor = parse_anchor_from_json(
                retry_response.get("output", ""), original_question=question
            )
            notes.append("Recovered after one JSON parse retry.")
            raw_output = retry_response.get("output", raw_output)
        except Exception:
            parse_failed = True
            notes.append("Fallback: used first raw output as target_explanation.")
            anchor = TeachingAnchor(
                original_question=question,
                target_explanation=str(raw_output).strip(),
            )

    if prior_anchor is not None and not prior_anchor.is_empty():
        anchor = prior_anchor.merge_with(anchor)

    trace: Optional[dict] = None
    if trace_cb is not None:
        trace = trace_cb.to_trace_dict(
            parse_retry_count=parse_retry_count,
            parse_failed=parse_failed,
            notes=notes,
        )
        trace["raw_output"] = str(raw_output)

    return anchor, slide_tool, trace
