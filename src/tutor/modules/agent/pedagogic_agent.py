from __future__ import annotations

from typing import Optional

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool

from tutor.modules.agent.agent_tracing import AgentTraceCallback
from tutor.modules.agent.answer_agent import run_answer_agent
from tutor.modules.agent.pedagogy import (
    TeachingAnchor,
    TeachingSession,
    anchor_summary_for_tool_observation,
    format_anchor_for_pedagogic_prompt,
)
from tutor.modules.agent.summarizer import ConversationMemory, format_memory_for_prompt
from tutor.modules.models.base import BaseModel
from tutor.modules.models.qwen import LangChainQwen
from tutor.modules.retrieval.RAG import RAGModule, SlideRetrieverTool


_SOCRATIC_RULES = """You are a Socratic tutor for a university course. The TUTOR-ONLY ANCHOR block above contains the correct answer, supporting facts, and notes — they are private. Your job is to help the student understand, not to quiz them.

Be warm, patient, and collaborative. Read each student message together with the conversation memory so far, and pick the tutoring move that best helps THEM at THIS moment — not the move that is most "Socratic" on principle. Do NOT turn every reply into a question.

Tutoring moves available to you (mix them naturally across turns):
- Guiding question: when the student has some traction; ask one focused question that opens the next step. Use sparingly, not every turn.
- Small hint: when the student is stuck or close but missing a specific piece; reveal a clue, a definition, or a property — never the full target explanation.
- Mini-explanation: when the student lacks background needed to reason further; explain ONE supporting idea, analogy, or intuition in 1-3 sentences, then invite them to apply it.
- Gentle correction: when something the student said is wrong; name what is off, briefly state the right framing for THAT piece, and let them retry.
- Affirmation + nuance: when the student is right or nearly right; confirm what they got and add at most one nuance from the anchor.
- Reveal with credit: when the student explicitly gives up ("I don't know", "just tell me", "I give up"), or you judge they have stalled despite several hints, give the answer succinctly AND frame it around what they already said — highlight the parts of their reasoning that were on the right track and connect the final answer to those parts. Make them feel their thinking was meaningful, even when they did not reach the answer alone.

How to choose a move:
- Look at the latest student message and the prior turns. Ask yourself what they need most right now: a nudge, a fact, an explanation, a correction, validation, or the answer.
- Then deliver that single thing. A turn may combine moves naturally (e.g., affirm one thing, correct another, then ask one question) — keep it conversational, not a checklist.
- Vary your moves across consecutive turns; avoid back-to-back questions when a hint or short explanation would give the student more traction.

Style:
- Conversational plain prose, roughly 30-120 words. Up to ~180 only for the reveal-with-credit case.
- Do not announce or enumerate the move you are making; just make it.
- No bullet lists unless they genuinely help readability.

Hard constraints:
- Never paste the anchor's target_explanation verbatim — always reformulate in your own voice.
- Never read the misconceptions or scaffold_questions lists out as a literal enumeration; treat them as inner guidance.
- Reveal the full answer ONLY in the "Reveal with credit" situation described above.

When to call Query_Expert_Agent:
- The student shifts to a new topic and the current anchor no longer fits.
- You need a precise fact that is missing from the anchor.
- Otherwise: DO NOT call the tool. Reply directly with a tutoring move."""


def _build_pedagogic_template(
    anchor: Optional[TeachingAnchor],
    memory_block: str,
) -> str:
    anchor_block = format_anchor_for_pedagogic_prompt(anchor)
    safe_anchor = anchor_block.replace("{", "{{").replace("}", "}}")
    safe_memory = memory_block.replace("{", "{{").replace("}", "}}") if memory_block else ""

    return (
        _SOCRATIC_RULES
        + "\n\n"
        + safe_anchor
        + safe_memory
        + "\nYou have access to the following tools:\n\n{tools}\n\n"
        "Use the following format:\n\n"
        "Question: the student's latest message\n"
        "Thought: read the student's state (confused, partially right, fully right, asking outright, surrendering), decide which tutoring move helps them most right now, and whether you need to call Query_Expert_Agent\n"
        "Action: the action to take, should be one of [{tool_names}]\n"
        "Action Input: the input to the action\n"
        "Observation: the result of the action\n"
        "... (Thought/Action/Action Input/Observation can repeat but most turns should issue zero actions)\n"
        "Thought: I now know the right tutoring move\n"
        "Final Answer: <the tutoring move, written directly to the student>\n\n"
        "Begin!\n\n"
        "Question: {input}\n"
        "Thought:{agent_scratchpad}"
    )


def _build_direct_pedagogic_prompt(
    anchor: Optional[TeachingAnchor],
    memory_block: str,
    user_prompt: str,
) -> str:
    """Prompt used as a fallback when ReAct parsing fails.

    This path keeps the same pedagogic behavior but does not require tool-call
    output formatting, preventing iteration-limit failures from parser drift.
    """
    anchor_block = format_anchor_for_pedagogic_prompt(anchor)
    parts: list[str] = [
        _SOCRATIC_RULES,
        "",
        anchor_block.strip(),
    ]
    if memory_block.strip():
        parts.extend(["", memory_block.strip()])
    parts.extend(
        [
            "",
            "Student message:",
            user_prompt.strip(),
            "",
            "Write the tutor response directly to the student.",
            "Do not include Thought/Action/Observation/Final Answer labels.",
            "Return plain conversational text only.",
        ]
    )
    return "\n".join(parts)


def _make_query_expert_tool(
    qwen_model: BaseModel,
    rag_module: RAGModule,
    config: dict,
    session: TeachingSession,
    slide_manager: SlideRetrieverTool,
) -> Tool:
    """Build the single tool the Pedagogic Agent uses to consult the Answer Agent.

    Side effect: refreshes ``session.anchor`` by merging the freshly-built anchor
    into the existing one (or replacing if there was none).
    """

    def _query_expert(query: str) -> str:
        question = (query or "").strip()
        if not question:
            return "Query_Expert_Agent requires a non-empty query string."
        new_anchor, _, _ = run_answer_agent(
            qwen_model,
            rag_module,
            config,
            question=question,
            prior_anchor=session.anchor,
            slide_manager=slide_manager,
            debug=False,
        )
        if session.anchor is None:
            session.anchor = new_anchor
        else:
            session.anchor = session.anchor.merge_with(new_anchor)
        return anchor_summary_for_tool_observation(session.anchor)

    return Tool.from_function(
        name="Query_Expert_Agent",
        func=_query_expert,
        description=(
            "Consult the expert/answer agent to retrieve and synthesize new course material "
            "for the student's current doubt. Use ONLY when the student shifts topic or the "
            "current anchor is missing a fact you need to give an accurate hint. "
            "Input: a focused natural-language query string."
        ),
    )


def build_pedagogic_agent(
    qwen_model: BaseModel,
    rag_module: RAGModule,
    config: dict,
    session: TeachingSession,
    *,
    memory: Optional[ConversationMemory] = None,
    slide_manager: Optional[SlideRetrieverTool] = None,
):
    agent_cfg = config.get("agent_config", {}) or {}
    pedagogic_cfg = config.get("pedagogic_config", {}) or {}
    move_max_new_tokens = int(pedagogic_cfg.get("move_max_new_tokens", 256))
    max_iterations = int(pedagogic_cfg.get("max_iterations", 3))
    memory_enabled = bool((agent_cfg.get("memory", {}) or {}).get("enabled", True))

    slide_tool_manager = slide_manager if slide_manager is not None else SlideRetrieverTool(rag_module)

    llm = LangChainQwen(
        qwen_model=qwen_model,
        slide_manager=slide_tool_manager,
        agent_max_new_tokens=move_max_new_tokens,
    )

    expert_tool = _make_query_expert_tool(
        qwen_model, rag_module, config, session, slide_tool_manager
    )
    tools = [expert_tool]

    memory_block = (
        format_memory_for_prompt(memory) if memory_enabled and memory is not None else ""
    )

    template = _build_pedagogic_template(session.anchor, memory_block)
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
    """Run the Pedagogic Agent for one student turn.

    ``session`` may be mutated by the agent through ``Query_Expert_Agent``.
    Returns the tutoring move plus the slide manager used (which has
    accumulated any slides retrieved during the turn).

    No post-hoc redaction: with reveal-with-credit as a legitimate move, the
    prompt itself is the authoritative gatekeeper for when to disclose the
    target explanation.
    """
    executor, slide_tool = build_pedagogic_agent(
        qwen_model,
        rag_module,
        config,
        session,
        memory=memory,
        slide_manager=slide_manager,
    )

    pedagogic_cfg = config.get("pedagogic_config", {}) or {}
    move_max_new_tokens = int(pedagogic_cfg.get("move_max_new_tokens", 256))
    memory_block = (
        format_memory_for_prompt(memory)
        if bool((config.get("agent_config", {}) or {}).get("memory", {}).get("enabled", True))
        and memory is not None
        else ""
    )

    def _fallback_direct_generation() -> str:
        direct_prompt = _build_direct_pedagogic_prompt(session.anchor, memory_block, prompt)
        return str(
            qwen_model.generate(direct_prompt, max_new_tokens=move_max_new_tokens)
        ).strip()

    trace_cb = AgentTraceCallback(agent_name="PedagogicAgent") if debug else None
    exec_callbacks = list(callbacks or [])
    if trace_cb is not None:
        exec_callbacks.append(trace_cb)

    used_fallback = False
    notes: list[str] = []
    try:
        response = executor.invoke(
            {"input": prompt},
            config={"callbacks": exec_callbacks} if exec_callbacks else None,
        )
        move = str(response.get("output", "")).strip()
    except Exception as e:
        notes.append(f"ReAct execution raised exception: {e}")
        move = ""

    # LangChain ReAct can fail when the model answers in plain prose without
    # Action/Final Answer tags; recover gracefully instead of returning an
    # iteration-limit/system error to the student.
    if (not move) or ("Agent stopped due to iteration limit or time limit." in move):
        used_fallback = True
        notes.append("Used direct-generation fallback.")
        move = _fallback_direct_generation()

    trace: Optional[dict] = None
    if trace_cb is not None:
        trace = trace_cb.to_trace_dict(
            used_fallback=used_fallback,
            notes=notes,
        )
        if not trace.get("raw_output"):
            trace["raw_output"] = move

    return move, slide_tool, trace
