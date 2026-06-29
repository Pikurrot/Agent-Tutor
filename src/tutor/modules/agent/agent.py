from __future__ import annotations
import json
from typing import Any, Optional

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks.base import BaseCallbackHandler

from tutor.modules.models.base import BaseModel
from tutor.modules.models.qwen import LangChainQwen
from tutor.modules.retrieval.RAG import SlideRetrieverTool, RAGModule
from tutor.modules.agent.summarizer import (
    ConversationMemory,
    format_memory_for_prompt,
)


def build_rag_agent(
    qwen_model: BaseModel,
    rag_module: RAGModule,
    config: dict,
    memory: Optional[ConversationMemory] = None,
):
    slide_tool_manager = SlideRetrieverTool(rag_module)
    agent_cfg = config.get("agent_config", {})
    agent_max_new_tokens = int(agent_cfg.get("max_new_tokens", 1024))
    memory_cfg = agent_cfg.get("memory", {}) or {}
    memory_enabled = bool(memory_cfg.get("enabled", True))

    llm = LangChainQwen(
        qwen_model=qwen_model,
        slide_manager=slide_tool_manager,
        agent_max_new_tokens=agent_max_new_tokens,
    )

    tools = [
        slide_tool_manager.get_tool("Search_All_Course_Context"),
    ]

    memory_block = (
        format_memory_for_prompt(memory) if memory_enabled and memory is not None else ""
    )
    if memory_block:
        memory_block = memory_block.replace("{", "{{").replace("}", "}}")

    document_list = "\n".join(
        [f'- "{doc_name}"' for doc_name in rag_module.retriever.documents_names]
    )
    template = (
        "Answer the following questions as best you can using lecture slide context.\n\n"
        "Retrieval instructions:\n"
        "- Search for context relevant to the question before answering.\n"
        "- If the retrieved context fully answers the question, go directly to Final Answer.\n"
        "- If the question involves multiple distinct concepts, make a separate search for each one.\n"
        "- If a search returns insufficient context, rephrase the query or use synonyms and try once more.\n"
        "- Base your answer primarily on the retrieved context and its vocabulary.\n\n"
        "Available lectures:\n"
        + document_list
        + "\n\nYou have access to the following tools:\n\n"
        "{tools}"
        + memory_block
        + "\n\nUse the following format:\n\n"
        "Question: the input question you must answer\n"
        "Thought: you should always think about what to do\n"
        "Action: the action to take, should be one of [{tool_names}]\n"
        "Action Input: the input to the action (a plain search query string)\n"
        "Observation: the result of the action\n"
        "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
        "Thought: I now know the final answer\n"
        "Final Answer: the final answer to the original input question\n\n"
        "Begin!\n\n"
        "Question: {input}\n"
        "Thought:{agent_scratchpad}"
    )

    prompt = PromptTemplate.from_template(template)
    
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=config.get("agent_config", {}).get("max_iterations", 4)
    )
    
    return agent_executor, slide_tool_manager


def _extract_thought_from_log(log: str) -> str:
    if "Action:" in log:
        return log.split("Action:")[0].strip()
    return log.strip()


def _normalize_action_input(tool_input: Any) -> Any:
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


def _metadata_from_action(tool: str, tool_input: Any) -> tuple[Optional[str], Optional[int]]:
    return None, None


class EvalAgentTraceCallback(BaseCallbackHandler):
    """Collects ReAct steps for evaluation (no observation content)."""

    def __init__(self):
        self.steps: list[dict[str, Any]] = []

    @property
    def chain_length(self) -> int:
        return len(self.steps)

    def on_agent_action(self, action, **kwargs):
        thought = _extract_thought_from_log(action.log)
        tool_input = _normalize_action_input(action.tool_input)
        document, slide_number = _metadata_from_action(action.tool, action.tool_input)
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


class StreamlitAgentCallbackHandler(BaseCallbackHandler):
    def __init__(self, st_status):
        self.status = st_status

    def on_agent_action(self, action, **kwargs):
        # 'action.log' contains the raw text like "Thought: I need to search..."
        # 'action.tool' is the name of the tool, e.g., "Search_Course_Slides"
        
        self.status.write(f"🤔 **Thinking:** {action.log.split('Action:')[0].strip()}")

        if action.tool == "Search_Course_Slides":
            self.status.update(label="Retrieving context from slides...", state="running")
        else:
            self.status.update(label=f"Using tool: {action.tool}...", state="running")

    def on_tool_end(self, output, **kwargs):
        truncated_output = output[:200].replace('\n', ' ') + "..." if len(output) > 200 else output
        self.status.write(f"📄 **Retrieved Context:** {truncated_output}")
        self.status.update(label="Reading retrieved context...", state="running")

    def on_agent_finish(self, finish, **kwargs):
        self.status.update(label="Response generated", state="complete", expanded=False)
