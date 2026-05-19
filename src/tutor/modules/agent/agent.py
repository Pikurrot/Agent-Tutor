from __future__ import annotations
from typing import Optional

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
        slide_tool_manager.get_tool("Search_Document_Context"),
        slide_tool_manager.get_tool("Retrieve_Slide_Context"),
    ]

    memory_block = (
        format_memory_for_prompt(memory) if memory_enabled and memory is not None else ""
    )
    if memory_block:
        memory_block = memory_block.replace("{", "{{").replace("}", "}}")

    template = """Answer the following questions as best you can.
Retrieval instructions:
- When the question involves more than one concept, idea or term, separate the retrieval into multiple steps, making one search query after the other. For instance, if the question is about "semantic segmentation and convolutional networks", make the search query (Action Input) be "semantic segmentation", retrieve the context (Observation) and then make another search query for "convolutional networks".
- When the retrieved context does not contain the information expected by the search query, rephrase the search query to be more specific, or use synonims. Be original.
- When retrieving context, answer the question mainly based on the information and vocabulary provided in the context.
""" + \
"You have access to the following documents:\n" + \
'\n'.join([f"- \"{doc_name}\"" for doc_name in rag_module.retriever.documents_names]) + \
"""
Always use one of the available tools.
You have access to the following tools:

{tools}""" + memory_block + """
Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

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
