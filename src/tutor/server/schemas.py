from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class SlideOut(BaseModel):
    caption: str
    image_b64: str
    mime_type: str = "image/png"


class LastInteraction(BaseModel):
    user: str
    assistant: str


class ConversationMemoryIO(BaseModel):
    summary: str = ""
    last_interaction: Optional[LastInteraction] = None


class TeachingAnchorIO(BaseModel):
    original_question: str = ""
    target_explanation: str = ""
    key_facts: list[str] = Field(default_factory=list)
    misconceptions: list[str] = Field(default_factory=list)
    scaffold_questions: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)


class TeachingSessionIO(BaseModel):
    anchor: Optional[TeachingAnchorIO] = None
    status: Literal["exploring", "narrowing", "close", "resolved"] = "exploring"
    hints_used: int = 0
    current_scaffold_index: int = 0


class AgentTraceStepIO(BaseModel):
    step: int
    thought: str
    action: str
    action_input: object
    document: Optional[str] = None
    slide_number: Optional[int] = None


class AgentTraceIO(BaseModel):
    agent_name: str
    steps: list[AgentTraceStepIO] = Field(default_factory=list)
    raw_output: str = ""
    chain_length: int = 0
    used_fallback: bool = False
    parse_retry_count: int = 0
    parse_failed: bool = False
    notes: list[str] = Field(default_factory=list)


class TutorDebugIO(BaseModel):
    teaching_anchor_snapshot: Optional[TeachingAnchorIO] = None
    answer_agent_trace: Optional[AgentTraceIO] = None
    pedagogic_agent_trace: Optional[AgentTraceIO] = None
    session_status: str = "exploring"
    hints_used: int = 0
    current_scaffold_index: int = 0


class CompleteRequest(BaseModel):
    model_path: str
    mode: Literal["basic", "rag", "agent", "tutor"]
    prompt: str
    memory: Optional[ConversationMemoryIO] = None
    teaching_session: Optional[TeachingSessionIO] = None
    debug: bool = False


class CompleteResponse(BaseModel):
    text: str
    slides: list[SlideOut] = Field(default_factory=list)
    memory: Optional[ConversationMemoryIO] = None
    teaching_session: Optional[TeachingSessionIO] = None
    debug_data: Optional[TutorDebugIO] = None


class WarmupRequest(BaseModel):
    model_path: str


class WarmupResponse(BaseModel):
    status: str = "ready"
