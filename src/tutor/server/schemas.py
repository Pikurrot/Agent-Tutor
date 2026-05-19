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


class CompleteRequest(BaseModel):
    model_path: str
    mode: Literal["basic", "rag", "agent"]
    prompt: str
    memory: Optional[ConversationMemoryIO] = None


class CompleteResponse(BaseModel):
    text: str
    slides: list[SlideOut] = Field(default_factory=list)
    memory: Optional[ConversationMemoryIO] = None
