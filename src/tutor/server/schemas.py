from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SlideOut(BaseModel):
    caption: str
    image_b64: str
    mime_type: str = "image/png"


class CompleteRequest(BaseModel):
    model_path: str
    mode: Literal["basic", "rag", "agent"]
    prompt: str


class CompleteResponse(BaseModel):
    text: str
    slides: list[SlideOut] = Field(default_factory=list)
