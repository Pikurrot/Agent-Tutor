from __future__ import annotations

import base64
import io
from typing import Any

import streamlit as st
from PIL import Image

COLS_PER_SLIDE_ROW = 4

PUBLIC_MODEL_PATH = "Qwen/Qwen3-VL-8B-Instruct"
PUBLIC_API_MODE = "tutor"


def encode_slides_for_storage(slides: list[dict] | None) -> list[dict[str, str]]:
    """Convert UI slide dicts ``{"image": PIL, "caption": str}`` to JSON-safe form."""
    if not slides:
        return []
    out: list[dict[str, str]] = []
    for slide in slides:
        buf = io.BytesIO()
        img = slide["image"]
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        out.append(
            {
                "caption": str(slide.get("caption", "")),
                "image_b64": b64,
                "mime_type": "image/png",
            }
        )
    return out


def decode_slides_from_storage(stored: list[dict] | None) -> list[dict[str, Any]]:
    """Restore UI slide dicts from JSON-safe storage form."""
    if not stored:
        return []
    slides_out: list[dict[str, Any]] = []
    for item in stored:
        raw = base64.b64decode(item["image_b64"])
        img = Image.open(io.BytesIO(raw))
        slides_out.append({"image": img, "caption": item.get("caption", "")})
    return slides_out


def render_slide_gallery(slides: list | None) -> None:
    if not slides:
        return
    st.caption("Sources · retrieved slides")
    for row_start in range(0, len(slides), COLS_PER_SLIDE_ROW):
        chunk = slides[row_start : row_start + COLS_PER_SLIDE_ROW]
        cols = st.columns(len(chunk))
        for col, slide in zip(cols, chunk, strict=True):
            with col:
                st.image(slide["image"], caption=slide["caption"], width="stretch")


def render_status_banner(message: str) -> None:
    """Single-line status rectangle shown while the tutor is working."""
    st.markdown(
        f"""
        <div style="
            padding: 12px 16px;
            background: #f0f2f6;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            color: #31333f;
            font-size: 0.95rem;
            margin-bottom: 0.75rem;
        ">{message}</div>
        """,
        unsafe_allow_html=True,
    )
