from __future__ import annotations

import base64
import io
from functools import lru_cache
from typing import Optional

from collections.abc import Iterator

from tutor.core.chat import generate_response
from tutor.core.streaming import get_agent_mock_stream_config, mock_stream_text
from tutor.modules.agent.agent import build_rag_agent
from tutor.modules.agent.summarizer import ConversationMemory, roll_memory
from tutor.modules.retrieval.RAG import RAGModule
from tutor.server.schemas import ConversationMemoryIO
from tutor.utils.config import load_config
from tutor.utils.misc import get_model
from tutor.utils.paths import DEFAULT_CONFIG_PATH, MODELS_CACHE_DIR

_rag_singleton: RAGModule | None = None


def _memory_in_to_dataclass(memory_in: Optional[ConversationMemoryIO]) -> ConversationMemory:
    if memory_in is None:
        return ConversationMemory.empty()
    last = memory_in.last_interaction
    return ConversationMemory(
        summary=memory_in.summary or "",
        last_interaction=(
            {"user": last.user, "assistant": last.assistant} if last is not None else None
        ),
    )


def _memory_to_io(memory: ConversationMemory) -> ConversationMemoryIO:
    return ConversationMemoryIO.model_validate(memory.to_dict())


def encode_slides_for_json(slides: list[dict]) -> list[dict]:
    out: list[dict] = []
    for s in slides:
        buf = io.BytesIO()
        img = s["image"]
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        out.append(
            {
                "caption": s["caption"],
                "image_b64": b64,
                "mime_type": "image/png",
            }
        )
    return out


@lru_cache(maxsize=8)
def _cached_model(model_path: str):
    cfg = load_config(DEFAULT_CONFIG_PATH)
    model, model_type = get_model(model_path, str(MODELS_CACHE_DIR), cfg)
    model.eval()
    return model, model_type


def get_rag_module() -> RAGModule:
    global _rag_singleton
    if _rag_singleton is None:
        _rag_singleton = RAGModule(load_config(DEFAULT_CONFIG_PATH))
    return _rag_singleton


def run_completion(
    model_path: str,
    mode: str,
    prompt: str,
    memory: Optional[ConversationMemoryIO] = None,
) -> tuple[str, list[dict], Optional[ConversationMemoryIO]]:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    model, _model_type = _cached_model(model_path)
    if mode == "basic":
        return generate_response(model, prompt), [], None
    if mode == "rag":
        rag = get_rag_module()
        augmented, slides_ui, rag_images = rag.retrieve_and_augment(prompt, on_progress=None)
        text = generate_response(model, augmented, images=rag_images)
        return text, encode_slides_for_json(slides_ui), None
    if mode == "agent":
        rag = get_rag_module()
        mem = _memory_in_to_dataclass(memory)
        agent_executor, slide_manager = build_rag_agent(model, rag, cfg, memory=mem)
        response_dict = agent_executor.invoke({"input": prompt})
        text = response_dict["output"]
        new_mem = _update_memory_after_agent(cfg, mem, prompt, text, model)
        return (
            text,
            encode_slides_for_json(slide_manager.retrieved_slides),
            _memory_to_io(new_mem),
        )
    raise ValueError(f"Unknown mode: {mode}")


def iter_completion(
    model_path: str,
    mode: str,
    prompt: str,
    memory: Optional[ConversationMemoryIO] = None,
) -> Iterator[tuple[str, object]]:
    """Yield ``("token", str)`` chunks, then one ``("end", {"slides": ..., "memory": ...})``."""
    cfg = load_config(DEFAULT_CONFIG_PATH)
    delay, chunk = get_agent_mock_stream_config(cfg)
    model, _model_type = _cached_model(model_path)
    if mode == "basic":
        for piece in model.stream_generate(prompt):
            yield ("token", piece)
        yield ("end", {"slides": [], "memory": None})
        return
    if mode == "rag":
        rag = get_rag_module()
        augmented, slides_ui, rag_images = rag.retrieve_and_augment(prompt, on_progress=None)
        encoded = encode_slides_for_json(slides_ui)
        for piece in model.stream_generate(augmented, images=rag_images):
            yield ("token", piece)
        yield ("end", {"slides": encoded, "memory": None})
        return
    if mode == "agent":
        rag = get_rag_module()
        mem = _memory_in_to_dataclass(memory)
        agent_executor, slide_manager = build_rag_agent(model, rag, cfg, memory=mem)
        response_dict = agent_executor.invoke({"input": prompt})
        text = response_dict["output"]
        encoded = encode_slides_for_json(slide_manager.retrieved_slides)
        for piece in mock_stream_text(text, delay, chunk):
            yield ("token", piece)
        new_mem = _update_memory_after_agent(cfg, mem, prompt, text, model)
        yield ("end", {"slides": encoded, "memory": _memory_to_io(new_mem).model_dump()})
        return
    raise ValueError(f"Unknown mode: {mode}")


def _update_memory_after_agent(
    cfg: dict,
    mem: ConversationMemory,
    user_prompt: str,
    assistant_text: str,
    model,
) -> ConversationMemory:
    agent_cfg = cfg.get("agent_config", {}) or {}
    memory_cfg = agent_cfg.get("memory", {}) or {}
    if not bool(memory_cfg.get("enabled", True)):
        return mem
    summary_max_new_tokens = int(memory_cfg.get("summary_max_new_tokens", 512))
    return roll_memory(
        mem,
        new_user=user_prompt,
        new_assistant=assistant_text,
        model=model,
        summary_max_new_tokens=summary_max_new_tokens,
    )
