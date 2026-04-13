from __future__ import annotations

import base64
import io
from functools import lru_cache

from collections.abc import Iterator

from tutor.core.chat import generate_response
from tutor.core.streaming import get_agent_mock_stream_config, mock_stream_text
from tutor.modules.agent.agent import build_rag_agent
from tutor.modules.retrieval.RAG import RAGModule
from tutor.utils.config import load_config
from tutor.utils.misc import get_model
from tutor.utils.paths import DEFAULT_CONFIG_PATH, MODELS_CACHE_DIR

_rag_singleton: RAGModule | None = None


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


def run_completion(model_path: str, mode: str, prompt: str) -> tuple[str, list[dict]]:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    model, _model_type = _cached_model(model_path)
    if mode == "basic":
        return generate_response(model, prompt), []
    if mode == "rag":
        rag = get_rag_module()
        augmented, slides_ui = rag.retrieve_and_augment(prompt, on_progress=None)
        text = generate_response(model, augmented)
        return text, encode_slides_for_json(slides_ui)
    if mode == "agent":
        rag = get_rag_module()
        agent_executor, slide_manager = build_rag_agent(model, rag, cfg)
        response_dict = agent_executor.invoke({"input": prompt})
        text = response_dict["output"]
        return text, encode_slides_for_json(slide_manager.retrieved_slides)
    raise ValueError(f"Unknown mode: {mode}")


def iter_completion(model_path: str, mode: str, prompt: str) -> Iterator[tuple[str, object]]:
    """Yield ``("token", str)`` chunks, then one ``("slides", encoded_slide_dicts)``."""
    cfg = load_config(DEFAULT_CONFIG_PATH)
    delay, chunk = get_agent_mock_stream_config(cfg)
    model, _model_type = _cached_model(model_path)
    if mode == "basic":
        for piece in model.stream_generate(prompt):
            yield ("token", piece)
        yield ("slides", [])
        return
    if mode == "rag":
        rag = get_rag_module()
        augmented, slides_ui = rag.retrieve_and_augment(prompt, on_progress=None)
        encoded = encode_slides_for_json(slides_ui)
        for piece in model.stream_generate(augmented):
            yield ("token", piece)
        yield ("slides", encoded)
        return
    if mode == "agent":
        rag = get_rag_module()
        agent_executor, slide_manager = build_rag_agent(model, rag, cfg)
        response_dict = agent_executor.invoke({"input": prompt})
        text = response_dict["output"]
        encoded = encode_slides_for_json(slide_manager.retrieved_slides)
        for piece in mock_stream_text(text, delay, chunk):
            yield ("token", piece)
        yield ("slides", encoded)
        return
    raise ValueError(f"Unknown mode: {mode}")
