from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional

if TYPE_CHECKING:
    from tutor.core.evaluation import EvalItem
from tutor.core.chat import generate_response
from tutor.server.inference import _cached_model, get_rag_module
from tutor.utils.config import load_config
from tutor.utils.paths import DEFAULT_CONFIG_PATH

EvalMode = Literal["agent", "llm_context", "llm_baseline", "conversation", "rag"]

DEFAULT_LLM_BASELINE_TEMPLATE = """\
You are answering questions for a university deep learning course. You do not have access to lecture slides or transcripts. Answer as well as you can from general knowledge. If the question refers to lecture-specific wording or examples, make a reasonable guess and say when you are uncertain.

Question: {question}"""

DEFAULT_RAG_TEMPLATE = """\
You are answering questions for a university deep learning course.
Use only the lecture context below to answer the question.
If the answer is not supported by the context, say so briefly.

## Retrieved context
{context}

## Question
{question}"""

DEFAULT_LLM_CONTEXT_TEMPLATE = """\
You are answering questions for a university deep learning course. Use only the lecture transcripts below. If the answer is not supported by the transcripts, say so briefly.

## Lecture transcripts
{context}

## Question
{question}"""


@dataclass
class PredictionOutput:
    generated: str
    duration_s: float
    slides_count: int = 0
    chain_length: int = 0
    agent_steps: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


def format_lecture_chunk(document_name: str, slide_index: int, transcript: str) -> str:
    header = f'[document: "{document_name}" | slide_number: {slide_index + 1}]'
    return f"{header}\n{transcript}"


def format_all_chunks(chunks: list[dict]) -> str:
    parts = [
        format_lecture_chunk(c["document_name"], c["slide_index"], c["transcript"])
        for c in chunks
    ]
    return "\n---\n".join(parts)


def count_tokens(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def truncate_chunks_to_token_budget(
    chunks: list[dict],
    tokenizer: Any,
    max_tokens: int,
) -> tuple[list[dict], int, bool]:
    """Keep chunks from the start; drop slides from the end until within budget."""
    if max_tokens <= 0 or not chunks:
        return [], 0, bool(chunks)

    kept: list[dict] = []
    for i, chunk in enumerate(chunks):
        candidate = kept + [chunk]
        text = format_all_chunks(candidate)
        tokens = count_tokens(tokenizer, text)
        if tokens > max_tokens and kept:
            break
        if tokens > max_tokens and not kept:
            # Single huge slide: include it anyway (caller may truncate string later)
            kept = [chunk]
            break
        kept = candidate

    final_text = format_all_chunks(kept)
    final_tokens = count_tokens(tokenizer, final_text)
    truncated = len(kept) < len(chunks)
    return kept, final_tokens, truncated


def build_llm_context_corpus(
    model: Any,
    mode_cfg: dict,
) -> tuple[str, dict[str, Any]]:
    """Load all lecture transcripts, truncate to token budget, return context block + metadata."""
    rag = get_rag_module()
    chunks = rag.get_all_lecture_chunks()
    total_slides = len(chunks)

    max_input_tokens = int(mode_cfg.get("max_input_tokens", 28000))
    reserve = int(mode_cfg.get("prompt_reserve_tokens", 1500))
    transcript_budget = max(512, max_input_tokens - reserve)

    tokenizer = model.tokenizer
    kept_chunks, context_tokens, truncated = truncate_chunks_to_token_budget(
        chunks, tokenizer, transcript_budget
    )
    context_text = format_all_chunks(kept_chunks)

    documents_included = sorted({c["document_name"] for c in kept_chunks})
    meta = {
        "total_slides": total_slides,
        "slides_included": len(kept_chunks),
        "context_tokens": context_tokens,
        "context_truncated": truncated,
        "documents_included": documents_included,
    }
    print(
        f"LLM context: {len(kept_chunks)}/{total_slides} slides, "
        f"{context_tokens} tokens"
        f"{' (truncated)' if truncated else ''}"
    )
    return context_text, meta


def _generate_with_model(model: Any, prompt: str, max_new_tokens: int) -> str:
    return str(model.generate(prompt, max_new_tokens=max_new_tokens)).strip()


def predict_rag(model: Any, question: str, mode_cfg: dict) -> PredictionOutput:
    rag = get_rag_module()
    max_new_tokens = int(mode_cfg.get("max_new_tokens", 1024))
    template = mode_cfg.get("prompt_template") or None
    t0 = time.perf_counter()
    augmented_prompt, slides_ui, _images = rag.retrieve_and_augment(query=question)
    if template is not None:
        context_block = "\n---\n".join(
            format_lecture_chunk(d["document_name"], d["slide_index"], d["transcript"])
            for d in rag.retriever.retrieve(question)[0]
        )
        prompt = template.format(context=context_block, question=question)
    else:
        prompt = augmented_prompt
    generated = _generate_with_model(model, prompt, max_new_tokens)
    retrieved_docs = sorted({s["caption"].split(" · ")[0] for s in slides_ui})
    return PredictionOutput(
        generated=generated,
        duration_s=time.perf_counter() - t0,
        slides_count=len(slides_ui),
        chain_length=1,
        extra={"retrieved_documents": retrieved_docs},
    )


def predict_agent(model_path: str, question: str) -> PredictionOutput:
    from tutor.core.evaluation import run_agent_with_trace

    t0 = time.perf_counter()
    generated, agent_steps, chain_length, slides_count = run_agent_with_trace(
        model_path, question
    )
    return PredictionOutput(
        generated=generated,
        duration_s=time.perf_counter() - t0,
        slides_count=slides_count,
        chain_length=chain_length,
        agent_steps=agent_steps,
    )


def predict_llm_baseline(
    model: Any,
    question: str,
    mode_cfg: dict,
) -> PredictionOutput:
    template = mode_cfg.get("prompt_template") or DEFAULT_LLM_BASELINE_TEMPLATE
    max_new_tokens = int(mode_cfg.get("max_new_tokens", 1024))
    prompt = template.format(question=question)
    t0 = time.perf_counter()
    generated = _generate_with_model(model, prompt, max_new_tokens)
    return PredictionOutput(
        generated=generated,
        duration_s=time.perf_counter() - t0,
    )


def predict_llm_context(
    model: Any,
    question: str,
    mode_cfg: dict,
    context_text: str,
    context_meta: dict[str, Any],
) -> PredictionOutput:
    template = mode_cfg.get("prompt_template") or DEFAULT_LLM_CONTEXT_TEMPLATE
    max_new_tokens = int(mode_cfg.get("max_new_tokens", 1024))
    prompt = template.format(context=context_text, question=question)
    t0 = time.perf_counter()
    generated = _generate_with_model(model, prompt, max_new_tokens)
    return PredictionOutput(
        generated=generated,
        duration_s=time.perf_counter() - t0,
        extra=dict(context_meta),
    )


@dataclass
class EvalRunContext:
    mode: EvalMode
    model_path: str
    mode_cfg: dict
    model: Any = None
    context_text: str = ""
    context_meta: dict[str, Any] = field(default_factory=dict)


def build_eval_run_context(
    mode: EvalMode,
    eval_cfg: dict,
) -> EvalRunContext:
    modes_cfg = eval_cfg.get("modes") or {}
    if mode not in modes_cfg:
        raise ValueError(f"eval config missing modes.{mode}")
    mode_cfg = modes_cfg[mode]
    model_path = eval_cfg.get("model_path", "Qwen/Qwen3-8B")

    ctx = EvalRunContext(mode=mode, model_path=model_path, mode_cfg=mode_cfg)

    if mode == "agent":
        return ctx

    load_config(DEFAULT_CONFIG_PATH)
    ctx.model, _ = _cached_model(model_path)

    if mode == "llm_context":
        ctx.context_text, ctx.context_meta = build_llm_context_corpus(ctx.model, mode_cfg)

    return ctx  # rag and llm_baseline need no further setup


def generate_prediction(
    run_ctx: EvalRunContext,
    item: "EvalItem",
) -> PredictionOutput:
    if run_ctx.mode == "agent":
        return predict_agent(run_ctx.model_path, item.question)
    if run_ctx.mode == "llm_baseline":
        if run_ctx.model is None:
            raise RuntimeError("LLM model not loaded for llm_baseline")
        return predict_llm_baseline(run_ctx.model, item.question, run_ctx.mode_cfg)
    if run_ctx.mode == "llm_context":
        if run_ctx.model is None:
            raise RuntimeError("LLM model not loaded for llm_context")
        return predict_llm_context(
            run_ctx.model,
            item.question,
            run_ctx.mode_cfg,
            run_ctx.context_text,
            run_ctx.context_meta,
        )
    if run_ctx.mode == "rag":
        if run_ctx.model is None:
            raise RuntimeError("LLM model not loaded for rag")
        return predict_rag(run_ctx.model, item.question, run_ctx.mode_cfg)
    raise ValueError(f"Unknown eval mode: {run_ctx.mode!r}")
