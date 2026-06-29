from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from openai import OpenAI

from tutor.core.eval_backends import EvalMode, build_eval_run_context, generate_prediction
from tutor.modules.agent.agent import EvalAgentTraceCallback, build_rag_agent
from tutor.modules.models.openai_model import openai_generate_answer
from tutor.server.inference import _cached_model, get_rag_module
from tutor.utils.config import load_config
from tutor.utils.misc import parse_json_response
from tutor.utils.paths import DEFAULT_CONFIG_PATH, PROJECT_ROOT

MatchMetric = Literal["Substantial Match", "Partial Match", "Mismatch"]

MATCH_METRICS: frozenset[str] = frozenset(
    {"Substantial Match", "Partial Match", "Mismatch"}
)

DEFAULT_JUDGE_SYSTEM_INSTRUCTION = """\
You are an impartial grader for a university course tutoring agent.

Your task: compare the GENERATED answer (from the agent) against the REFERENCE answer
(ground truth) for the same QUESTION, and assign exactly one rubric category.

Rubric (use these exact labels):
- Substantial Match: The predicted answer captures the core meaning of the ground truth.
  Minor phrasing differences are acceptable.
- Partial Match: The predicted answer includes some correct elements but misses key context
  or includes minor inaccuracies.
- Mismatch: The predicted answer is entirely wrong, irrelevant, or contains severe
  hallucinations.

Guidance:
- Write your reasoning first (2-5 sentences), comparing generated vs reference before
  choosing the category.
- Empty, missing, or non-answers (e.g. "Agent stopped due to iteration limit") are Mismatch.
- Extra helpful detail is fine if the core meaning still matches.

Respond with strict JSON only (no markdown fences), using exactly these keys in this order:
- "reasoning": string (your comparison and justification for the category)
- "metric": string, exactly one of: "Substantial Match", "Partial Match", "Mismatch"
"""


@dataclass
class EvalItem:
    id: str
    question: str
    answer: str


@dataclass
class EvalResult:
    id: str
    question: str
    ground_truth: str
    generated: str
    metric: Optional[MatchMetric]
    reasoning: Optional[str]
    error: Optional[str]
    duration_s: float
    slides_count: int
    chain_length: int
    agent_steps: list[dict[str, Any]] = field(default_factory=list)
    eval_mode: str = "agent"
    extra: dict[str, Any] = field(default_factory=dict)


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_dataset(dataset_path: Path) -> list[EvalItem]:
    path = _resolve_path(dataset_path)
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "items" in raw:
        entries = raw["items"]
    elif isinstance(raw, list):
        entries = raw
    else:
        raise ValueError(
            f"Dataset must be a JSON list or object with 'items' key, got {type(raw).__name__}"
        )

    items: list[EvalItem] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Dataset entry {i} must be an object, got {type(entry).__name__}")
        question = entry.get("question")
        answer = entry.get("answer")
        if not question or not answer:
            raise ValueError(f"Dataset entry {i} must have 'question' and 'answer' fields")
        item_id = entry.get("id", str(i))
        items.append(EvalItem(id=str(item_id), question=str(question), answer=str(answer)))
    return items


def _parse_results_file(text: str) -> list[dict[str, Any]]:
    """Parse JSONL; fall back to a single pretty-printed JSON object."""
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    if rows:
        return rows
    stripped = text.strip()
    if stripped:
        single = json.loads(stripped)
        if isinstance(single, dict):
            return [single]
    return []


def load_all_results(results_path: Path) -> list[dict[str, Any]]:
    if not results_path.exists():
        return []
    text = results_path.read_text(encoding="utf-8")
    return _parse_results_file(text)


def load_results_by_id(results_path: Path) -> dict[str, dict[str, Any]]:
    """Last row wins per id (handles accidental duplicates)."""
    by_id: dict[str, dict[str, Any]] = {}
    for row in load_all_results(results_path):
        item_id = row.get("id")
        if item_id is not None:
            by_id[str(item_id)] = row
    return by_id


def is_judged(row: dict[str, Any]) -> bool:
    """True when the row has a valid rubric metric (new format)."""
    return row.get("metric") in MATCH_METRICS


def needs_judge_only(row: dict[str, Any]) -> bool:
    """Re-judge when agent output exists but metric is missing or legacy."""
    generated = row.get("generated")
    if not generated:
        return False
    if is_judged(row) and "acceptable" not in row:
        return False
    return True


def _apply_judge_fields(
    row: dict[str, Any],
    metric: Optional[str],
    reasoning: Optional[str],
    error: Optional[str],
) -> dict[str, Any]:
    """Write rubric fields and drop legacy boolean acceptable."""
    row.pop("acceptable", None)
    row["metric"] = metric
    row["reasoning"] = reasoning
    row["error"] = error
    return row


def save_all_results(results_path: Path, rows: list[dict[str, Any]]) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def ordered_results_rows(
    all_items: list[EvalItem], by_id: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    return [by_id[item.id] for item in all_items if item.id in by_id]


def persist_results(
    results_path: Path,
    all_items: list[EvalItem],
    by_id: dict[str, dict[str, Any]],
) -> None:
    save_all_results(results_path, ordered_results_rows(all_items, by_id))


def resolve_mode_paths(eval_cfg: dict, mode: EvalMode) -> tuple[dict, Path, Path]:
    """Return (mode_cfg, results_path, summary_path) for the given eval mode."""
    modes_cfg = eval_cfg.get("modes") or {}
    if mode not in modes_cfg:
        raise ValueError(
            f"eval config missing modes.{mode}; "
            f"available: {', '.join(sorted(modes_cfg)) or '(none)'}"
        )
    mode_cfg = modes_cfg[mode]
    output_dir = _resolve_path(Path(eval_cfg.get("output_dir", "eval_runs")))
    results_file = mode_cfg.get("results_file", f"{mode}_results.jsonl")
    summary_file = mode_cfg.get("summary_file", f"summary_{mode}.json")
    return mode_cfg, output_dir / results_file, output_dir / summary_file


def _result_to_row(result: EvalResult) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": result.id,
        "question": result.question,
        "ground_truth": result.ground_truth,
        "generated": result.generated,
        "metric": result.metric,
        "reasoning": result.reasoning,
        "error": result.error,
        "duration_s": round(result.duration_s, 3),
        "slides_count": result.slides_count,
        "chain_length": result.chain_length,
        "agent_steps": result.agent_steps,
        "eval_mode": result.eval_mode,
    }
    row.update(result.extra)
    return row


def compute_summary_stats(rows: list[dict[str, Any]], total_dataset_size: int) -> dict[str, Any]:
    completed_count = len(rows)
    substantial_count = sum(1 for r in rows if r.get("metric") == "Substantial Match")
    partial_count = sum(1 for r in rows if r.get("metric") == "Partial Match")
    mismatch_count = sum(1 for r in rows if r.get("metric") == "Mismatch")
    failed_judge_count = sum(1 for r in rows if not is_judged(r))
    substantial_match_rate = (
        substantial_count / completed_count if completed_count else 0.0
    )
    return {
        "substantial_match_rate": substantial_match_rate,
        "completed_count": completed_count,
        "total_dataset_size": total_dataset_size,
        "substantial_match_count": substantial_count,
        "partial_match_count": partial_count,
        "mismatch_count": mismatch_count,
        "failed_judge_count": failed_judge_count,
    }


def write_summary(summary_path: Path, summary: dict[str, Any]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


_ITERATION_LIMIT_SENTINEL = "Agent stopped due to iteration limit"
_FALLBACK_MAX_NEW_TOKENS = 2048


def run_agent_with_trace(
    model_path: str,
    prompt: str,
) -> tuple[str, list[dict[str, Any]], int, int]:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    model, _model_type = _cached_model(model_path)
    rag = get_rag_module()
    agent_executor, slide_manager = build_rag_agent(model, rag, cfg, memory=None)
    trace_callback = EvalAgentTraceCallback()
    response_dict = agent_executor.invoke(
        {"input": prompt},
        config={"callbacks": [trace_callback]},
    )
    generated = response_dict["output"]

    if _ITERATION_LIMIT_SENTINEL in generated:
        print(f"  [fallback] iteration limit hit — generating from accumulated context")
        if slide_manager.retrieved_transcripts:
            context = "\n---\n".join(slide_manager.retrieved_transcripts)
        else:
            retrieved_data, _ = rag.retrieve(prompt)
            context = slide_manager.prepare_retrieved_data(retrieved_data)
        fallback_prompt = (
            "You are answering a university deep learning course question.\n"
            "Use only the lecture context below.\n\n"
            f"## Context\n{context}\n\n"
            f"## Question\n{prompt}"
        )
        generated = str(model.generate(fallback_prompt, max_new_tokens=_FALLBACK_MAX_NEW_TOKENS)).strip()

    agent_steps = trace_callback.steps
    chain_length = trace_callback.chain_length
    slides_count = len(slide_manager.retrieved_slides)
    return generated, agent_steps, chain_length, slides_count


class JudgeClient:
    """Wraps a single OpenAI client for evaluation judging."""

    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is required for evaluation judging"
            )
        self.client = OpenAI(api_key=api_key)
        print("Judge: OpenAI client initialized")


def judge_generate(
    judge_client: JudgeClient,
    prompt: str,
    *,
    model: str,
    temperature: float,
    thinking_budget: int = 0,
    system_instruction: Optional[str] = None,
) -> str:
    """Call OpenAI through a JudgeClient for rubric judging and conversation evaluation."""
    return openai_generate_answer(
        judge_client.client,
        prompt=prompt,
        model=model,
        temperature=temperature,
        system_instruction=system_instruction,
    )


def _build_judge_user_message(question: str, ground_truth: str, generated: str) -> str:
    return (
        f"Question: {question}\n\n"
        f"Reference answer (ground truth):\n{ground_truth}\n\n"
        f"Generated answer:\n{generated}"
    )


def _normalize_metric(value: Any) -> Optional[MatchMetric]:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if stripped in MATCH_METRICS:
        return stripped  # type: ignore[return-value]
    # Tolerate minor casing / spacing drift from the judge
    lowered = stripped.lower()
    for label in MATCH_METRICS:
        if label.lower() == lowered:
            return label  # type: ignore[return-value]
    return None


def _parse_judge_response(text: str) -> tuple[Optional[MatchMetric], Optional[str], Optional[str]]:
    try:
        parsed = parse_json_response(text)
    except (json.JSONDecodeError, ValueError) as e:
        return None, None, f"Failed to parse judge JSON: {e}"

    reasoning = parsed.get("reasoning")
    metric = _normalize_metric(parsed.get("metric"))
    if not isinstance(reasoning, str):
        return None, None, f"Judge response missing string 'reasoning': {parsed!r}"
    if metric is None:
        return None, None, (
            f"Judge response missing valid 'metric' "
            f"(expected one of {sorted(MATCH_METRICS)}): {parsed!r}"
        )
    return metric, reasoning, None


def judge_answer(
    judge_client: JudgeClient,
    question: str,
    ground_truth: str,
    generated: str,
    *,
    model: str,
    temperature: float,
    thinking_budget: int,
    system_instruction: str,
) -> tuple[Optional[MatchMetric], Optional[str], Optional[str]]:
    def _call(prompt: str) -> str:
        return judge_generate(
            judge_client,
            prompt,
            model=model,
            temperature=temperature,
            thinking_budget=thinking_budget,
            system_instruction=system_instruction,
        )

    user_message = _build_judge_user_message(question, ground_truth, generated)
    try:
        raw = _call(user_message)
    except Exception as e:
        return None, None, f"Judge API error: {e}"

    metric, reasoning, error = _parse_judge_response(raw)
    if error is None:
        return metric, reasoning, None

    retry_prompt = (
        user_message
        + "\n\nYour previous response was not valid JSON. "
        "Respond with strict JSON only, no markdown, keys: "
        'reasoning (string), metric (one of "Substantial Match", "Partial Match", "Mismatch").'
    )
    try:
        raw_retry = _call(retry_prompt)
    except Exception as e:
        return None, None, f"Judge API error on retry: {e}"
    return _parse_judge_response(raw_retry)


def run_evaluation(
    eval_cfg: dict,
    *,
    mode: EvalMode = "agent",
    limit: Optional[int] = None,
    fresh: bool = False,
) -> Path:
    dataset_path = eval_cfg.get("dataset_path")
    if not dataset_path:
        raise ValueError("eval config must specify dataset_path")

    _mode_cfg, results_path, summary_path = resolve_mode_paths(eval_cfg, mode)
    output_dir = results_path.parent
    resume = bool(eval_cfg.get("resume", True)) and not fresh
    model_path = eval_cfg.get("model_path", "Qwen/Qwen3-8B")
    cfg_limit = eval_cfg.get("limit")
    effective_limit = limit if limit is not None else cfg_limit

    judge_cfg = eval_cfg.get("judge", {}) or {}
    judge_model = judge_cfg.get("model", "gpt-5.4-mini")
    judge_temperature = float(judge_cfg.get("temperature", 0.0))
    judge_thinking_budget = int(judge_cfg.get("thinking_budget", 0))
    judge_system = judge_cfg.get("system_instruction") or DEFAULT_JUDGE_SYSTEM_INSTRUCTION

    all_items = load_dataset(Path(dataset_path))
    total_dataset_size = len(all_items)
    if effective_limit is not None:
        all_items = all_items[: int(effective_limit)]

    output_dir.mkdir(parents=True, exist_ok=True)

    if fresh:
        if results_path.exists():
            results_path.unlink()
        if summary_path.exists():
            summary_path.unlink()

    by_id: dict[str, dict[str, Any]] = load_results_by_id(results_path) if resume else {}

    judged_count = sum(
        1 for item in all_items if item.id in by_id and is_judged(by_id[item.id])
    )
    judge_retry_count = sum(
        1 for item in all_items if item.id in by_id and needs_judge_only(by_id[item.id])
    )
    agent_pending_count = sum(1 for item in all_items if item.id not in by_id)

    work_items: list[tuple[EvalItem, str]] = []
    for item in all_items:
        existing = by_id.get(item.id)
        if existing is not None and is_judged(existing) and "acceptable" not in existing:
            continue
        if existing is not None and needs_judge_only(existing):
            work_items.append((item, "judge_only"))
        else:
            work_items.append((item, "full"))

    if not work_items:
        rows = ordered_results_rows(all_items, by_id)
        stats = compute_summary_stats(rows, total_dataset_size)
        print(f"All {len(all_items)} items already evaluated (resume).")
        if rows:
            print(
                f"  Substantial Match: {stats['substantial_match_count']}/{stats['completed_count']} "
                f"({stats['substantial_match_rate']:.1%})"
            )
        print(f"  Results: {results_path}")
        return output_dir

    if judged_count or judge_retry_count or agent_pending_count:
        predict_work_count = sum(1 for _, m in work_items if m == "full")
        print(
            f"Resuming ({mode}): {judged_count} judged, {judge_retry_count} to re-judge only, "
            f"{predict_work_count} need prediction"
        )

    print(f"Eval mode: {mode}")
    run_ctx = build_eval_run_context(mode, eval_cfg)
    judge_client = JudgeClient()
    started_at = datetime.now(timezone.utc).isoformat()
    processed_in_run = 0

    for item, mode in work_items:
        processed_in_run += 1
        global_idx = judged_count + processed_in_run
        total_to_run = len(all_items)
        print(f"Eval {global_idx}/{total_to_run}: {item.id}")

        if mode == "judge_only":
            existing = by_id[item.id]
            generated = str(existing["generated"])
            print("  -> re-judging (prediction preserved)")
            t0 = time.perf_counter()
            metric, reasoning, error = judge_answer(
                judge_client,
                item.question,
                item.answer,
                generated,
                model=judge_model,
                temperature=judge_temperature,
                thinking_budget=judge_thinking_budget,
                system_instruction=judge_system,
            )
            judge_duration = time.perf_counter() - t0
            by_id[item.id] = _apply_judge_fields(existing, metric, reasoning, error)
            chain_length = existing.get("chain_length", 0)
            agent_duration = existing.get("duration_s", 0)
        else:
            try:
                pred = generate_prediction(run_ctx, item)
            except Exception as e:
                print(f"  -> prediction error (not saved, will retry on resume): {e}")
                processed_in_run -= 1
                continue

            agent_duration = pred.duration_s
            generated = pred.generated
            agent_steps = pred.agent_steps
            chain_length = pred.chain_length
            slides_count = pred.slides_count

            metric, reasoning, error = judge_answer(
                judge_client,
                item.question,
                item.answer,
                generated,
                model=judge_model,
                temperature=judge_temperature,
                thinking_budget=judge_thinking_budget,
                system_instruction=judge_system,
            )

            by_id[item.id] = _result_to_row(
                EvalResult(
                    id=item.id,
                    question=item.question,
                    ground_truth=item.answer,
                    generated=generated,
                    metric=metric,
                    reasoning=reasoning,
                    error=error,
                    duration_s=agent_duration,
                    slides_count=slides_count,
                    chain_length=chain_length,
                    agent_steps=agent_steps,
                    eval_mode=mode,
                    extra=pred.extra,
                )
            )
            judge_duration = 0.0

        persist_results(results_path, all_items, by_id)

        rows = ordered_results_rows(all_items, by_id)
        stats = compute_summary_stats(rows, total_dataset_size)
        summary = {
            "started_at": started_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "eval_mode": mode,
            "results_path": str(results_path),
            "dataset_path": str(dataset_path),
            "model_path": model_path,
            "judge": {
                "model": judge_model,
                "temperature": judge_temperature,
                "thinking_budget": judge_thinking_budget,
            },
            **stats,
        }
        write_summary(summary_path, summary)

        metric = by_id[item.id].get("metric")
        status = metric if metric else "judge failed"
        if mode == "judge_only":
            print(f"  -> {status} (judge {judge_duration:.1f}s)")
        else:
            print(f"  -> {status} ({agent_duration:.1f}s, chain_length={chain_length})")

    rows = ordered_results_rows(all_items, by_id)
    stats = compute_summary_stats(rows, total_dataset_size)
    print()
    print(
        f"Evaluation complete: {stats['substantial_match_count']}/{stats['completed_count']} "
        f"Substantial Match ({stats['substantial_match_rate']:.1%})"
    )
    print(
        f"  Partial Match: {stats['partial_match_count']}, "
        f"Mismatch: {stats['mismatch_count']}"
    )
    if stats["failed_judge_count"]:
        print(f"  Judge failures: {stats['failed_judge_count']}")
    print(f"  Results: {results_path}")
    print(f"  Summary: {summary_path}")

    return output_dir
