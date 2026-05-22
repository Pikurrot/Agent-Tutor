from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from google import genai

from tutor.modules.agent.agent import EvalAgentTraceCallback, build_rag_agent
from tutor.modules.models.gemini import gemini_generate_answer
from tutor.server.inference import _cached_model, get_rag_module
from tutor.utils.config import load_config
from tutor.utils.misc import parse_json_response
from tutor.utils.paths import DEFAULT_CONFIG_PATH, PROJECT_ROOT

DEFAULT_JUDGE_SYSTEM_INSTRUCTION = """\
You are an impartial grader for a university course tutoring agent.

Your task: decide whether the GENERATED answer satisfactorily answers the QUESTION
in the same sense as the REFERENCE answer (ground truth).

Grading criteria:
- The generated answer should be semantically equivalent to the reference or cover the same key facts.
- Paraphrasing and additional helpful detail are acceptable.
- Mark as not acceptable if the generated answer contains factual errors, omits critical points
  from the reference, or answers a different question.

Respond with strict JSON only (no markdown fences), using exactly these keys:
- "acceptable": boolean (true if the generated answer is acceptable, false otherwise)
- "reasoning": string (2-5 sentences explaining your judgment, comparing generated vs reference)
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
    acceptable: Optional[bool]
    reasoning: Optional[str]
    error: Optional[str]
    duration_s: float
    slides_count: int
    chain_length: int
    agent_steps: list[dict[str, Any]] = field(default_factory=list)


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
    return isinstance(row.get("acceptable"), bool)


def needs_judge_only(row: dict[str, Any]) -> bool:
    generated = row.get("generated")
    return bool(generated) and row.get("acceptable") is None


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
º

def _result_to_row(result: EvalResult) -> dict[str, Any]:
    return {
        "id": result.id,
        "question": result.question,
        "ground_truth": result.ground_truth,
        "generated": result.generated,
        "acceptable": result.acceptable,
        "reasoning": result.reasoning,
        "error": result.error,
        "duration_s": round(result.duration_s, 3),
        "slides_count": result.slides_count,
        "chain_length": result.chain_length,
        "agent_steps": result.agent_steps,
    }


def compute_summary_stats(rows: list[dict[str, Any]], total_dataset_size: int) -> dict[str, Any]:
    completed_count = len(rows)
    acceptable_count = sum(1 for r in rows if r.get("acceptable") is True)
    failed_judge_count = sum(1 for r in rows if r.get("acceptable") is None)
    accuracy = acceptable_count / completed_count if completed_count else 0.0
    return {
        "accuracy": accuracy,
        "completed_count": completed_count,
        "total_dataset_size": total_dataset_size,
        "acceptable_count": acceptable_count,
        "failed_judge_count": failed_judge_count,
    }


def write_summary(summary_path: Path, summary: dict[str, Any]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


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
    agent_steps = trace_callback.steps
    chain_length = trace_callback.chain_length
    slides_count = len(slide_manager.retrieved_slides)
    return generated, agent_steps, chain_length, slides_count


def _build_judge_client() -> genai.Client:
    keys_raw = os.getenv("GEMINI_API_KEYS")
    if not keys_raw:
        raise EnvironmentError(
            "GEMINI_API_KEYS environment variable is required for evaluation judging"
        )
    api_key = keys_raw.split(",")[0].strip()
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEYS is set but empty")
    return genai.Client(api_key=api_key)


def _build_judge_user_message(question: str, ground_truth: str, generated: str) -> str:
    return (
        f"Question: {question}\n\n"
        f"Reference answer (ground truth):\n{ground_truth}\n\n"
        f"Generated answer (agent):\n{generated}"
    )


def _parse_judge_response(text: str) -> tuple[Optional[bool], Optional[str], Optional[str]]:
    try:
        parsed = parse_json_response(text)
    except (json.JSONDecodeError, ValueError) as e:
        return None, None, f"Failed to parse judge JSON: {e}"

    acceptable = parsed.get("acceptable")
    reasoning = parsed.get("reasoning")
    if not isinstance(acceptable, bool):
        return None, None, f"Judge response missing boolean 'acceptable': {parsed!r}"
    if not isinstance(reasoning, str):
        return None, None, f"Judge response missing string 'reasoning': {parsed!r}"
    return acceptable, reasoning, None


def judge_answer(
    client: genai.Client,
    question: str,
    ground_truth: str,
    generated: str,
    *,
    model: str,
    temperature: float,
    thinking_budget: int,
    system_instruction: str,
) -> tuple[Optional[bool], Optional[str], Optional[str]]:
    user_message = _build_judge_user_message(question, ground_truth, generated)
    try:
        raw = gemini_generate_answer(
            client,
            prompt=user_message,
            model=model,
            thinking_budget=thinking_budget,
            temperature=temperature,
            system_instruction=system_instruction,
        )
    except Exception as e:
        return None, None, f"Judge API error: {e}"

    acceptable, reasoning, error = _parse_judge_response(raw)
    if error is None:
        return acceptable, reasoning, None

    retry_prompt = (
        user_message
        + "\n\nYour previous response was not valid JSON. "
        "Respond with strict JSON only, no markdown, keys: acceptable (boolean), reasoning (string)."
    )
    try:
        raw_retry = gemini_generate_answer(
            client,
            prompt=retry_prompt,
            model=model,
            thinking_budget=thinking_budget,
            temperature=temperature,
            system_instruction=system_instruction,
        )
    except Exception as e:
        return None, None, f"Judge API error on retry: {e}"
    return _parse_judge_response(raw_retry)


def run_evaluation(
    eval_cfg: dict,
    *,
    limit: Optional[int] = None,
    fresh: bool = False,
) -> Path:
    dataset_path = eval_cfg.get("dataset_path")
    if not dataset_path:
        raise ValueError("eval config must specify dataset_path")

    output_dir = _resolve_path(Path(eval_cfg.get("output_dir", "eval_runs")))
    results_file = eval_cfg.get("results_file", "results.jsonl")
    resume = bool(eval_cfg.get("resume", True)) and not fresh
    model_path = eval_cfg.get("model_path", "Qwen/Qwen3-8B")
    cfg_limit = eval_cfg.get("limit")
    effective_limit = limit if limit is not None else cfg_limit

    judge_cfg = eval_cfg.get("judge", {}) or {}
    judge_model = judge_cfg.get("model", "gemini-2.5-flash")
    judge_temperature = float(judge_cfg.get("temperature", 0.0))
    judge_thinking_budget = int(judge_cfg.get("thinking_budget", 0))
    judge_system = judge_cfg.get("system_instruction") or DEFAULT_JUDGE_SYSTEM_INSTRUCTION

    all_items = load_dataset(Path(dataset_path))
    total_dataset_size = len(all_items)
    if effective_limit is not None:
        all_items = all_items[: int(effective_limit)]

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / results_file
    summary_path = output_dir / "summary.json"

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
        if existing is not None and is_judged(existing):
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
                f"  Accuracy: {stats['acceptable_count']}/{stats['completed_count']} "
                f"({stats['accuracy']:.1%})"
            )
        print(f"  Results: {results_path}")
        return output_dir

    if judged_count or judge_retry_count or agent_pending_count:
        agent_work_count = sum(1 for _, m in work_items if m == "full")
        print(
            f"Resuming: {judged_count} judged, {judge_retry_count} to re-judge only, "
            f"{agent_work_count} need agent"
        )

    judge_client = _build_judge_client()
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
            print("  -> re-judging (agent output preserved)")
            t0 = time.perf_counter()
            acceptable, reasoning, error = judge_answer(
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
            existing["acceptable"] = acceptable
            existing["reasoning"] = reasoning
            existing["error"] = error
            by_id[item.id] = existing
            chain_length = existing.get("chain_length", 0)
            agent_duration = existing.get("duration_s", 0)
        else:
            t0 = time.perf_counter()
            try:
                generated, agent_steps, chain_length, slides_count = run_agent_with_trace(
                    model_path, item.question
                )
            except Exception as e:
                print(f"  -> agent error (not saved, will retry on resume): {e}")
                processed_in_run -= 1
                continue

            agent_duration = time.perf_counter() - t0

            acceptable, reasoning, error = judge_answer(
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
                    acceptable=acceptable,
                    reasoning=reasoning,
                    error=error,
                    duration_s=agent_duration,
                    slides_count=slides_count,
                    chain_length=chain_length,
                    agent_steps=agent_steps,
                )
            )
            judge_duration = 0.0

        persist_results(results_path, all_items, by_id)

        rows = ordered_results_rows(all_items, by_id)
        stats = compute_summary_stats(rows, total_dataset_size)
        summary = {
            "started_at": started_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
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

        acceptable = by_id[item.id].get("acceptable")
        status = "acceptable" if acceptable else ("not acceptable" if acceptable is False else "judge failed")
        if mode == "judge_only":
            print(f"  -> {status} (judge {judge_duration:.1f}s)")
        else:
            print(f"  -> {status} ({agent_duration:.1f}s, chain_length={chain_length})")

    rows = ordered_results_rows(all_items, by_id)
    stats = compute_summary_stats(rows, total_dataset_size)
    print()
    print(
        f"Evaluation complete: {stats['acceptable_count']}/{stats['completed_count']} "
        f"acceptable (accuracy {stats['accuracy']:.1%})"
    )
    if stats["failed_judge_count"]:
        print(f"  Judge failures: {stats['failed_judge_count']}")
    print(f"  Results: {results_path}")
    print(f"  Summary: {summary_path}")

    return output_dir
