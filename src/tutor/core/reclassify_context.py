from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI

from tutor.modules.models.openai_model import openai_generate_answer
from tutor.utils.paths import PROJECT_ROOT

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

RECLASSIFY_SYSTEM_INSTRUCTION = """\
You are an expert deep learning educator classifying study questions.

A question is context_dependent=true ONLY if the correct answer requires:
  - a specific figure, diagram, or visual that only appears in the lecture slides
  - a specific numerical result, benchmark, or experiment unique to this course material
  - a concept named or framed in a course-specific way not derivable from standard DL literature

A question is context_dependent=false when:
  - the answer can be derived from standard deep learning knowledge (textbooks, papers, general practice)
  - the question uses course-specific language ("the professor says...", "according to the lecture...",
    "why does the professor mention...") but the underlying answer is standard theory
  - the answer explains a well-known concept, algorithm, property, or phenomenon in deep learning

CRITICAL: Judge by the ANSWER CONTENT, not the question phrasing. A question that mentions
"the professor" or "the lecture" is still context_dependent=false if the answer only conveys
general deep learning knowledge that any practitioner would know.

Respond with strict JSON only (no markdown fences): a JSON array of objects with exactly:
  [{"id": "<id>", "context_dependent": <boolean>}, ...]
Do not include any text outside the JSON array.
"""

RECLASSIFY_USER_TEMPLATE = """\
For each Q&A pair below, decide whether the answer requires specific lecture material \
(slides, figures, course-specific experiments) or whether it can be given correctly \
from general deep learning knowledge alone.

{qa_block}
"""


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    path = _resolve(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Dataset must be a JSON array, got {type(data).__name__}")
    return data


def _save_dataset(path: Path, samples: list[dict[str, Any]]) -> None:
    path = _resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

class _ReclassifyClient:
    def __init__(self, model: str, temperature: float) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY environment variable is required")
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str) -> str:
        return openai_generate_answer(
            self.client,
            prompt=prompt,
            model=self.model,
            temperature=self.temperature,
            system_instruction=RECLASSIFY_SYSTEM_INSTRUCTION,
        )


# ---------------------------------------------------------------------------
# Batch classifier
# ---------------------------------------------------------------------------

def _build_qa_block(samples: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for s in samples:
        parts.append(
            f"ID: {s['id']}\n"
            f"Question: {s['question']}\n"
            f"Answer: {s['answer']}"
        )
    return "\n\n".join(parts)


def _parse_json_list(raw: str) -> list[dict[str, Any]]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON array, got {type(parsed).__name__}")
    return parsed


def reclassify_batch(
    client: _ReclassifyClient,
    samples_batch: list[dict[str, Any]],
) -> dict[str, bool]:
    """Call the LLM on one batch of samples; return {id: context_dependent}."""
    qa_block = _build_qa_block(samples_batch)
    prompt = RECLASSIFY_USER_TEMPLATE.format(qa_block=qa_block)
    raw = client.generate(prompt)
    items = _parse_json_list(raw)
    result: dict[str, bool] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        cd = item.get("context_dependent")
        if item_id is not None and cd is not None:
            result[str(item_id)] = bool(cd)
    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_reclassify_context(
    *,
    dataset_path: Path,
    model: str = "gpt-5.4-mini",
    temperature: float = 0.0,
    batch_size: int = 20,
    dry_run: bool = False,
    ids_filter: Optional[list[str]] = None,
) -> None:
    """Re-classify context_dependent for all (or filtered) samples in the dataset."""
    samples = _load_dataset(dataset_path)
    print(f"Loaded {len(samples)} sample(s) from {_resolve(dataset_path)}")

    targets = samples
    if ids_filter:
        id_set = set(ids_filter)
        targets = [s for s in samples if str(s.get("id", "")) in id_set]
        print(f"Filtering to {len(targets)} sample(s) matching --ids")

    if not targets:
        print("No samples to reclassify.")
        return

    client = _ReclassifyClient(model=model, temperature=temperature)
    print(f"Model: {model}  |  batch_size: {batch_size}  |  dry_run: {dry_run}")

    id_to_sample = {str(s.get("id", i)): s for i, s in enumerate(samples)}
    changed = 0
    failed_batches = 0

    for batch_start in range(0, len(targets), batch_size):
        batch = targets[batch_start : batch_start + batch_size]
        batch_end = min(batch_start + batch_size, len(targets))
        print(f"  Batch {batch_start + 1}–{batch_end} / {len(targets)} ...", end=" ", flush=True)

        try:
            classifications = reclassify_batch(client, batch)
        except Exception as exc:
            print(f"ERROR: {exc}. Skipping batch.")
            failed_batches += 1
            continue

        batch_changes = 0
        for sample_id, new_cd in classifications.items():
            if sample_id not in id_to_sample:
                continue
            s = id_to_sample[sample_id]
            old_cd = s.get("context_dependent")
            if old_cd != new_cd:
                batch_changes += 1
                changed += 1
                if dry_run:
                    old_label = str(old_cd) if old_cd is not None else "missing"
                    print(
                        f"\n    [dry-run] {sample_id}: "
                        f"{old_label} → {new_cd}"
                    )
                else:
                    s["context_dependent"] = new_cd

        print(f"done ({len(classifications)} classified, {batch_changes} changed)")

        if not dry_run:
            _save_dataset(dataset_path, samples)

    # Summary
    true_count = sum(1 for s in samples if s.get("context_dependent") is True)
    false_count = sum(1 for s in samples if s.get("context_dependent") is False)

    print(
        f"\nReclassification complete: {changed} value(s) changed"
        + (f", {failed_batches} batch(es) failed" if failed_batches else "")
        + ("  [DRY RUN — no changes written]" if dry_run else "")
    )
    if not dry_run:
        print(
            f"Dataset totals: context_dependent=true: {true_count}, "
            f"context_dependent=false: {false_count}"
        )
        print(f"Saved to {_resolve(dataset_path)}")
