from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from google import genai

from tutor.modules.models.gemini import gemini_generate_answer
from tutor.utils.paths import PROJECT_ROOT

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

GENERATION_SYSTEM_INSTRUCTION = """\
You are a question-and-answer generator for a university deep learning course.
Given a full lecture transcript, generate student study questions — the kind of short,
direct question a student might ask while reviewing lecture material.
Some questions should rely on content only found in this specific lecture (figures,
named examples, professor-specific wording, references to specific experiments),
while others should be answerable from general deep-learning knowledge alone.
Respond with strict JSON only (no markdown fences): a JSON array of objects with
exactly these keys:
- "question": string
- "answer": string
- "context_dependent": boolean (true if the question requires this specific lecture's
  content to answer correctly; false if any deep-learning practitioner could answer
  it from general knowledge)
Do not include any text outside the JSON array.
"""

GENERATION_USER_TEMPLATE = """\
Lecture: {lecture_name}

Transcript:
{full_transcript}

Generate {k} questions and answers for this lecture.
Aim for a mix: roughly half context_dependent=true and half context_dependent=false.
"""

BACKFILL_SYSTEM_INSTRUCTION = """\
You are classifying whether study questions about deep learning require specific
lecture content (slides, professor examples, figures, named experiments) to answer
correctly, or whether they can be answered from general knowledge.
Respond with strict JSON only (no markdown fences): a JSON array of objects with
exactly these keys:
- "id": string (unchanged from input)
- "context_dependent": boolean
Do not include any text outside the JSON array.
"""

BACKFILL_USER_TEMPLATE = """\
Classify each of the following questions. Set context_dependent to true if answering
correctly requires information specific to the lecture that introduced it (e.g. a
named figure, a specific example from slides, a concept framed in a particular way
by the professor), and false if any competent deep-learning practitioner could answer
it from general knowledge.

Questions:
{questions_block}
"""


# ---------------------------------------------------------------------------
# Gemini client helpers
# ---------------------------------------------------------------------------

class RotatingGeminiClient:
    """Multi-key Gemini client with automatic rotation on quota/API errors."""

    def __init__(self, model: str, temperature: float = 0.5) -> None:
        keys_raw = os.getenv("GEMINI_API_KEYS")
        if not keys_raw:
            raise EnvironmentError("GEMINI_API_KEYS environment variable is required")
        self.api_keys = [k.strip() for k in keys_raw.split(",") if k.strip()]
        if not self.api_keys:
            raise EnvironmentError("GEMINI_API_KEYS is set but contains no valid keys")
        self.model = model
        self.temperature = temperature
        self._idx = 0
        self._errors = [0] * len(self.api_keys)
        self.client = genai.Client(api_key=self.api_keys[self._idx])
        print(f"Gemini client: {len(self.api_keys)} API key(s), model={model}")

    def _rotate(self, exc: Exception) -> None:
        self._errors[self._idx] += 1
        print(f"  Key[{self._idx}] error ({self._errors[self._idx]}): {exc}. Rotating...")
        if all(e >= 2 for e in self._errors):
            raise RuntimeError(
                f"All {len(self.api_keys)} Gemini API key(s) exhausted."
            ) from exc
        self._idx = (self._idx + 1) % len(self.api_keys)
        self.client = genai.Client(api_key=self.api_keys[self._idx])

    def generate(self, prompt: str, system_instruction: str) -> str:
        while True:
            try:
                return gemini_generate_answer(
                    self.client,
                    prompt=prompt,
                    model=self.model,
                    thinking_budget=0,
                    temperature=self.temperature,
                    system_instruction=system_instruction,
                )
            except Exception as exc:
                self._rotate(exc)


# ---------------------------------------------------------------------------
# Dataset I/O
# ---------------------------------------------------------------------------

def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_qa_dataset(path: Path) -> list[dict[str, Any]]:
    path = _resolve_path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"Dataset at {path} must be a JSON list")
    return raw


def save_qa_dataset(path: Path, samples: list[dict[str, Any]]) -> None:
    path = _resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------

def _parse_ex_id(id_str: str) -> Optional[int]:
    m = re.fullmatch(r"ex(\d+)", str(id_str))
    return int(m.group(1)) if m else None


def next_id(existing: list[dict[str, Any]]) -> str:
    nums = [n for s in existing if (n := _parse_ex_id(s.get("id", ""))) is not None]
    return f"ex{max(nums) + 1}" if nums else "ex1"


def next_ids(existing: list[dict[str, Any]], count: int) -> list[str]:
    nums = [n for s in existing if (n := _parse_ex_id(s.get("id", ""))) is not None]
    start = (max(nums) + 1) if nums else 1
    return [f"ex{start + i}" for i in range(count)]


# ---------------------------------------------------------------------------
# Transcript loading
# ---------------------------------------------------------------------------

def build_full_transcript(chunks: list[dict[str, Any]]) -> str:
    parts = []
    for chunk in chunks:
        slide_num = chunk.get("slide_index", 0) + 1
        text = chunk.get("transcript", "").strip()
        if text:
            parts.append(f"[slide {slide_num}]\n{text}")
    return "\n\n".join(parts)


def load_lecture_transcript(retriever: Any, document_name: str) -> str:
    chunks = retriever.load_document_transcripts_only(document_name)
    return build_full_transcript(chunks)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _parse_json_list(raw: str) -> list[dict[str, Any]]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON list, got {type(parsed).__name__}")
    return parsed


def generate_qa_for_lecture(
    client: RotatingGeminiClient,
    lecture_name: str,
    transcript: str,
    k: int,
) -> list[dict[str, Any]]:
    prompt = GENERATION_USER_TEMPLATE.format(
        lecture_name=lecture_name,
        full_transcript=transcript,
        k=k,
    )
    raw = client.generate(prompt, system_instruction=GENERATION_SYSTEM_INSTRUCTION)
    items = _parse_json_list(raw)
    validated: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        question = item.get("question")
        answer = item.get("answer")
        context_dependent = item.get("context_dependent")
        if not question or not answer:
            continue
        validated.append(
            {
                "question": str(question),
                "answer": str(answer),
                "context_dependent": bool(context_dependent),
                "lecture": lecture_name,
            }
        )
    return validated


# ---------------------------------------------------------------------------
# Backfill
# ---------------------------------------------------------------------------

def _build_backfill_questions_block(samples: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for s in samples:
        lines.append(
            f"ID: {s['id']}\n"
            f"Question: {s['question']}\n"
            f"Answer: {s['answer']}"
        )
    return "\n\n".join(lines)


def backfill_context_dependent(
    client: RotatingGeminiClient,
    samples_batch: list[dict[str, Any]],
) -> dict[str, bool]:
    questions_block = _build_backfill_questions_block(samples_batch)
    prompt = BACKFILL_USER_TEMPLATE.format(questions_block=questions_block)
    raw = client.generate(prompt, system_instruction=BACKFILL_SYSTEM_INSTRUCTION)
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

def run_qa_generation(
    tutor_cfg: dict[str, Any],
    eval_cfg: dict[str, Any],
    *,
    output_path: Path,
    k: int = 5,
    backfill_batch_size: int = 20,
    lecture_filter: Optional[str] = None,
    overwrite: bool = False,
    backfill_only: bool = False,
    no_backfill: bool = False,
    gemini_model: str = "gemini-2.5-flash",
    gemini_temperature: float = 0.5,
) -> None:
    from tutor.modules.retrieval.retriever import Retriever

    if overwrite and _resolve_path(output_path).exists():
        _resolve_path(output_path).unlink()
        print(f"Overwrite: removed existing {output_path}")

    samples = load_qa_dataset(output_path)
    client = RotatingGeminiClient(model=gemini_model, temperature=gemini_temperature)

    # ------------------------------------------------------------------
    # Phase 1: backfill context_dependent for samples that lack it
    # ------------------------------------------------------------------
    if not no_backfill:
        missing = [s for s in samples if "context_dependent" not in s]
        if missing:
            print(
                f"Backfill: {len(missing)} sample(s) missing 'context_dependent' "
                f"(batch size={backfill_batch_size})"
            )
            id_to_idx = {s["id"]: i for i, s in enumerate(samples)}
            for batch_start in range(0, len(missing), backfill_batch_size):
                batch = missing[batch_start : batch_start + backfill_batch_size]
                batch_end = min(batch_start + backfill_batch_size, len(missing))
                print(f"  Backfill batch {batch_start + 1}–{batch_end}/{len(missing)}")
                try:
                    classifications = backfill_context_dependent(client, batch)
                except Exception as e:
                    print(f"  Backfill batch error: {e}. Skipping batch.")
                    continue
                for sample_id, cd in classifications.items():
                    if sample_id in id_to_idx:
                        samples[id_to_idx[sample_id]]["context_dependent"] = cd
                save_qa_dataset(output_path, samples)
            print(f"Backfill complete. Dataset saved to {output_path}")
        else:
            print("Backfill: all samples already have 'context_dependent'. Skipping.")

    if backfill_only:
        return

    # ------------------------------------------------------------------
    # Phase 2: generate new Q&A for each lecture
    # ------------------------------------------------------------------
    retriever = Retriever(tutor_cfg)
    lecture_names = retriever.documents_names

    if lecture_filter:
        lecture_names = [n for n in lecture_names if lecture_filter.lower() in n.lower()]
        if not lecture_names:
            print(f"No lectures matching '{lecture_filter}'. Available: {retriever.documents_names}")
            return

    print(f"Generating {k} Q&A pair(s) for {len(lecture_names)} lecture(s)")
    for lecture_name in sorted(lecture_names):
        print(f"  Lecture: {lecture_name}")
        try:
            transcript = load_lecture_transcript(retriever, lecture_name)
        except Exception as e:
            print(f"    Could not load transcript: {e}. Skipping.")
            continue

        if not transcript.strip():
            print(f"    Empty transcript. Skipping.")
            continue

        try:
            new_items = generate_qa_for_lecture(client, lecture_name, transcript, k)
        except Exception as e:
            print(f"    Generation error: {e}. Skipping.")
            continue

        if not new_items:
            print(f"    No valid items returned. Skipping.")
            continue

        ids = next_ids(samples, len(new_items))
        for item, item_id in zip(new_items, ids):
            item["id"] = item_id
            samples.append(item)

        save_qa_dataset(output_path, samples)
        print(f"    Added {len(new_items)} item(s) (ids {ids[0]}–{ids[-1]}). Total: {len(samples)}")

    print(f"\nDone. Dataset has {len(samples)} sample(s) at {_resolve_path(output_path)}")
