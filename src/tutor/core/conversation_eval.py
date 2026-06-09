from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from tutor.core.evaluation import (
    EvalItem,
    RotatingJudgeClient,
    load_dataset,
    load_results_by_id,
    ordered_results_rows,
    resolve_mode_paths,
    rotating_gemini_generate,
    save_all_results,
    write_summary,
)
from tutor.modules.agent.summarizer import ConversationMemory
from tutor.modules.agent.pedagogy import TeachingAnchor, TeachingSession
from tutor.modules.agent.tutor_orchestrator import run_tutor_turn
from tutor.server.inference import _cached_model, get_rag_module
from tutor.utils.config import load_config
from tutor.utils.misc import parse_json_response
from tutor.utils.paths import DEFAULT_CONFIG_PATH

StudentSignal = Literal["understood", "give_up"]
StudentEnded = Literal["understood", "gave_up", "max_turns"]

UNDERSTOOD_MARKER = "[UNDERSTOOD]"
GIVE_UP_MARKER = "[GIVE_UP]"

_UNDERSTOOD_RE = re.compile(r"\[UNDERSTOOD\]", re.IGNORECASE)
_GIVE_UP_RE = re.compile(r"\[GIVE_UP\]", re.IGNORECASE)


DEFAULT_STUDENT_SYSTEM_INSTRUCTION = """\
You are role-playing as a university student in a deep learning course. You have a genuine doubt and you do NOT initially know the answer. You are talking to a Socratic tutor that will deliberately NOT tell you the answer directly, but will guide you with questions, hints, and small explanations.

How to behave:
- Engage genuinely and think step by step. Try to answer the tutor's questions and follow its hints.
- Reason ONLY from the tutor's guidance plus basic reasoning. Do NOT use outside knowledge to jump straight to the final answer; work it out gradually from what the tutor tells you.
- Stay in character as a learner: it is fine to be unsure, to make a guess, or to ask a clarifying question.

Ending the conversation (use these markers literally, on their own at the end of your message):
- When you genuinely believe you understand and can state the answer to your ORIGINAL question, write your answer in your own words and end the message with {understood}.
- If after several exchanges the hints are not helping you make progress, you may give up and explicitly ask the tutor to just tell you the answer; end that message with {give_up}.
- Consider giving up only after roughly {give_up_after} unhelpful exchanges; otherwise keep trying.

Keep each message concise (1-4 sentences). Do not narrate these instructions."""


DEFAULT_CONVERSATION_JUDGE_SYSTEM_INSTRUCTION = """\
You are an impartial grader evaluating a Socratic tutoring conversation.

You are given the student's ORIGINAL question, the REFERENCE answer (ground truth), and the full numbered TRANSCRIPT of a conversation between a simulated student and a Socratic tutor. The tutor is supposed to guide the student to the answer WITHOUT revealing it, unless the student explicitly gives up.

Decide two things, comparing what is said against the reference answer:
1. Did the STUDENT correctly articulate the answer to the original question on their own at some point? Merely saying "I understand" is NOT enough; the student must actually state the correct content. If yes, give the 1-based TURN number where the student first states it correctly.
2. Did the TUTOR explicitly reveal the full final answer (i.e. tell the student the answer rather than just hinting)? If yes, give the 1-based TURN number of the tutor message where it first reveals it.

A "turn" is one student message plus the tutor's reply (turn numbers are shown in the transcript).

Respond with strict JSON only (no markdown fences), using exactly these keys:
- "reasoning": string (2-5 sentences justifying both judgments)
- "student_reached": boolean
- "student_reach_turn": integer or null (1-based turn; null if student_reached is false)
- "tutor_revealed": boolean
- "tutor_reveal_turn": integer or null (1-based turn; null if tutor_revealed is false)
"""


# --------------------------------------------------------------------------- #
# Student simulator
# --------------------------------------------------------------------------- #
def parse_student_signal(text: str) -> tuple[str, Optional[StudentSignal]]:
    """Strip [UNDERSTOOD]/[GIVE_UP] markers and report which (if any) appeared.

    [UNDERSTOOD] takes precedence if both somehow appear.
    """
    if not text:
        return "", None
    signal: Optional[StudentSignal] = None
    if _UNDERSTOOD_RE.search(text):
        signal = "understood"
    elif _GIVE_UP_RE.search(text):
        signal = "give_up"
    clean = _UNDERSTOOD_RE.sub("", text)
    clean = _GIVE_UP_RE.sub("", clean)
    clean = clean.strip()
    return clean, signal


def _format_transcript_for_student(transcript: list[dict]) -> str:
    lines: list[str] = []
    for entry in transcript:
        role = "Tutor" if entry["role"] == "tutor" else "You (student)"
        lines.append(f"{role}: {entry['content']}")
    return "\n".join(lines)


def _build_student_prompt(question: str, transcript: list[dict]) -> str:
    history = _format_transcript_for_student(transcript)
    return (
        f"Your original question for the tutor was:\n{question}\n\n"
        f"Conversation so far:\n{history}\n\n"
        "Write your next message to the tutor."
    )


def generate_student_message(
    judge_client: RotatingJudgeClient,
    question: str,
    transcript: list[dict],
    student_cfg: dict,
) -> tuple[str, Optional[StudentSignal]]:
    model = student_cfg.get("model", "gemini-2.5-flash")
    temperature = float(student_cfg.get("temperature", 0.7))
    thinking_budget = int(student_cfg.get("thinking_budget", 0))
    give_up_after = int(student_cfg.get("give_up_after", 6))
    system_instruction = student_cfg.get("system_instruction") or (
        DEFAULT_STUDENT_SYSTEM_INSTRUCTION.format(
            understood=UNDERSTOOD_MARKER,
            give_up=GIVE_UP_MARKER,
            give_up_after=give_up_after,
        )
    )
    prompt = _build_student_prompt(question, transcript)
    raw = rotating_gemini_generate(
        judge_client,
        prompt,
        model=model,
        temperature=temperature,
        thinking_budget=thinking_budget,
        system_instruction=system_instruction,
    )
    return parse_student_signal(str(raw))


# --------------------------------------------------------------------------- #
# Conversation judge
# --------------------------------------------------------------------------- #
def _format_transcript_for_judge(transcript: list[dict]) -> str:
    lines: list[str] = []
    for entry in transcript:
        role = "TUTOR" if entry["role"] == "tutor" else "STUDENT"
        lines.append(f"[turn {entry['turn']}] {role}: {entry['content']}")
    return "\n".join(lines)


def _coerce_turn(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        turn = int(value)
    except (TypeError, ValueError):
        return None
    return turn if turn >= 1 else None


def _parse_conversation_judge_response(
    text: str,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    try:
        parsed = parse_json_response(text)
    except (json.JSONDecodeError, ValueError) as e:
        return None, f"Failed to parse conversation judge JSON: {e}"

    reasoning = parsed.get("reasoning")
    if not isinstance(reasoning, str):
        return None, f"Judge response missing string 'reasoning': {parsed!r}"
    if not isinstance(parsed.get("student_reached"), bool):
        return None, f"Judge response missing boolean 'student_reached': {parsed!r}"
    if not isinstance(parsed.get("tutor_revealed"), bool):
        return None, f"Judge response missing boolean 'tutor_revealed': {parsed!r}"

    student_reached = bool(parsed["student_reached"])
    tutor_revealed = bool(parsed["tutor_revealed"])
    result = {
        "student_reached": student_reached,
        "student_reach_turn": _coerce_turn(parsed.get("student_reach_turn"))
        if student_reached
        else None,
        "tutor_revealed": tutor_revealed,
        "tutor_reveal_turn": _coerce_turn(parsed.get("tutor_reveal_turn"))
        if tutor_revealed
        else None,
        "judge_reasoning": reasoning,
    }
    return result, None


def judge_conversation(
    judge_client: RotatingJudgeClient,
    item: EvalItem,
    transcript: list[dict],
    *,
    model: str,
    temperature: float,
    thinking_budget: int,
    system_instruction: str,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    transcript_text = _format_transcript_for_judge(transcript)
    user_message = (
        f"Original question:\n{item.question}\n\n"
        f"Reference answer (ground truth):\n{item.answer}\n\n"
        f"Transcript:\n{transcript_text}"
    )

    def _call(prompt: str) -> str:
        return rotating_gemini_generate(
            judge_client,
            prompt,
            model=model,
            temperature=temperature,
            thinking_budget=thinking_budget,
            system_instruction=system_instruction,
        )

    try:
        raw = _call(user_message)
    except Exception as e:
        return None, f"Conversation judge API error: {e}"

    result, error = _parse_conversation_judge_response(raw)
    if error is None:
        return result, None

    retry_prompt = (
        user_message
        + "\n\nYour previous response was not valid JSON. Respond with strict JSON only, "
        "no markdown, keys: reasoning (string), student_reached (boolean), "
        "student_reach_turn (integer or null), tutor_revealed (boolean), "
        "tutor_reveal_turn (integer or null)."
    )
    try:
        raw_retry = _call(retry_prompt)
    except Exception as e:
        return None, f"Conversation judge API error on retry: {e}"
    return _parse_conversation_judge_response(raw_retry)


# --------------------------------------------------------------------------- #
# Conversation runner
# --------------------------------------------------------------------------- #
def _new_conversation_row(item: EvalItem) -> dict[str, Any]:
    return {
        "id": item.id,
        "question": item.question,
        "ground_truth": item.answer,
        "eval_mode": "conversation",
        "status": "in_progress",
        "transcript": [],
        "num_turns": 0,
        "student_ended": None,
        "memory_state": ConversationMemory.empty().to_dict(),
        "session_state": TeachingSession.empty().to_dict(),
        "student_reached": None,
        "student_reach_turn": None,
        "tutor_revealed": None,
        "tutor_reveal_turn": None,
        "judge_reasoning": None,
        "error": None,
    }


def run_one_conversation(
    qwen_model: Any,
    rag_module: Any,
    config: dict,
    conv_cfg: dict,
    item: EvalItem,
    judge_client: RotatingJudgeClient,
    row: dict[str, Any],
    persist: Any,
) -> dict[str, Any]:
    """Drive (or resume) one student<->tutor conversation to completion.

    Persists ``row`` after every appended transcript entry via ``persist``
    (a no-arg callable closing over the results path + by_id map), so an
    interruption mid-conversation never loses progress.
    """
    max_turns = int(conv_cfg.get("max_turns", 40))
    student_cfg = dict(conv_cfg.get("student", {}) or {})
    student_cfg.setdefault("give_up_after", conv_cfg.get("give_up_after", 6))

    memory = ConversationMemory.from_dict(row.get("memory_state"))
    session = TeachingSession.from_dict(row.get("session_state"))
    transcript: list[dict] = list(row.get("transcript") or [])

    # Determine the next student message + whether a give-up reveal is pending.
    pending_give_up = False
    if not transcript:
        student_msg: Optional[str] = item.question
        transcript.append({"turn": 1, "role": "student", "content": item.question})
        row["transcript"] = transcript
        row["num_turns"] = 1
        persist()
    else:
        last = transcript[-1]
        if last["role"] == "student":
            student_msg = last["content"]
            pending_give_up = bool(row.get("_pending_give_up"))
        else:
            # Last entry is a tutor reply: we need a fresh student message.
            student_msg = None

    def current_turn() -> int:
        return transcript[-1]["turn"] if transcript else 1

    while True:
        # 1. Tutor responds to the latest student message (if one is pending).
        if student_msg is not None:
            move, _slides, memory, session, _debug = run_tutor_turn(
                qwen_model,
                rag_module,
                config,
                student_msg,
                memory=memory,
                session=session,
                callbacks=None,
                debug=False,
            )
            transcript.append(
                {"turn": current_turn(), "role": "tutor", "content": move}
            )
            row["transcript"] = transcript
            row["memory_state"] = memory.to_dict()
            row["session_state"] = session.to_dict()
            row["num_turns"] = current_turn()
            if row.get("anchor_bootstrap") is None and session.anchor is not None:
                row["anchor_bootstrap"] = {
                    "valid": session.anchor.is_valid(),
                    "parse_failed": not session.anchor.is_valid(),
                }
            persist()
            student_msg = None

            if pending_give_up:
                row["student_ended"] = "gave_up"
                break

        # 2. Stop if we hit the safety cap.
        if current_turn() >= max_turns:
            row["student_ended"] = "max_turns"
            break

        # 3. Student replies to the tutor's latest move.
        clean, signal = generate_student_message(
            judge_client, item.question, transcript, student_cfg
        )
        next_turn = current_turn() + 1
        transcript.append({"turn": next_turn, "role": "student", "content": clean})
        row["transcript"] = transcript
        row["num_turns"] = next_turn

        if signal == "understood":
            row["student_ended"] = "understood"
            row.pop("_pending_give_up", None)
            persist()
            break
        if signal == "give_up":
            pending_give_up = True
            row["_pending_give_up"] = True
        else:
            row.pop("_pending_give_up", None)
        persist()

        student_msg = clean

    row.pop("_pending_give_up", None)
    row["status"] = "conversation_complete"
    row["memory_state"] = memory.to_dict()
    row["session_state"] = session.to_dict()
    persist()
    return row


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def derive_outcome(row: dict[str, Any]) -> tuple[bool, Optional[int], bool, Optional[int]]:
    """Return (success, success_turn, telling, telling_turn) for one judged row."""
    reached = bool(row.get("student_reached"))
    revealed = bool(row.get("tutor_revealed"))
    reach_turn = _coerce_turn(row.get("student_reach_turn")) if reached else None
    reveal_turn = _coerce_turn(row.get("tutor_reveal_turn")) if revealed else None

    # Success: the student reached the answer on their own, i.e. not after the
    # tutor already revealed it.
    success = reached
    if reached and revealed and reach_turn is not None and reveal_turn is not None:
        success = reach_turn <= reveal_turn
    success_turn = reach_turn if success else None

    # Telling: the tutor revealed the answer before the student figured it out.
    telling = revealed
    if revealed and reached and reveal_turn is not None and reach_turn is not None:
        telling = reveal_turn < reach_turn
    telling_turn = reveal_turn if telling else None

    return success, success_turn, telling, telling_turn


def _is_judged(row: dict[str, Any]) -> bool:
    return (
        isinstance(row.get("student_reached"), bool)
        and isinstance(row.get("tutor_revealed"), bool)
        and row.get("status") == "judged"
    )


def _anchor_from_row(row: dict[str, Any]) -> Optional[TeachingAnchor]:
    session_data = row.get("session_state")
    if not isinstance(session_data, dict):
        return None
    anchor_data = session_data.get("anchor")
    if not isinstance(anchor_data, dict):
        return None
    return TeachingAnchor.from_dict(anchor_data)


def _row_has_valid_anchor(row: dict[str, Any]) -> bool:
    anchor = _anchor_from_row(row)
    return anchor is not None and anchor.is_valid()


def _needs_full_conversation_reset(row: dict[str, Any]) -> bool:
    """True when an existing row must be discarded and the conversation re-run."""
    if _is_judged(row) and not _row_has_valid_anchor(row):
        return True
    if (
        row.get("status") == "conversation_complete"
        and row.get("transcript")
        and not _row_has_valid_anchor(row)
    ):
        return True
    return False


def compute_conversation_summary(
    rows: list[dict[str, Any]],
    total_dataset_size: int,
    k_values: list[int],
) -> dict[str, Any]:
    judged = [r for r in rows if _is_judged(r)]
    judged_count = len(judged)
    failed_count = sum(
        1 for r in rows if r.get("error") and not _is_judged(r)
    )

    success_flags: list[tuple[bool, Optional[int]]] = []
    telling_flags: list[tuple[bool, Optional[int]]] = []
    turns: list[int] = []
    for r in judged:
        success, success_turn, telling, telling_turn = derive_outcome(r)
        success_flags.append((success, success_turn))
        telling_flags.append((telling, telling_turn))
        if isinstance(r.get("num_turns"), int):
            turns.append(int(r["num_turns"]))

    def rate(flags: list[tuple[bool, Optional[int]]], k: Optional[int]) -> float:
        if not judged_count:
            return 0.0
        if k is None:
            count = sum(1 for ok, _turn in flags if ok)
        else:
            count = sum(
                1 for ok, turn in flags if ok and turn is not None and turn <= k
            )
        return count / judged_count

    success_at_k = {str(k): rate(success_flags, k) for k in k_values}
    telling_at_k = {str(k): rate(telling_flags, k) for k in k_values}

    return {
        "judged_count": judged_count,
        "total_dataset_size": total_dataset_size,
        "failed_count": failed_count,
        "success_overall": rate(success_flags, None),
        "telling_overall": rate(telling_flags, None),
        "success_at_k": success_at_k,
        "telling_at_k": telling_at_k,
        "avg_turns": (sum(turns) / len(turns)) if turns else 0.0,
    }


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def _classify_work(row: Optional[dict[str, Any]]) -> str:
    """Return 'skip' | 'judge_only' | 'run'."""
    if row is None:
        return "run"
    if _is_judged(row):
        if _row_has_valid_anchor(row):
            return "skip"
        return "run"
    if row.get("status") == "conversation_complete" and row.get("transcript"):
        if _row_has_valid_anchor(row):
            return "judge_only"
        return "run"
    return "run"


def run_conversation_evaluation(
    eval_cfg: dict,
    *,
    limit: Optional[int] = None,
    fresh: bool = False,
) -> Path:
    dataset_path = eval_cfg.get("dataset_path")
    if not dataset_path:
        raise ValueError("eval config must specify dataset_path")

    conv_cfg, results_path, summary_path = resolve_mode_paths(eval_cfg, "conversation")
    output_dir = results_path.parent
    resume = bool(eval_cfg.get("resume", True)) and not fresh
    model_path = eval_cfg.get("model_path", "Qwen/Qwen3-8B")
    cfg_limit = eval_cfg.get("limit")
    effective_limit = limit if limit is not None else cfg_limit

    k_values = [int(k) for k in (conv_cfg.get("k_values") or [1, 3, 5, 10])]

    judge_cfg = eval_cfg.get("judge", {}) or {}
    judge_model = judge_cfg.get("model", "gemini-2.5-flash")
    judge_temperature = float(judge_cfg.get("temperature", 0.0))
    judge_thinking_budget = int(judge_cfg.get("thinking_budget", 0))
    judge_system = (
        conv_cfg.get("judge_system_instruction")
        or DEFAULT_CONVERSATION_JUDGE_SYSTEM_INSTRUCTION
    )
    student_model = (conv_cfg.get("student", {}) or {}).get("model", "gemini-2.5-flash")

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

    def persist() -> None:
        save_all_results(results_path, ordered_results_rows(all_items, by_id))

    def rewrite_summary() -> None:
        rows = ordered_results_rows(all_items, by_id)
        stats = compute_conversation_summary(rows, total_dataset_size, k_values)
        summary = {
            "started_at": started_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "eval_mode": "conversation",
            "results_path": str(results_path),
            "dataset_path": str(dataset_path),
            "model_path": model_path,
            "student": {"model": student_model},
            "judge": {
                "model": judge_model,
                "temperature": judge_temperature,
                "thinking_budget": judge_thinking_budget,
            },
            "k_values": k_values,
            **stats,
        }
        write_summary(summary_path, summary)

    work_items: list[tuple[EvalItem, str]] = []
    for item in all_items:
        work = _classify_work(by_id.get(item.id))
        if work != "skip":
            work_items.append((item, work))

    started_at = datetime.now(timezone.utc).isoformat()

    skipped = len(all_items) - len(work_items)
    if not work_items:
        print(f"All {len(all_items)} conversations already evaluated (resume).")
        rewrite_summary()
        print(f"  Results: {results_path}")
        return output_dir

    run_count = sum(1 for _, w in work_items if w == "run")
    judge_only_count = sum(1 for _, w in work_items if w == "judge_only")
    print(
        f"Conversation eval: {skipped} done, {run_count} to run, "
        f"{judge_only_count} to judge-only"
    )

    print("Loading Qwen model + RAG...")
    cfg = load_config(DEFAULT_CONFIG_PATH)
    qwen_model, _ = _cached_model(model_path)
    rag_module = get_rag_module()
    judge_client = RotatingJudgeClient()

    processed = 0
    for item, work in work_items:
        processed += 1
        print(f"Conversation {processed}/{len(work_items)}: {item.id} ({work})")

        row = by_id.get(item.id) or _new_conversation_row(item)
        by_id[item.id] = row

        try:
            if work == "run":
                if _needs_full_conversation_reset(row):
                    row = _new_conversation_row(item)
                    by_id[item.id] = row
                t0 = time.perf_counter()
                row = run_one_conversation(
                    qwen_model,
                    rag_module,
                    cfg,
                    conv_cfg,
                    item,
                    judge_client,
                    row,
                    persist,
                )
                by_id[item.id] = row
                print(
                    f"  -> conversation complete: {row.get('num_turns')} turns, "
                    f"ended={row.get('student_ended')} ({time.perf_counter() - t0:.1f}s)"
                )

            # Judge (for both freshly-run and judge_only items).
            result, error = judge_conversation(
                judge_client,
                item,
                row.get("transcript") or [],
                model=judge_model,
                temperature=judge_temperature,
                thinking_budget=judge_thinking_budget,
                system_instruction=judge_system,
            )
            if error is not None:
                row["error"] = error
                by_id[item.id] = row
                persist()
                print(f"  -> judge failed (will retry on resume): {error}")
                continue

            row.update(result)
            row["status"] = "judged"
            row["error"] = None
            by_id[item.id] = row
            persist()
            rewrite_summary()

            success, success_turn, telling, telling_turn = derive_outcome(row)
            print(
                f"  -> judged: reached={row['student_reached']}@{row['student_reach_turn']}, "
                f"revealed={row['tutor_revealed']}@{row['tutor_reveal_turn']} "
                f"(success={success}, telling={telling})"
            )
        except Exception as e:
            row["error"] = str(e)
            by_id[item.id] = row
            persist()
            print(f"  -> error (progress saved, moving on): {e}")
            continue

    rewrite_summary()
    rows = ordered_results_rows(all_items, by_id)
    stats = compute_conversation_summary(rows, total_dataset_size, k_values)
    print()
    print(f"Conversation evaluation complete: {stats['judged_count']} judged")
    print(
        f"  Success@k: "
        + ", ".join(f"@{k}={stats['success_at_k'][str(k)]:.1%}" for k in k_values)
    )
    print(
        f"  Telling@k: "
        + ", ".join(f"@{k}={stats['telling_at_k'][str(k)]:.1%}" for k in k_values)
    )
    print(
        f"  Overall: success={stats['success_overall']:.1%}, "
        f"telling={stats['telling_overall']:.1%}, avg_turns={stats['avg_turns']:.1f}"
    )
    if stats["failed_count"]:
        print(f"  Failed/incomplete: {stats['failed_count']}")
    print(f"  Results: {results_path}")
    print(f"  Summary: {summary_path}")

    return output_dir
