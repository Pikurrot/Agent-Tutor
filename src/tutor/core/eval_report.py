from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from tutor.utils.paths import PROJECT_ROOT

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACCURACY_MODES = ["llm_baseline", "llm_context", "rag", "agent"]

MODE_DISPLAY_NAMES: dict[str, str] = {
    "llm_baseline": "LLM Baseline (no context)",
    "llm_context": "LLM Full Context",
    "rag": "RAG (single-pass)",
    "agent": "Agent (ReAct+RAG)",
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def load_mode_data(
    output_dir: Path,
    mode_cfg: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load (summary_dict, rows_list) for one mode; both empty when files absent."""
    summary_file = mode_cfg.get("summary_file", "")
    results_file = mode_cfg.get("results_file", "")
    summary = _load_json(output_dir / summary_file) if summary_file else {}
    rows = _load_jsonl(output_dir / results_file) if results_file else []
    return summary, rows


def _find_legacy_summary(output_dir: Path) -> dict[str, Any]:
    """Fall back to the old summary.json for the agent mode."""
    return _load_json(output_dir / "summary.json")


def _load_context_dependent_map(dataset_path: str) -> dict[str, bool]:
    """Return {question_id: context_dependent} from the questions JSON dataset."""
    path = _resolve(Path(dataset_path))
    if not path.exists():
        return {}
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    result: dict[str, bool] = {}
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        if "context_dependent" not in entry:
            continue
        qid = str(entry.get("id", i))
        result[qid] = bool(entry["context_dependent"])
    return result


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _pct(num: int, denom: int) -> str:
    if denom == 0:
        return "—"
    return f"{num / denom * 100:.1f}%"


def _metric_counts(rows: list[dict[str, Any]]) -> tuple[int, int, int, int]:
    """Return (substantial, partial, mismatch, failed_judge) from JSONL rows."""
    substantial = sum(1 for r in rows if r.get("metric") == "Substantial Match")
    partial = sum(1 for r in rows if r.get("metric") == "Partial Match")
    mismatch = sum(1 for r in rows if r.get("metric") == "Mismatch")
    failed = sum(1 for r in rows if r.get("metric") not in {"Substantial Match", "Partial Match", "Mismatch"})
    return substantial, partial, mismatch, failed


# ---------------------------------------------------------------------------
# Section 1: metadata
# ---------------------------------------------------------------------------

def section_metadata(
    eval_cfg: dict[str, Any],
    summaries: dict[str, dict[str, Any]],
) -> str:
    dataset_path = eval_cfg.get("dataset_path", "unknown")
    model_path = eval_cfg.get("model_path", "unknown")
    judge_model = eval_cfg.get("judge", {}).get("model", "unknown")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # pick total_dataset_size from any available summary
    total = next(
        (s.get("total_dataset_size") for s in summaries.values() if s.get("total_dataset_size")),
        "unknown",
    )

    lines = [
        "## Run metadata",
        "",
        f"| Key | Value |",
        f"| --- | ----- |",
        f"| Dataset | `{dataset_path}` |",
        f"| Dataset size | {total} |",
        f"| Tutor model | `{model_path}` |",
        f"| Judge model | `{judge_model}` |",
        f"| Report generated | {now} |",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 2: accuracy table
# ---------------------------------------------------------------------------

def section_accuracy_table(
    modes_data: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]],
    mode_order: list[str],
) -> str:
    lines = [
        "## Answer accuracy",
        "",
        "Grading rubric: **Substantial Match** = core meaning correct; "
        "**Partial Match** = some correct elements but misses key points; "
        "**Mismatch** = wrong or completely off.",
        "",
        "| Method | N | Substantial Match | Partial Match | Mismatch | Failed Judge |",
        "| ------ | --: | :-: | :-: | :-: | --: |",
    ]

    for mode in mode_order:
        if mode not in modes_data or mode == "conversation":
            continue
        name = MODE_DISPLAY_NAMES.get(mode, mode)
        summary, rows = modes_data[mode]

        if not summary and not rows:
            lines.append(f"| {name} | — | — | — | — | — |")
            continue

        n = summary.get("completed_count") or len(rows)
        if rows:
            substantial, partial, mismatch, failed = _metric_counts(rows)
        else:
            substantial = summary.get("substantial_match_count", 0)
            partial = summary.get("partial_match_count", 0)
            mismatch = summary.get("mismatch_count", 0)
            failed = summary.get("failed_judge_count", 0)

        lines.append(
            f"| {name} | {n} "
            f"| {_pct(substantial, n)} "
            f"| {_pct(partial, n)} "
            f"| {_pct(mismatch, n)} "
            f"| {failed} |"
        )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 3: context-dependent breakdown
# ---------------------------------------------------------------------------

def section_context_breakdown(
    modes_data: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]],
    mode_order: list[str],
    context_dep_map: Optional[dict[str, bool]] = None,
) -> str:
    """Compare each method's accuracy on context_dependent=True vs False questions.

    ``context_dep_map`` is a ``{question_id: bool}`` lookup built from the
    questions dataset. When provided, rows that lack the field are enriched via
    their ``id`` key before grouping, so the breakdown works even when result
    JSONL files were written before the field existed.
    """
    any_data = False
    table_rows: list[str] = []

    for mode in mode_order:
        if mode not in modes_data or mode == "conversation":
            continue
        _summary, rows = modes_data[mode]

        # Enrich rows with context_dependent from the map when absent in the row
        if context_dep_map:
            enriched: list[dict[str, Any]] = []
            for r in rows:
                if "context_dependent" in r:
                    enriched.append(r)
                else:
                    row_id = str(r.get("id", ""))
                    if row_id in context_dep_map:
                        enriched.append({**r, "context_dependent": context_dep_map[row_id]})
            annotated = enriched
        else:
            annotated = [r for r in rows if "context_dependent" in r]

        if not annotated:
            continue
        any_data = True
        name = MODE_DISPLAY_NAMES.get(mode, mode)

        for cd_value, label in [(True, "Yes"), (False, "No")]:
            subset = [r for r in annotated if r.get("context_dependent") is cd_value]
            if not subset:
                continue
            n = len(subset)
            substantial, partial, mismatch, _ = _metric_counts(subset)
            table_rows.append(
                f"| {name} | {label} | {n} "
                f"| {_pct(substantial, n)} "
                f"| {_pct(partial, n)} "
                f"| {_pct(mismatch, n)} |"
            )

    if not any_data:
        return (
            "## Context-dependent breakdown\n\n"
            "_No samples have the `context_dependent` field yet. "
            "Run `tutor generate-qa --backfill-only` to classify existing samples._\n\n"
        )

    lines = [
        "## Context-dependent breakdown",
        "",
        "Questions labelled **Yes** require specific lecture content to answer correctly; "
        "**No** can be answered from general knowledge alone.",
        "",
        "| Method | Context-dep. | N | Substantial Match | Partial Match | Mismatch |",
        "| ------ | :-: | --: | :-: | :-: | :-: |",
    ] + table_rows + [""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 4: agent trace statistics
# ---------------------------------------------------------------------------

def section_agent_traces(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "## Agent trace statistics\n\n_No agent results available._\n\n"

    chain_lengths = [int(r["chain_length"]) for r in rows if isinstance(r.get("chain_length"), int)]
    durations = [float(r["duration_s"]) for r in rows if isinstance(r.get("duration_s"), (int, float))]

    lines = ["## Agent trace statistics", ""]

    if chain_lengths:
        avg_cl = statistics.mean(chain_lengths)
        counts: dict[str, int] = {"0": 0, "1": 0, "2": 0, "3": 0, "4+": 0}
        for cl in chain_lengths:
            key = str(cl) if cl <= 3 else "4+"
            counts[key] = counts.get(key, 0) + 1
        n = len(chain_lengths)

        lines += [
            f"Average retrieval chain length: **{avg_cl:.2f}** across {n} samples.",
            "",
            "| Chain length | Count | Fraction |",
            "| :-: | --: | :-: |",
        ]
        for label in ["0", "1", "2", "3", "4+"]:
            c = counts[label]
            if c > 0 or label != "0":
                lines.append(f"| {label} | {c} | {_pct(c, n)} |")
        lines.append("")

    if durations:
        avg_d = statistics.mean(durations)
        med_d = statistics.median(durations)
        lines += [
            f"Response time — mean: **{avg_d:.1f}s**, median: **{med_d:.1f}s** "
            f"(min {min(durations):.1f}s, max {max(durations):.1f}s).",
            "",
        ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 5: conversation
# ---------------------------------------------------------------------------

def section_conversation(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> str:
    if not summary:
        return "## Conversation evaluation\n\n_No conversation results available._\n\n"

    judged = summary.get("judged_count", 0)
    total = summary.get("total_dataset_size", 0)
    failed = summary.get("failed_count", 0)
    success_overall = summary.get("success_overall", 0.0)
    telling_overall = summary.get("telling_overall", 0.0)
    avg_turns = summary.get("avg_turns", 0.0)
    k_values: dict[str, float] = summary.get("success_at_k", {})
    telling_values: dict[str, float] = summary.get("telling_at_k", {})

    lines = [
        "## Conversation evaluation (Socratic tutoring)",
        "",
        f"The tutor engages in multi-turn Socratic dialogue. "
        f"**Success@k**: student reaches the correct answer by turn k (without being told). "
        f"**Telling@k**: tutor reveals the answer by turn k.",
        "",
        f"Evaluated on **{judged}** / {total} samples "
        f"(avg {avg_turns:.1f} turns/conversation"
        + (f"; {failed} failed" if failed else "")
        + ").",
        f"Overall success rate (any turn): **{success_overall * 100:.1f}%** "
        f"| telling rate: **{telling_overall * 100:.1f}%**",
        "",
        "| k | Success@k | Telling@k |",
        "| --: | :-: | :-: |",
    ]

    for k_str in sorted(k_values.keys(), key=lambda x: int(x)):
        s = k_values.get(k_str, 0.0)
        t = telling_values.get(k_str, 0.0)
        lines.append(f"| {k_str} | {s * 100:.1f}% | {t * 100:.1f}% |")

    lines.append("")

    # Per-sample turn distribution from rows
    turn_counts = [int(r["num_turns"]) for r in rows if isinstance(r.get("num_turns"), int) and r["num_turns"] > 0]
    if turn_counts:
        avg_r = statistics.mean(turn_counts)
        med_r = statistics.median(turn_counts)
        lines += [
            f"Turn distribution: mean **{avg_r:.1f}**, median **{med_r:.1f}** "
            f"(min {min(turn_counts)}, max {max(turn_counts)}).",
            "",
        ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 6: coverage table
# ---------------------------------------------------------------------------

def section_coverage(
    modes_data: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]],
    mode_order: list[str],
) -> str:
    lines = [
        "## Evaluation coverage",
        "",
        "| Mode | Completed | Dataset size | Failed judge |",
        "| ---- | --: | --: | --: |",
    ]

    for mode in mode_order:
        if mode not in modes_data:
            continue
        name = MODE_DISPLAY_NAMES.get(mode, mode)
        summary, rows = modes_data[mode]

        if not summary and not rows:
            lines.append(f"| {name} | — | — | — |")
            continue

        if mode == "conversation":
            completed = summary.get("judged_count", len(rows))
            failed = summary.get("failed_count", 0)
        else:
            completed = summary.get("completed_count") or len(rows)
            failed = summary.get("failed_judge_count", 0)

        total = summary.get("total_dataset_size", "—")
        lines.append(f"| {name} | {completed} | {total} | {failed} |")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report assembler
# ---------------------------------------------------------------------------

def generate_report(
    eval_cfg: dict[str, Any],
    requested_modes: Optional[list[str]],
    output_path: Path,
) -> str:
    output_dir = _resolve(Path(eval_cfg.get("output_dir", "eval_runs")))
    modes_cfg: dict[str, Any] = eval_cfg.get("modes") or {}

    if requested_modes:
        mode_order = [m for m in requested_modes if m in modes_cfg]
    else:
        mode_order = list(modes_cfg.keys())

    # Load all data
    modes_data: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for mode in mode_order:
        cfg = modes_cfg[mode]
        summary, rows = load_mode_data(output_dir, cfg)
        # legacy fallback for agent mode
        if mode == "agent" and not summary:
            summary = _find_legacy_summary(output_dir)
        modes_data[mode] = (summary, rows)

    # Collect all summaries for metadata section
    all_summaries = {m: s for m, (s, _) in modes_data.items()}

    # Build context_dependent lookup from the questions dataset
    dataset_path = eval_cfg.get("dataset_path", "")
    context_dep_map = _load_context_dependent_map(dataset_path) if dataset_path else {}

    # Build sections
    accuracy_mode_order = [m for m in ACCURACY_MODES if m in mode_order]
    conv_summary, conv_rows = modes_data.get("conversation", ({}, []))
    agent_rows = modes_data.get("agent", ({}, []))[1]

    parts = [
        "# Evaluation report",
        "",
        section_metadata(eval_cfg, all_summaries),
        section_coverage(modes_data, mode_order),
        section_accuracy_table(modes_data, accuracy_mode_order),
        section_context_breakdown(modes_data, accuracy_mode_order, context_dep_map),
        section_agent_traces(agent_rows),
        section_conversation(conv_summary, conv_rows),
    ]

    report = "\n".join(parts)

    output_path = _resolve(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Report written to {output_path}")
    return report
