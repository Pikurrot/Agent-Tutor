from __future__ import annotations

import argparse
import dotenv
from pathlib import Path

from tutor.core.conversation_eval import run_conversation_evaluation
from tutor.core.evaluation import run_evaluation
from tutor.utils.config import load_config
from tutor.utils.paths import DEFAULT_CONFIG_PATH, PROJECT_ROOT

dotenv.load_dotenv()

DEFAULT_EVAL_CONFIG_PATH = PROJECT_ROOT / "configs" / "eval.yaml"


def add_eval_args(p: argparse.ArgumentParser):
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Tutor config directory (model, RAG, agent settings)",
    )
    p.add_argument(
        "--eval-config",
        type=Path,
        default=DEFAULT_EVAL_CONFIG_PATH,
        help="Evaluation YAML config",
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Override dataset_path from eval config",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output_dir from eval config",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of dataset items to evaluate",
    )
    p.add_argument(
        "--fresh",
        action="store_true",
        help="Truncate results and start evaluation from scratch (ignores resume)",
    )
    p.add_argument(
        "--mode",
        choices=["agent", "llm_context", "llm_baseline", "conversation", "rag"],
        default="agent",
        help=(
            "Evaluation backend: agent (ReAct+RAG, default), "
            "rag (single retrieve-then-generate pass), "
            "llm_context (Qwen with all lecture transcripts), "
            "llm_baseline (Qwen question-only), "
            "conversation (Gemini student talks to the Socratic tutor)"
        ),
    )


def run_eval(args: argparse.Namespace):
    load_config(args.config)
    eval_cfg = load_config(args.eval_config)

    if args.dataset is not None:
        eval_cfg["dataset_path"] = str(args.dataset)
    if args.output_dir is not None:
        eval_cfg["output_dir"] = str(args.output_dir)

    if args.mode == "conversation":
        run_conversation_evaluation(eval_cfg, limit=args.limit, fresh=args.fresh)
    else:
        run_evaluation(eval_cfg, mode=args.mode, limit=args.limit, fresh=args.fresh)
