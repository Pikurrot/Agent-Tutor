from __future__ import annotations

import argparse
import dotenv
from pathlib import Path

from tutor.utils.config import load_config
from tutor.utils.paths import DEFAULT_CONFIG_PATH, PROJECT_ROOT

dotenv.load_dotenv()

DEFAULT_EVAL_CONFIG_PATH = PROJECT_ROOT / "configs" / "eval.yaml"


def add_generate_qa_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Tutor config directory (model, retrieval, agent settings)",
    )
    p.add_argument(
        "--eval-config",
        type=Path,
        default=DEFAULT_EVAL_CONFIG_PATH,
        help="Evaluation YAML config (used to read dataset_path)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file (default: dataset_path from eval config)",
    )
    p.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of Q&A pairs to generate per lecture (default: 5)",
    )
    p.add_argument(
        "--lecture",
        type=str,
        default=None,
        metavar="NAME",
        help="Only generate for lectures whose name contains this string (case-insensitive)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Truncate the output file before generating (default: append)",
    )
    p.add_argument(
        "--backfill-only",
        action="store_true",
        help="Only run backfill of context_dependent; skip generation",
    )
    p.add_argument(
        "--no-backfill",
        action="store_true",
        help="Skip backfill even if existing samples are missing context_dependent",
    )
    p.add_argument(
        "-m",
        "--backfill-batch-size",
        type=int,
        default=20,
        metavar="M",
        help="Number of samples per backfill batch (default: 20)",
    )
    p.add_argument(
        "--model",
        type=str,
        default="gpt-5.4-mini",
        help="OpenAI model to use for generation and backfill (default: gpt-5.4-mini)",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Sampling temperature (default: 0.5)",
    )


def run_generate_qa(args: argparse.Namespace) -> None:
    from tutor.core.qa_generation import run_qa_generation

    tutor_cfg = load_config(args.config)
    eval_cfg = load_config(args.eval_config)

    if args.output is not None:
        output_path = args.output
    else:
        dataset_path = eval_cfg.get("dataset_path")
        if not dataset_path:
            raise ValueError(
                "No --output given and eval config has no dataset_path. "
                "Specify --output explicitly."
            )
        output_path = Path(dataset_path)

    run_qa_generation(
        tutor_cfg,
        eval_cfg,
        output_path=output_path,
        k=args.k,
        backfill_batch_size=args.backfill_batch_size,
        lecture_filter=args.lecture,
        overwrite=args.overwrite,
        backfill_only=args.backfill_only,
        no_backfill=args.no_backfill,
        model=args.model,
        temperature=args.temperature,
    )
