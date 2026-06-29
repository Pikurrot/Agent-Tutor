from __future__ import annotations

import argparse
from pathlib import Path

import dotenv

from tutor.utils.config import load_config
from tutor.utils.paths import PROJECT_ROOT

dotenv.load_dotenv()

DEFAULT_EVAL_CONFIG_PATH = PROJECT_ROOT / "configs" / "eval.yaml"


def add_reclassify_context_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--eval-config",
        type=Path,
        default=DEFAULT_EVAL_CONFIG_PATH,
        help="Evaluation YAML config used to resolve the default dataset path "
             "(default: configs/eval.yaml)",
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to the Q&A dataset JSON file. Overrides the path in eval config.",
    )
    p.add_argument(
        "--model",
        type=str,
        default="gpt-5.4-mini",
        help="OpenAI model to use for classification (default: gpt-5.4-mini)",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of Q&A pairs per LLM call (default: 20)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print proposed changes without writing to disk",
    )
    p.add_argument(
        "--ids",
        type=str,
        default=None,
        metavar="ID1,ID2,...",
        help="Comma-separated list of question IDs to reclassify (default: all)",
    )


def run_reclassify_context(args: argparse.Namespace) -> None:
    from tutor.core.reclassify_context import run_reclassify_context as _run

    eval_cfg = load_config(args.eval_config)

    dataset_path: Path
    if args.dataset:
        dataset_path = args.dataset
    else:
        raw = eval_cfg.get("dataset_path")
        if not raw:
            raise ValueError(
                "No dataset path provided. Use --dataset or set dataset_path in eval config."
            )
        dataset_path = Path(raw)

    ids_filter = None
    if args.ids:
        ids_filter = [i.strip() for i in args.ids.split(",") if i.strip()]

    _run(
        dataset_path=dataset_path,
        model=args.model,
        temperature=args.temperature,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        ids_filter=ids_filter,
    )
