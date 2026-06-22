from __future__ import annotations

import argparse
import dotenv
from pathlib import Path

from tutor.utils.config import load_config
from tutor.utils.paths import PROJECT_ROOT

dotenv.load_dotenv()

DEFAULT_EVAL_CONFIG_PATH = PROJECT_ROOT / "configs" / "eval.yaml"
DEFAULT_OUTPUT_PATH = Path("eval_runs/report.md")


def add_eval_report_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--eval-config",
        type=Path,
        default=DEFAULT_EVAL_CONFIG_PATH,
        help="Evaluation YAML config (default: configs/eval.yaml)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output Markdown file (default: eval_runs/report.md)",
    )
    p.add_argument(
        "--modes",
        type=str,
        default=None,
        metavar="MODE1,MODE2,...",
        help=(
            "Comma-separated list of modes to include "
            "(default: all modes defined in eval config). "
            "Available: agent, rag, llm_context, llm_baseline, conversation"
        ),
    )


def run_eval_report(args: argparse.Namespace) -> None:
    from tutor.core.eval_report import generate_report

    eval_cfg = load_config(args.eval_config)

    requested_modes = None
    if args.modes:
        requested_modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    generate_report(eval_cfg, requested_modes, args.output)
