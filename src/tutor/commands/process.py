from __future__ import annotations
import argparse
from pathlib import Path

from tutor.utils.config import load_config
from tutor.utils.paths import DEFAULT_CONFIG_PATH


def add_process_args(p: argparse.ArgumentParser):
    # Config
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH
    )
    # Path
    p.add_argument(
        "--path",
        type=Path,
        required=True,
        help="Path to the file or directory to process"
    )
    # Output path
    p.add_argument(
        "--output_path",
        type=Path,
        required=False,
        default=None,
        help="Path to the output directory. If not provided, the output will be saved in the same directory as the input file."
    )

def run_process(args: argparse.Namespace):
    cfg = load_config(args.config)

