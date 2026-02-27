from __future__ import annotations
import argparse
from pathlib import Path

from tutor.utils.config import load_config
from tutor.utils.paths import DEFAULT_CONFIG_PATH
from tutor.core.chat import cli_send_message


def add_chat_args(p: argparse.ArgumentParser):
    # Config
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH
    )
    # Message
    p.add_argument(
        "--msg",
        type=str,
        required=True,
        help="Message to send to the model"
    )
    # PDF path
    p.add_argument(
        "--pdf_path",
        type=Path,
        required=False,
        default=None,
        help="Path to the PDF file to send to the model"
    )

def run_chat(args: argparse.Namespace):
    cfg = load_config(args.config)
    cli_send_message(cfg, args.msg, args.pdf_path)
