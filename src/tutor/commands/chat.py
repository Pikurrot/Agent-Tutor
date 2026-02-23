from __future__ import annotations
import argparse
from pathlib import Path

from tutor.utils.config import load_config
from tutor.utils.paths import DEFAULT_CONFIG_PATH
from tutor.core.chat import send_message


def add_chat_args(p: argparse.ArgumentParser):
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH
    )

def run_chat(args: argparse.Namespace):
    cfg = load_config(args.config)
    send_message(cfg)
