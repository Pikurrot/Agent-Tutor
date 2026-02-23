from __future__ import annotations
import os
import sys
import argparse

from tutor.utils.config import load_config
from tutor.utils.paths import DEFAULT_CONFIG_PATH

cfg = load_config(DEFAULT_CONFIG_PATH)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = cfg["visible_devices"]
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from tutor.commands.chat import add_chat_args, run_chat  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(prog="tutor", description="Tutor CLI tool")
    sub = p.add_subparsers(dest="command", help="Subcommand to run")

    p_chat = sub.add_parser("chat", help="Run chat")
    add_chat_args(p_chat)

    return p


def main():
    parser = build_parser()

    try:
        import argcomplete
        argcomplete.autocomplete(parser)
    except ImportError:
        pass

    args = parser.parse_args()

    if args.command == "chat":
        run_chat(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
