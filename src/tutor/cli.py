from __future__ import annotations
import os
import sys
import argparse
import dotenv

from tutor.utils.config import load_config
from tutor.utils.paths import DEFAULT_CONFIG_PATH

cfg = load_config(DEFAULT_CONFIG_PATH)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = cfg["visible_devices"]
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from tutor.commands.chat import add_chat_args, run_chat  # noqa: E402
from tutor.commands.app import add_app_args, run_app  # noqa: E402
from tutor.commands.public import add_public_args, run_public  # noqa: E402
from tutor.commands.serve import add_serve_args, run_serve  # noqa: E402
from tutor.utils.misc import seed_everything # noqa: E402
from tutor.commands.process import add_process_args, run_process # noqa: E402
from tutor.commands.eval import add_eval_args, run_eval  # noqa: E402
from tutor.commands.generate_qa import add_generate_qa_args, run_generate_qa  # noqa: E402
from tutor.commands.eval_report import add_eval_report_args, run_eval_report  # noqa: E402
from tutor.commands.reclassify_context import add_reclassify_context_args, run_reclassify_context  # noqa: E402


dotenv.load_dotenv()
seed_everything(42)

def build_parser():
    p = argparse.ArgumentParser(prog="tutor", description="Tutor CLI tool")
    sub = p.add_subparsers(dest="command", help="Subcommand to run")

    p_chat = sub.add_parser("chat", help="Run chat")
    add_chat_args(p_chat)

    p_process = sub.add_parser("process", help="Run process")
    add_process_args(p_process)

    p_app = sub.add_parser("app", help="Launch Streamlit chat GUI")
    add_app_args(p_app)

    p_public = sub.add_parser("public", help="Launch student-facing tutor GUI")
    add_public_args(p_public)

    p_serve = sub.add_parser("serve", help="Run inference HTTP API (model + RAG in this process)")
    add_serve_args(p_serve)

    p_eval = sub.add_parser(
        "eval",
        help="Run evaluation on a Q&A dataset (agent, llm_context, or llm_baseline)",
    )
    add_eval_args(p_eval)

    p_gqa = sub.add_parser(
        "generate-qa",
        help="Generate Q&A dataset from lecture transcripts using Gemini",
    )
    add_generate_qa_args(p_gqa)

    p_report = sub.add_parser(
        "eval-report",
        help="Generate Markdown report from evaluation results",
    )
    add_eval_report_args(p_report)

    p_reclass = sub.add_parser(
        "reclassify-context",
        help="Re-evaluate context_dependent labels in the Q&A dataset using an LLM",
    )
    add_reclassify_context_args(p_reclass)

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
    elif args.command == "process":
        run_process(args)
    elif args.command == "app":
        run_app(args)
    elif args.command == "public":
        run_public(args)
    elif args.command == "serve":
        run_serve(args)
    elif args.command == "eval":
        run_eval(args)
    elif args.command == "generate-qa":
        run_generate_qa(args)
    elif args.command == "eval-report":
        run_eval_report(args)
    elif args.command == "reclassify-context":
        run_reclassify_context(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
