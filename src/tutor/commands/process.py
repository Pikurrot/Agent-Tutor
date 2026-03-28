from __future__ import annotations
import argparse
from pathlib import Path

from tutor.utils.config import load_config
from tutor.utils.paths import DEFAULT_CONFIG_PATH
from tutor.core.process import process_videos


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
        required=False,
        help="Path to the file or directory to process"
    )
    # Pdf path
    p.add_argument(
        "--pdf_path",
        type=Path,
        required=False,
        help="Path to the PDF corresponding to the video."
    )
    # Video paths
    p.add_argument(
        "--video_paths",
        nargs="+",
        type=str,
        required=False,
        help="List of video paths to process."
    )


def run_process(args: argparse.Namespace):
    cfg = load_config(args.config)
    path = args.path
    pdf_path = args.pdf_path
    video_paths = args.video_paths
    if path is not None and path.is_file():
        if str(path).endswith(".mp4"):
            if pdf_path is None:
                raise ValueError("PDF path is required if the provided path is a video file.")
            process_videos(cfg, [str(path)], str(pdf_path))
        else:
            raise ValueError("Invalid file type.")
    elif video_paths is not None and len(video_paths) > 0:
        if pdf_path is None:
            raise ValueError("PDF path is required if the provided video paths are provided.")
        process_videos(cfg, video_paths, str(pdf_path))
    else:
        raise ValueError("Invalid video paths. Please provide a valid list of video paths.")
    print("File processed successfully.")
