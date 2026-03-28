from __future__ import annotations
from tutor.modules.processor.processor import Processor


def process_videos(
    cfg: dict,
    video_paths: list[str],
    pdf_path: str
) -> None:
    processor = Processor(cfg)
    processor.process_videos(video_paths, pdf_path)
