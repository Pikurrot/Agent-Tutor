from __future__ import annotations
from pathlib import Path
from tutor.modules.processor.processor import Processor


def process_videos(
    cfg: dict,
    video_paths: list[str],
    pdf_path: str
) -> None:
    processor = Processor(cfg)
    processor.process_videos(video_paths, pdf_path)


def process_subject_directory(
    cfg: dict,
    dir_path: str
) -> None:
    subject_name = Path(dir_path).stem
    slides2videos = cfg["slides2videos"][subject_name]
    processor = Processor(cfg)
    pdfs_dir = Path(dir_path) / "raw" / "slides"
    videos_dir = Path(dir_path) / "raw" / "videos"
    
    for i, (pdf_name, video_names) in enumerate(slides2videos.items()):
        print(f"Processing {i+1} of {len(slides2videos)}: {pdf_name}...")
        pdf_path = pdfs_dir / pdf_name
        video_paths = [videos_dir / video_name for video_name in video_names]
        processor.process_videos(video_paths, pdf_path)
