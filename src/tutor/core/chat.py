from __future__ import annotations
from pathlib import Path
from typing import Optional, Generator

from tutor.utils.misc import get_model
from tutor.utils.paths import MODELS_CACHE_DIR


def cli_send_message(
    cfg: dict,
    msg: str,
    pdf_path: Optional[Path] = None
):
    model, model_type = get_model(cfg["model_path"], MODELS_CACHE_DIR, cfg)
    model.eval()

    prompts = [msg]
    pdfs = [[pdf_path]] if pdf_path is not None else None
    print(f"Generating answer with {model_type}...")
    if model_type == "gemini":
        _, pred_answers, _ = model(prompts, pdfs=pdfs, return_pred_answer=True)
    else:
        if pdf_path is not None:
            print(f"PFF support not enabled for {model_type}")
        _, pred_answers, _ = model(prompts, return_pred_answer=True)
    print(f"Answer: {pred_answers[0]}")


def generate_response(model, msg: str, images: Optional[list] = None) -> str:
    _, pred_answers, _ = model([msg], images=[images] if images else None, return_pred_answer=True)
    return pred_answers[0]


def stream_generate_response(model, msg: str, images: Optional[list] = None) -> Generator[str, None, None]:
    yield from model.stream_generate(msg, images=images)
