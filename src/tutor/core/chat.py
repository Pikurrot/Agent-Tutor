from __future__ import annotations
from pathlib import Path
from typing import Optional

from tutor.utils.misc import get_model
from tutor.utils.paths import MODELS_CACHE_DIR


def send_message(
    cfg: dict,
    msg: str,
    pdf_path: Optional[Path] = None
):
    model, model_type = get_model(cfg["model_path"], MODELS_CACHE_DIR, cfg)
    model.eval()

    prompts = [msg]
    pdfs = [[pdf_path]] if pdf_path is not None else None
    print(f"Generating answer with {model_type}...")
    _, pred_answers, _ = model(prompts, pdfs=pdfs, return_pred_answer=True)
    print(f"Answer: {pred_answers[0]}")


def free_chat(cfg: dict):
    pass
