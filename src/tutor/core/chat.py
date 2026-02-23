from __future__ import annotations
import dotenv

from tutor.utils.misc import seed_everything, get_model
from tutor.modules.models.gemini import GeminiModel
from tutor.utils.paths import MODELS_CACHE_DIR


dotenv.load_dotenv()
seed_everything(42)


def send_message(cfg: dict, msg: str):
    model, model_type = get_model(cfg["model_path"], MODELS_CACHE_DIR, cfg)
    model.eval()

    prompts = [msg]
    print(f"Generating answer with {model_type}...")
    _, pred_answers, _ = model(prompts, return_pred_answer=True)
    print(f"Answer: {pred_answers[0]}")


def free_chat(cfg: dict):
    pass
