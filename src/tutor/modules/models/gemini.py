from __future__ import annotations
import os
import pathlib
import numpy as np
from PIL import Image
from google import genai
from google.genai import types
from typing import Optional, Tuple, Any, Generator

from tutor.modules.models.base import BaseModel


def _build_contents(
    prompt: Optional[str] = None,
    images: Optional[list[Image.Image]] = None,
    pdfs: Optional[list] = None,
) -> list:
    contents = []
    if pdfs is not None:
        for pdf_path in pdfs:
            pdf_bytes = pathlib.Path(pdf_path).read_bytes()
            contents.append(types.Part.from_bytes(
                data=pdf_bytes,
                mime_type="application/pdf"
            ))
    if images is not None:
        contents.extend(images)
    if prompt is not None:
        contents.append(prompt)
    return contents


def _build_config(
    model: str,
    thinking_budget: int = 0,
    temperature: float = 0.0,
    system_instruction: Optional[str] = None,
) -> types.GenerateContentConfig:
    if model.startswith("gemini-2.5"):
        return types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
            system_instruction=system_instruction,
            temperature=temperature
        )
    return types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature
    )


class GeminiModel(BaseModel):
    def __init__(self, config: dict):
        self.config = config
        self.my_config = config.get("gemini_config", {})
        self.model_path = config.get("model_path", None)
        if self.model_path is None or not self.model_path.startswith("gemini"):
            self.model_path = self.my_config.get("model", "gemini-2.5-flash")
        self.thinking_budget = self.my_config.get("thinking_budget", 0)
        self.temperature = self.my_config.get("temperature", 0.0)
        self.system_instruction = self.my_config.get("system_instruction", None)

        self.api_keys = os.getenv("GEMINI_API_KEYS").split(",")
        self.current_api_key_idx = 0
        self.api_keys_accumulated_errors = np.zeros(len(self.api_keys))
        self.client = self.create_client()

    def get_current_api_key(self):
        return self.api_keys[self.current_api_key_idx]

    def increment_api_key_idx(self):
        self.current_api_key_idx = (self.current_api_key_idx + 1) % len(self.api_keys)
        return self.current_api_key_idx == 0

    def create_client(self):
        try:
            return genai.Client(api_key=self.get_current_api_key())
        except Exception as e:
            print(f"Warning: no Gemini client initialized: {e}")
            return None

    def _rotate_key_on_error(self, e: Exception):
        print(f"Error: {e}")
        self.api_keys_accumulated_errors[self.current_api_key_idx] += 1
        print("Trying next API key...")
        self.increment_api_key_idx()
        if not any(self.api_keys_accumulated_errors < 2):
            raise Exception("All Gemini API keys have been tried.")
        self.client = self.create_client()

    def generate(
        self,
        prompt: str,
        images: Optional[list] = None,
        pdfs: Optional[list] = None,
        **kwargs,
    ) -> str:
        try:
            contents = _build_contents(prompt, images, pdfs)
            config = _build_config(
                self.model_path, self.thinking_budget,
                self.temperature, self.system_instruction,
            )
            response = self.client.models.generate_content(
                model=self.model_path, contents=contents, config=config,
            )
            self.api_keys_accumulated_errors[self.current_api_key_idx] = 0
            return response.text
        except Exception as e:
            self._rotate_key_on_error(e)
            return self.generate(prompt, images, pdfs)

    def stream_generate(
        self,
        prompt: str,
        images: Optional[list] = None,
        pdfs: Optional[list] = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        try:
            contents = _build_contents(prompt, images, pdfs)
            config = _build_config(
                self.model_path, self.thinking_budget,
                self.temperature, self.system_instruction,
            )
            for chunk in self.client.models.generate_content_stream(
                model=self.model_path, contents=contents, config=config,
            ):
                if chunk.text:
                    yield chunk.text
            self.api_keys_accumulated_errors[self.current_api_key_idx] = 0
        except Exception as e:
            self._rotate_key_on_error(e)
            yield from self.stream_generate(prompt, images, pdfs)

    def __call__(
            self,
            prompts: list,
            images: Optional[list] = None,
            pdfs: Optional[list] = None,
            return_pred_answer: bool = False
    ) -> Tuple[list, list, list]:
        if images is None:
            images = [None]*len(prompts)
        if pdfs is None:
            pdfs = [None]*len(prompts)
        pred_answers = [
            self.generate(prompt, imgs, pdf_list)
            for prompt, imgs, pdf_list in zip(prompts, images, pdfs)
        ]
        outputs = None
        pred_answers_conf = None
        return outputs, pred_answers, pred_answers_conf


def gemini_generate_answer(
    client: genai.Client,
    prompt: Optional[str] = None,
    images: Optional[list[Image.Image]] = None,
    pdfs: Optional[list] = None,
    contents: Optional[list[Any]] = None,
    model: Optional[str] = None,
    thinking_budget: Optional[int] = None,
    temperature: Optional[float] = None,
    system_instruction: Optional[str] = None
) -> str:
    """
    Generate an answer for a single prompt using Gemini.
    :param prompt: prompt
    :param images: list of PIL images
    :param pdfs: list of PDF file paths (str or pathlib.Path)
    :param contents: list of content parts. Overrides prompt, images, and pdfs if provided.
    :param model: model to use (gemini-2.0-flash, gemini-2.5-flash, ...)
    :param thinking_budget: budget for thinking (0 to disable, -1 for dynamic)
    :param temperature: temperature for sampling (0.0 to 2.0)
    :param system_instruction: system instruction
    :return: answer
    """
    if model is None:
        model = "gemini-2.0-flash"
    if thinking_budget is None:
        thinking_budget = 0
    if temperature is None:
        temperature = 0.0

    if contents is None:
        contents = _build_contents(prompt, images, pdfs)

    config = _build_config(model, thinking_budget, temperature, system_instruction)
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config
    )

    return response.text

# TODO: for better batch generation, see: https://developers.googleblog.com/en/scale-your-ai-workloads-batch-mode-gemini-api/
def gemini_generate_answers(
    client: genai.Client,
    prompts: Optional[list[str]] = None,
    images: Optional[list[list[Image.Image]]] = None,
    contents: Optional[list[list[Any]]] = None,
    model: Optional[str] = "gemini-2.0-flash",
    thinking_budget: Optional[int] = 0,
    temperature: Optional[float] = 0.0,
    system_instruction: Optional[str] = None
) -> list[str]:
    """
    :param prompts: list of prompts
    :param images: list of lists of images
    :param contents: list of lists of content (text, images, videos, etc.)
    :param model: model to use (gemini-2.0-flash, gemini-2.5-flash, ...)
    :param thinking_budget: budget for thinking (0 to disable, -1 for dynamic)
    :return: list of answers
    """
    answers = []
    for i in range(len(prompts)):
        prompt = prompts[i]
        if images is not None and len(images) > i:
            image_lst = images[i]
        else:
            image_lst = []
        contents = contents[i] if contents is not None else None
        answer = gemini_generate_answer(prompt, image_lst, contents, model, thinking_budget, temperature, system_instruction)
        answers.append(answer)
    return answers
