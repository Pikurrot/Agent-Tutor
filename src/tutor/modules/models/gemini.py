from __future__ import annotations
import os
import numpy as np
from PIL import Image
from google import genai
from google.genai import types
from typing import Optional, Tuple, Any


class GeminiModel:
    def __init__(self, config: dict):
        self.config = config
        self.my_config = config.get("gemini_config", {})
        self.model_path = config.get("model_path", None)
        if self.model_path is None or not self.model_path.startswith("gemini"):
            self.model_path = self.my_config.get("model", "gemini-2.5-flash")
        self.thinking_budget = self.my_config.get("thinking_budget", 0)
        self.temperature = self.my_config.get("temperature", 0.0)
        self.system_instruction = self.my_config.get("system_instruction", None)

        # read list from .env
        self.api_keys = os.getenv("GEMINI_API_KEYS").split(",")
        self.current_api_key_idx = 0
        self.api_keys_accumulated_errors = np.zeros(len(self.api_keys))
        self.client = self.create_client()

    def get_current_api_key(self):
        return self.api_keys[self.current_api_key_idx]

    def increment_api_key_idx(self):
        self.current_api_key_idx = (self.current_api_key_idx + 1) % len(self.api_keys)
        return self.current_api_key_idx == 0 # True if all api keys have been tried

    def create_client(self):
        try:
            return genai.Client(api_key=self.get_current_api_key())
        except Exception as e:
            print(f"Warning: no Gemini client initialized: {e}")
            return None
    
    def generate(
        self,
        prompt: str,
        images: Optional[list] = None
    ) -> str:
        # print("Generating answer...")
        try:
            answer = gemini_generate_answer(
                client=self.client,
                prompt=prompt,
                images=images,
                model=self.model_path,
                thinking_budget=self.thinking_budget,
                temperature=self.temperature,
                system_instruction=self.system_instruction
            )
            self.api_keys_accumulated_errors[self.current_api_key_idx] = 0
            return answer
        except Exception as e:
            print(f"Error: {e}")
            self.api_keys_accumulated_errors[self.current_api_key_idx] += 1
            print("Trying next API key...")
            self.increment_api_key_idx()
            if not any(self.api_keys_accumulated_errors < 2): # 2 errors per API key means that API key has reached its limit for the day
                raise Exception("All Gemini API keys have been tried.")
            self.client = self.create_client()
            return self.generate(prompt, images)

    def __call__(
            self,
            prompts: list, # (bs,)
            images: Optional[list] = None # (bs, k) PIL images
    ) -> Tuple[list, list, list]:
        if images is None:
            images = [None]*len(prompts)
        pred_answers = [self.generate(prompt, image) for prompt, image in zip(prompts, images)]
        outputs = None
        pred_answers_conf = None
        return outputs, pred_answers, pred_answers_conf


def gemini_generate_answer(
    client: genai.Client,
    prompt: Optional[str] = None,
    images: Optional[list[Image.Image]] = None,
    contents: Optional[list[Any]] = None,
    model: Optional[str] = None,
    thinking_budget: Optional[int] = None,
    temperature: Optional[float] = None,
    system_instruction: Optional[str] = None
) -> str:
    """
    Generate an answer for a single prompt using Gemini.
    :param prompt: prompt
    :param images: list of images
    :param contents: list of content (text, images). Overrides prompt and images if provided.
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
        if images is not None:
            contents = [prompt] + images
        else:
            contents = [prompt]

    if model.startswith("gemini-2.5"):
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
            system_instruction=system_instruction,
            temperature=temperature
        )
    else:
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature
        )
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
