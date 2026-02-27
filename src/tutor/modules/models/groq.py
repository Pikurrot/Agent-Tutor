from __future__ import annotations
import os
import numpy as np
from groq import Groq
from typing import Optional, Tuple, Generator

from tutor.modules.models.base import BaseModel


class GroqModel(BaseModel):
    def __init__(self, config: dict):
        self.config = config
        self.my_config = config.get("groq_config", {})
        self.model_path = config.get("model_path", None)
        if self.model_path is None or not self.model_path.startswith("groq"):
             self.model_path = self.my_config.get("model", "openai/gpt-oss-120b")
        
        if self.model_path.startswith("groq/"):
            self.model_path = self.model_path[len("groq/"):]

        self.temperature = self.my_config.get("temperature", 1.0)
        self.max_completion_tokens = self.my_config.get("max_completion_tokens", 8192)
        self.top_p = self.my_config.get("top_p", 1.0)
        self.reasoning_effort = self.my_config.get("reasoning_effort", "medium")
        self.system_instruction = self.my_config.get("system_instruction", None)

        api_keys_str = os.getenv("GROQ_API_KEYS")
        if not api_keys_str:
            print("Warning: GROQ_API_KEYS environment variable not set. Groq client will not be initialized.")
            self.api_keys = []
        else:
            self.api_keys = api_keys_str.split(",")
        
        self.current_api_key_idx = 0
        self.api_keys_accumulated_errors = np.zeros(len(self.api_keys)) if self.api_keys else np.zeros(0)
        self.client = self.create_client()

    def get_current_api_key(self):
        return self.api_keys[self.current_api_key_idx]

    def increment_api_key_idx(self):
        if not self.api_keys:
            return True
        self.current_api_key_idx = (self.current_api_key_idx + 1) % len(self.api_keys)
        return self.current_api_key_idx == 0 

    def create_client(self):
        if not self.api_keys:
            return None
        try:
            return Groq(api_key=self.get_current_api_key())
        except Exception as e:
            print(f"Warning: no Groq client initialized: {e}")
            return None

    def _build_messages(self, prompt: str) -> list[dict]:
        messages = []
        if self.system_instruction:
            messages.append({"role": "system", "content": self.system_instruction})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _create_completion(self, prompt: str, stream: bool = False):
        messages = self._build_messages(prompt)
        return self.client.chat.completions.create(
            model=self.model_path,
            messages=messages,
            temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens,
            top_p=self.top_p,
            reasoning_effort=self.reasoning_effort,
            stream=stream,
            stop=None
        )

    def _rotate_key_on_error(self, e: Exception):
        print(f"Error: {e}")
        if self.api_keys:
            self.api_keys_accumulated_errors[self.current_api_key_idx] += 1
            print("Trying next API key...")
            self.increment_api_key_idx()
            if not any(self.api_keys_accumulated_errors < 2):
                raise Exception("All Groq API keys have been tried or failed.")
            self.client = self.create_client()
        else:
            raise e

    def generate(self, prompt: str, **kwargs) -> str:
        try:
            completion = self._create_completion(prompt, stream=True)
            full_response = ""
            for chunk in completion:
                full_response += (chunk.choices[0].delta.content or "")
            self.api_keys_accumulated_errors[self.current_api_key_idx] = 0
            return full_response
        except Exception as e:
            self._rotate_key_on_error(e)
            return self.generate(prompt)

    def stream_generate(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        try:
            completion = self._create_completion(prompt, stream=True)
            for chunk in completion:
                content = chunk.choices[0].delta.content or ""
                if content:
                    yield content
            self.api_keys_accumulated_errors[self.current_api_key_idx] = 0
        except Exception as e:
            self._rotate_key_on_error(e)
            yield from self.stream_generate(prompt)

    def __call__(
            self,
            prompts: list,
            images: Optional[list] = None,
            return_pred_answer: bool = False
    ) -> Tuple[list, list, list]:
        if images is None:
            images = [None]*len(prompts)
        pred_answers = [self.generate(prompt) for prompt in prompts]
        outputs = None
        pred_answers_conf = None
        return outputs, pred_answers, pred_answers_conf


def groq_generate_answer(
    client: Groq,
    prompt: str,
    model: str = "openai/gpt-oss-120b",
    temperature: float = 1.0,
    max_completion_tokens: int = 8192,
    top_p: float = 1.0,
    reasoning_effort: str = "medium",
    system_instruction: Optional[str] = None
) -> str:
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        top_p=top_p,
        reasoning_effort=reasoning_effort,
        stream=True,
        stop=None
    )

    full_response = ""
    for chunk in completion:
        full_response += (chunk.choices[0].delta.content or "")
    
    return full_response

def groq_generate_answers(
    client: Groq,
    prompts: list[str],
    model: str = "openai/gpt-oss-120b",
    temperature: float = 1.0,
    max_completion_tokens: int = 8192,
    top_p: float = 1.0,
    reasoning_effort: str = "medium",
    system_instruction: Optional[str] = None
) -> list[str]:
    return [groq_generate_answer(client, p, model, temperature, max_completion_tokens, top_p, reasoning_effort, system_instruction) for p in prompts]
