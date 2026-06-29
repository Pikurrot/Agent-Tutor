from __future__ import annotations
import base64
import io
import os
from typing import Optional, Tuple, Generator

from openai import OpenAI
from PIL import Image

from tutor.modules.models.base import BaseModel


def _image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_input_content(
    prompt: Optional[str] = None,
    images: Optional[list[Image.Image]] = None,
    pdfs: Optional[list] = None,
) -> list:
    content = []
    if pdfs:
        print(
            "Warning: OpenAI Responses API does not support raw PDF byte inputs. "
            "PDFs will be skipped."
        )
    if images:
        for img in images:
            b64 = _image_to_base64(img)
            content.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{b64}",
            })
    if prompt is not None:
        content.append({"type": "input_text", "text": prompt})
    return content


class OpenAIModel(BaseModel):
    def __init__(self, config: dict):
        self.config = config
        self.my_config = config.get("openai_config", {})
        self.model_path = config.get("model_path", None)
        if self.model_path is None or not self.model_path.startswith("gpt-"):
            self.model_path = self.my_config.get("model", "gpt-5.4-mini")
        self.temperature = self.my_config.get("temperature", 0.0)
        self.system_instruction = self.my_config.get("system_instruction", None)

        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)

    def generate(
        self,
        prompt: str,
        images: Optional[list] = None,
        pdfs: Optional[list] = None,
        **kwargs,
    ) -> str:
        content = _build_input_content(prompt, images, pdfs)
        kwargs_extra = {}
        if self.system_instruction:
            kwargs_extra["instructions"] = self.system_instruction
        response = self.client.responses.create(
            model=self.model_path,
            temperature=self.temperature,
            input=[{"role": "user", "content": content}],
            **kwargs_extra,
        )
        return response.output_text

    def stream_generate(
        self,
        prompt: str,
        images: Optional[list] = None,
        pdfs: Optional[list] = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        content = _build_input_content(prompt, images, pdfs)
        kwargs_extra = {}
        if self.system_instruction:
            kwargs_extra["instructions"] = self.system_instruction
        stream = self.client.responses.create(
            model=self.model_path,
            temperature=self.temperature,
            input=[{"role": "user", "content": content}],
            stream=True,
            **kwargs_extra,
        )
        for event in stream:
            delta = getattr(event, "delta", None)
            if delta:
                yield delta

    def __call__(
        self,
        prompts: list,
        images: Optional[list] = None,
        pdfs: Optional[list] = None,
        return_pred_answer: bool = False,
    ) -> Tuple[list, list, list]:
        if images is None:
            images = [None] * len(prompts)
        if pdfs is None:
            pdfs = [None] * len(prompts)
        pred_answers = [
            self.generate(prompt, imgs, pdf_list)
            for prompt, imgs, pdf_list in zip(prompts, images, pdfs)
        ]
        return None, pred_answers, None


def openai_generate_answer(
    client: OpenAI,
    prompt: Optional[str] = None,
    images: Optional[list[Image.Image]] = None,
    pdfs: Optional[list] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    system_instruction: Optional[str] = None,
) -> str:
    """Generate an answer for a single prompt using the OpenAI Responses API."""
    if model is None:
        model = "gpt-5.4-mini"
    if temperature is None:
        temperature = 0.0

    content = _build_input_content(prompt, images, pdfs)
    kwargs_extra = {}
    if system_instruction:
        kwargs_extra["instructions"] = system_instruction

    response = client.responses.create(
        model=model,
        temperature=temperature,
        input=[{"role": "user", "content": content}],
        **kwargs_extra,
    )
    return response.output_text
