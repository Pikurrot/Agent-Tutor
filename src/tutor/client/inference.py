from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field
from typing import Generator

import httpx
from PIL import Image


def _decode_slides_payload(slides_raw: list) -> list[dict]:
    slides_out: list[dict] = []
    for s in slides_raw:
        raw = base64.b64decode(s["image_b64"])
        img = Image.open(io.BytesIO(raw))
        slides_out.append({"image": img, "caption": s["caption"]})
    return slides_out


def complete(
    base_url: str,
    model_path: str,
    mode: str,
    prompt: str,
    *,
    timeout: float = 600.0,
) -> tuple[str, list[dict]]:
    """
    Call POST /v1/complete on the inference server.

    Returns (assistant_text, slides) where each slide is
    {"image": PIL.Image, "caption": str} for render_slide_gallery.
    """
    url = base_url.rstrip("/") + "/v1/complete"
    payload = {"model_path": model_path, "mode": mode, "prompt": prompt}
    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

    text = data["text"]
    slides_out = _decode_slides_payload(data.get("slides") or [])
    return text, slides_out


@dataclass
class StreamingOutcome:
    text: str = ""
    slides: list[dict] = field(default_factory=list)


def iter_streaming_complete(
    base_url: str,
    model_path: str,
    mode: str,
    prompt: str,
    outcome: StreamingOutcome,
    *,
    timeout: float = 600.0,
) -> Generator[str, None, None]:
    """
    POST /v1/complete/stream (NDJSON). Yields text chunks; fills ``outcome`` on success.
    """
    url = base_url.rstrip("/") + "/v1/complete/stream"
    payload = {"model_path": model_path, "mode": mode, "prompt": prompt}
    parts: list[str] = []
    with httpx.Client(timeout=timeout) as client:
        with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                line = line.strip()
                if not line:
                    continue
                msg = json.loads(line)
                t = msg.get("t")
                if t == "tok":
                    chunk = msg["d"]
                    parts.append(chunk)
                    yield chunk
                elif t == "end":
                    outcome.slides = _decode_slides_payload(msg.get("slides") or [])
                    outcome.text = "".join(parts)
                elif t == "err":
                    raise RuntimeError(msg.get("d", "inference error"))
                else:
                    raise RuntimeError(f"Unknown stream message: {msg}")
    if not outcome.text:
        outcome.text = "".join(parts)
