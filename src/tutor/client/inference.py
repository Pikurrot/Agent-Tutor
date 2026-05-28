from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field
from typing import Generator, Optional

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
    memory: Optional[dict] = None,
    teaching_session: Optional[dict] = None,
    debug: bool = False,
    timeout: float = 600.0,
) -> tuple[str, list[dict], Optional[dict], Optional[dict], Optional[dict]]:
    """
    Call POST /v1/complete on the inference server.

    Returns ``(assistant_text, slides, updated_memory, updated_teaching_session)``
    where each slide is ``{"image": PIL.Image, "caption": str}`` and the last
    two fields are ``None`` for modes that don't use them.
    """
    url = base_url.rstrip("/") + "/v1/complete"
    payload: dict = {"model_path": model_path, "mode": mode, "prompt": prompt}
    if memory is not None:
        payload["memory"] = memory
    if teaching_session is not None:
        payload["teaching_session"] = teaching_session
    payload["debug"] = bool(debug)
    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

    text = data["text"]
    slides_out = _decode_slides_payload(data.get("slides") or [])
    return (
        text,
        slides_out,
        data.get("memory"),
        data.get("teaching_session"),
        data.get("debug_data"),
    )


@dataclass
class StreamingOutcome:
    text: str = ""
    slides: list[dict] = field(default_factory=list)
    memory: Optional[dict] = None
    teaching_session: Optional[dict] = None
    debug_data: Optional[dict] = None


def iter_streaming_complete(
    base_url: str,
    model_path: str,
    mode: str,
    prompt: str,
    outcome: StreamingOutcome,
    *,
    memory: Optional[dict] = None,
    teaching_session: Optional[dict] = None,
    debug: bool = False,
    timeout: float = 600.0,
) -> Generator[str, None, None]:
    """
    POST /v1/complete/stream (NDJSON). Yields text chunks; fills ``outcome`` on success.
    """
    url = base_url.rstrip("/") + "/v1/complete/stream"
    payload: dict = {"model_path": model_path, "mode": mode, "prompt": prompt}
    if memory is not None:
        payload["memory"] = memory
    if teaching_session is not None:
        payload["teaching_session"] = teaching_session
    payload["debug"] = bool(debug)
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
                    outcome.memory = msg.get("memory")
                    outcome.teaching_session = msg.get("teaching_session")
                    outcome.debug_data = msg.get("debug_data")
                    outcome.text = "".join(parts)
                elif t == "err":
                    raise RuntimeError(msg.get("d", "inference error"))
                else:
                    raise RuntimeError(f"Unknown stream message: {msg}")
    if not outcome.text:
        outcome.text = "".join(parts)
