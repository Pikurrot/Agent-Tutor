from __future__ import annotations

import time
from typing import Generator


def get_agent_mock_stream_config(cfg: dict) -> tuple[float, int]:
    """Delay (seconds) and chunk size (Unicode chars) for simulated agent streaming."""
    ac = cfg.get("agent_config") or {}
    delay = float(ac.get("mock_stream_delay_seconds", 0.02))
    chunk = max(1, int(ac.get("mock_stream_chunk_chars", 1)))
    return delay, chunk


def mock_stream_text(
    text: str,
    delay_seconds: float,
    chunk_chars: int = 1,
) -> Generator[str, None, None]:
    """Yield successive slices of `text` after a fixed delay (agent mode stand-in for token stream)."""
    if chunk_chars < 1:
        chunk_chars = 1
    for i in range(0, len(text), chunk_chars):
        if delay_seconds > 0:
            time.sleep(delay_seconds)
        yield text[i : i + chunk_chars]
