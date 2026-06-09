from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from tutor.modules.agent.pedagogy import TeachingSession
from tutor.ui.common import decode_slides_from_storage, encode_slides_for_storage
from tutor.utils.paths import SAVE_ROOT

DEFAULT_CONVERSATIONS_DIR = SAVE_ROOT / "student_conversations"
DEFAULT_TITLE = "New conversation"
TITLE_MAX_LEN = 48


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _conversation_path(conversations_dir: Path, conversation_id: str) -> Path:
    return conversations_dir / f"{conversation_id}.json"


def _empty_memory_dict() -> dict[str, Any]:
    return {"summary": "", "last_interaction": None}


def _empty_teaching_session_dict() -> dict[str, Any]:
    return TeachingSession.empty().to_dict()


def title_from_first_user_message(messages: list[dict[str, Any]]) -> str:
    for msg in messages:
        if msg.get("role") == "user":
            text = str(msg.get("content", "")).strip()
            if not text:
                continue
            if len(text) <= TITLE_MAX_LEN:
                return text
            return text[: TITLE_MAX_LEN - 1].rstrip() + "…"
    return DEFAULT_TITLE


@dataclass
class ConversationRecord:
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    conversation_memory: dict[str, Any] = field(default_factory=_empty_memory_dict)
    teaching_session: dict[str, Any] = field(default_factory=_empty_teaching_session_dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": self.messages,
            "conversation_memory": self.conversation_memory,
            "teaching_session": self.teaching_session,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationRecord":
        return cls(
            id=str(data["id"]),
            title=str(data.get("title") or DEFAULT_TITLE),
            created_at=str(data.get("created_at") or _now_iso()),
            updated_at=str(data.get("updated_at") or _now_iso()),
            messages=list(data.get("messages") or []),
            conversation_memory=dict(data.get("conversation_memory") or _empty_memory_dict()),
            teaching_session=dict(
                data.get("teaching_session") or _empty_teaching_session_dict()
            ),
        )


@dataclass
class ConversationSummary:
    id: str
    title: str
    updated_at: str


class StudentConversationStore:
    def __init__(self, conversations_dir: Path | None = None) -> None:
        self.conversations_dir = conversations_dir or DEFAULT_CONVERSATIONS_DIR
        self.conversations_dir.mkdir(parents=True, exist_ok=True)

    def list_summaries(self) -> list[ConversationSummary]:
        summaries: list[ConversationSummary] = []
        for path in self.conversations_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            summaries.append(
                ConversationSummary(
                    id=str(data.get("id", path.stem)),
                    title=str(data.get("title") or DEFAULT_TITLE),
                    updated_at=str(data.get("updated_at") or ""),
                )
            )
        summaries.sort(key=lambda s: s.updated_at, reverse=True)
        return summaries

    def create(self) -> ConversationRecord:
        now = _now_iso()
        record = ConversationRecord(
            id=str(uuid.uuid4()),
            title=DEFAULT_TITLE,
            created_at=now,
            updated_at=now,
        )
        self.save(record)
        return record

    def load(self, conversation_id: str) -> ConversationRecord:
        path = _conversation_path(self.conversations_dir, conversation_id)
        if not path.exists():
            raise FileNotFoundError(f"Conversation not found: {conversation_id}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return ConversationRecord.from_dict(data)

    def save(self, record: ConversationRecord) -> None:
        record.updated_at = _now_iso()
        if record.title == DEFAULT_TITLE:
            record.title = title_from_first_user_message(record.messages)
        path = _conversation_path(self.conversations_dir, record.id)
        tmp = path.with_suffix(".json.tmp")
        payload = json.dumps(record.to_dict(), ensure_ascii=False, indent=2)
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(path)

    def delete(self, conversation_id: str) -> None:
        path = _conversation_path(self.conversations_dir, conversation_id)
        if path.exists():
            path.unlink()


def messages_for_storage(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip PIL images from assistant messages; store slides as b64."""
    stored: list[dict[str, Any]] = []
    for msg in messages:
        entry = {"role": msg["role"], "content": msg.get("content", "")}
        if msg.get("role") == "assistant" and msg.get("slides"):
            slides = msg["slides"]
            if slides and isinstance(slides[0], dict) and "image_b64" in slides[0]:
                entry["slides"] = slides
            else:
                entry["slides"] = encode_slides_for_storage(slides)
        stored.append(entry)
    return stored


def messages_for_display(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Restore PIL images for Streamlit rendering."""
    displayed: list[dict[str, Any]] = []
    for msg in messages:
        entry = dict(msg)
        if msg.get("role") == "assistant" and msg.get("slides"):
            entry["slides"] = decode_slides_from_storage(msg["slides"])
        displayed.append(entry)
    return displayed


def record_from_session(
    conversation_id: str,
    *,
    title: str,
    created_at: str,
    messages: list[dict[str, Any]],
    conversation_memory: dict[str, Any],
    teaching_session: dict[str, Any],
) -> ConversationRecord:
    return ConversationRecord(
        id=conversation_id,
        title=title,
        created_at=created_at,
        updated_at=_now_iso(),
        messages=messages_for_storage(messages),
        conversation_memory=conversation_memory,
        teaching_session=teaching_session,
    )


def apply_record_to_session(record: ConversationRecord) -> dict[str, Any]:
    """Return session-state fields loaded from a conversation record."""
    return {
        "active_conversation_id": record.id,
        "conversation_title": record.title,
        "conversation_created_at": record.created_at,
        "messages": messages_for_display(record.messages),
        "conversation_memory": record.conversation_memory,
        "teaching_session": record.teaching_session,
    }
