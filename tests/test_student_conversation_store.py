from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tutor.core.student_conversation_store import (
    StudentConversationStore,
    messages_for_display,
    messages_for_storage,
    record_from_session,
    title_from_first_user_message,
)
from tutor.modules.agent.pedagogy import TeachingSession
from tutor.ui.common import decode_slides_from_storage, encode_slides_for_storage


class TestStudentConversationStore(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.store = StudentConversationStore(Path(self._tmpdir.name))

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_create_save_load_roundtrip(self) -> None:
        record = self.store.create()
        record.messages = [
            {"role": "user", "content": "What is backpropagation?"},
            {"role": "assistant", "content": "Let's start with the chain rule."},
        ]
        record.conversation_memory = {
            "summary": "Discussing gradients",
            "last_interaction": {"user": "hi", "assistant": "hello"},
        }
        record.teaching_session = TeachingSession.empty().to_dict()
        self.store.save(record)

        loaded = self.store.load(record.id)
        self.assertEqual(loaded.messages, record.messages)
        self.assertEqual(loaded.conversation_memory, record.conversation_memory)
        self.assertEqual(loaded.teaching_session, record.teaching_session)
        self.assertEqual(loaded.title, "What is backpropagation?")

    def test_list_summaries_newest_first(self) -> None:
        first = self.store.create()
        second = self.store.create()
        first.messages = [{"role": "user", "content": "First question"}]
        second.messages = [{"role": "user", "content": "Second question"}]
        self.store.save(first)
        self.store.save(second)

        summaries = self.store.list_summaries()
        self.assertEqual(len(summaries), 2)
        self.assertEqual(summaries[0].id, second.id)
        self.assertEqual(summaries[1].id, first.id)

    def test_delete_removes_file(self) -> None:
        record = self.store.create()
        path = Path(self._tmpdir.name) / f"{record.id}.json"
        self.assertTrue(path.exists())
        self.store.delete(record.id)
        self.assertFalse(path.exists())

    def test_title_from_first_user_message(self) -> None:
        long_q = "x" * 60
        title = title_from_first_user_message([{"role": "user", "content": long_q}])
        self.assertTrue(title.endswith("…"))
        self.assertLessEqual(len(title), 48)

    def test_slide_b64_roundtrip_in_messages(self) -> None:
        img = Image.new("RGB", (8, 8), color=(255, 0, 0))
        slides = [{"image": img, "caption": "lecture · slide 1"}]
        stored = messages_for_storage(
            [{"role": "assistant", "content": "See slide.", "slides": slides}]
        )
        self.assertIn("image_b64", stored[0]["slides"][0])
        displayed = messages_for_display(stored)
        self.assertEqual(displayed[0]["slides"][0]["caption"], "lecture · slide 1")
        self.assertIsInstance(displayed[0]["slides"][0]["image"], Image.Image)

    def test_encode_decode_slides_helpers(self) -> None:
        img = Image.new("RGB", (4, 4), color=(0, 128, 255))
        encoded = encode_slides_for_storage([{"image": img, "caption": "doc · slide 2"}])
        decoded = decode_slides_from_storage(encoded)
        self.assertEqual(decoded[0]["caption"], "doc · slide 2")

    def test_record_from_session_atomic_write(self) -> None:
        record = record_from_session(
            "test-id",
            title="New conversation",
            created_at="2026-01-01T00:00:00+00:00",
            messages=[{"role": "user", "content": "Hello"}],
            conversation_memory={"summary": "", "last_interaction": None},
            teaching_session=TeachingSession.empty().to_dict(),
        )
        path = Path(self._tmpdir.name) / "manual.json"
        path.write_text(json.dumps(record.to_dict()), encoding="utf-8")
        self.assertIn("Hello", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
