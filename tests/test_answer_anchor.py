from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from tutor.core.conversation_eval import _classify_work
from tutor.modules.agent.answer_agent import run_answer_agent
from tutor.modules.agent.pedagogy import (
    TeachingAnchor,
    TeachingSession,
    is_anchor_failure_text,
)


VALID_ANCHOR_JSON = """{
  "target_explanation": "Backpropagation computes gradients via the chain rule.",
  "key_facts": ["chain rule", "loss gradient"],
  "misconceptions": ["confusing forward and backward pass"],
  "scaffold_questions": ["What does the loss depend on?", "How do we propagate error?"],
  "citations": ["lecture · slide 3"]
}"""


class TestAnchorValidity(unittest.TestCase):
    def test_rejects_iteration_limit_poison(self) -> None:
        anchor = TeachingAnchor(
            target_explanation="Agent stopped due to iteration limit or time limit.",
            key_facts=["x"],
        )
        self.assertFalse(anchor.is_valid())
        self.assertTrue(
            is_anchor_failure_text(
                "Agent stopped due to iteration limit or time limit."
            )
        )

    def test_accepts_well_formed_anchor(self) -> None:
        anchor = TeachingAnchor.from_dict(
            {
                "target_explanation": "Gradient descent updates weights.",
                "key_facts": ["learning rate"],
                "scaffold_questions": ["What is the update rule?"],
            }
        )
        self.assertTrue(anchor.is_valid())

    def test_from_dict_drops_invalid_anchor_in_session(self) -> None:
        session = TeachingSession.from_dict(
            {
                "anchor": {
                    "target_explanation": "Agent stopped due to iteration limit.",
                    "key_facts": ["only fact"],
                }
            }
        )
        self.assertIsNone(session.anchor)
        self.assertFalse(session.has_anchor())


class TestClassifyWork(unittest.TestCase):
    def test_skip_judged_with_valid_anchor(self) -> None:
        row = {
            "status": "judged",
            "student_reached": True,
            "tutor_revealed": False,
            "session_state": {
                "anchor": {
                    "target_explanation": "ok",
                    "key_facts": ["a"],
                    "scaffold_questions": ["q?"],
                }
            },
        }
        self.assertEqual(_classify_work(row), "skip")

    def test_run_judged_with_invalid_anchor(self) -> None:
        row = {
            "status": "judged",
            "student_reached": True,
            "tutor_revealed": False,
            "session_state": {
                "anchor": {
                    "target_explanation": "Agent stopped due to iteration limit.",
                    "key_facts": ["a"],
                }
            },
        }
        self.assertEqual(_classify_work(row), "run")

    def test_judge_only_when_complete_with_valid_anchor(self) -> None:
        row = {
            "status": "conversation_complete",
            "transcript": [{"turn": 1, "role": "student", "content": "hi"}],
            "session_state": {
                "anchor": {
                    "target_explanation": "ok",
                    "scaffold_questions": ["q?"],
                }
            },
        }
        self.assertEqual(_classify_work(row), "judge_only")


class TestRunAnswerAgent(unittest.TestCase):
    def test_direct_path_parses_anchor_and_populates_slides(self) -> None:
        qwen = MagicMock()
        qwen.generate.return_value = VALID_ANCHOR_JSON

        rag = MagicMock()
        rag.retrieve.return_value = (
            [
                {
                    "document_name": "lecture",
                    "slide_index": 2,
                    "transcript": "chain rule content",
                    "image": None,
                }
            ],
            {},
        )
        rag.retriever.documents_names = ["lecture"]
        rag.slides_for_ui = lambda data: [
            {"image": d.get("image"), "caption": f'{d["document_name"]} · slide 3'}
            for d in data
        ]

        anchor, slide_tool, trace = run_answer_agent(
            qwen,
            rag,
            {"pedagogic_config": {"anchor_max_new_tokens": 128}},
            "What is backpropagation?",
            debug=True,
        )

        self.assertTrue(anchor.is_valid())
        self.assertIn("chain rule", anchor.target_explanation.lower())
        self.assertEqual(len(slide_tool.retrieved_slides), 1)
        self.assertIsNotNone(trace)
        assert trace is not None
        self.assertTrue(trace["anchor_valid"])
        self.assertFalse(trace["parse_failed"])
        qwen.generate.assert_called()


if __name__ == "__main__":
    unittest.main()
