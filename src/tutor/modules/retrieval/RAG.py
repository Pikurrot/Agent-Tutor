from __future__ import annotations
import json
from typing import Any, Callable, Optional
from langchain_core.tools import StructuredTool, Tool

from tutor.modules.retrieval.retriever import Retriever


class RAGModule:
    def __init__(self, config: dict):
        self.config = config
        self.retriever = Retriever(self.config)

    def retrieve(
        self,
        query: Optional[str] = None,
        document_name: Optional[str] = None,
        slide_number: Optional[int] = None,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> tuple[list[dict], dict]:
        return self.retriever.retrieve(query, document_name, slide_number, on_progress=on_progress)

    def augment_prompt(
        self,
        prompt: str,
        retrieved_data: list[dict]
    ) -> str:
        transcripts = [data["transcript"] for data in retrieved_data]

        augmented_prompt = "Given the following related lecture transcripts:\n"
        augmented_prompt += "\n".join(transcripts) + "\n"
        augmented_prompt += "Respond to the following query: " + prompt + "\n"
        augmented_prompt += "Respond briefly and concisely, focusing on the given information in the transcripts."
        return augmented_prompt

    @staticmethod
    def slides_for_ui(retrieved_data: list[dict]) -> list[dict[str, Any]]:
        slides: list[dict[str, Any]] = []
        for d in retrieved_data:
            slides.append(
                {
                    "image": d["image"],
                    "caption": f'{d["document_name"]} · slide {d["slide_index"] + 1}',
                }
            )
        return slides

    def get_all_lecture_chunks(self) -> list[dict]:
        """Return all slide transcript chunks across lectures (text only)."""
        return self.retriever.load_all_slide_transcripts()

    def retrieve_and_augment(
        self,
        query: Optional[str] = None,
        document_name: Optional[str] = None,
        slide_number: Optional[int] = None,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> tuple[str, list[dict[str, Any]], list]:
        retrieved_data, _metadata = self.retrieve(query, document_name, slide_number, on_progress=on_progress)
        if on_progress is not None:
            on_progress("Building augmented prompt…")
        augmented_prompt = self.augment_prompt(query, retrieved_data)
        images = [d["image"] for d in retrieved_data]
        return augmented_prompt, self.slides_for_ui(retrieved_data), images


class SlideRetrieverTool:
    def __init__(self, rag_module: RAGModule):
        self.rag_module = rag_module
        self.retrieved_slides = []
        self.on_progress_callback: Optional[Callable[[str], None]] = None

    def set_progress_callback(self, callback: Callable[[str], None]):
        self.on_progress_callback = callback

    def prepare_retrieved_data(self, retrieved_data: list[dict]) -> str:
        self.retrieved_slides.extend(self.rag_module.slides_for_ui(retrieved_data))

        if not retrieved_data:
            return "No relevant slides found."

        chunks: list[str] = []
        for d in retrieved_data:
            doc_name = d.get("document_name", "unknown document")
            slide_index = d.get("slide_index")
            header = (
                f"[document: \"{doc_name}\" | slide_number: {slide_index+1}]"
                if slide_index is not None
                else f"[document: \"{doc_name}\"]"
            )
            chunks.append(f"{header}\n{d['transcript']}")
        return "\n---\n".join(chunks)

    def retrieve_full_context(
        self,
        query: str
    ) -> str:
        retrieved_data, _ = self.rag_module.retrieve(query, on_progress=self.on_progress_callback)
        return self.prepare_retrieved_data(retrieved_data)

    def retrieve_document_context(
        self,
        document_name: str,
        query: str
    ) -> str:
        retrieved_data, _ = self.rag_module.retrieve(query, document_name, on_progress=self.on_progress_callback)
        return self.prepare_retrieved_data(retrieved_data)

    def retrieve_slide_context(
        self,
        document_name: str,
        slide_number: int,
    ) -> str:
        retrieved_data, _ = self.rag_module.retrieve(None, document_name, slide_number, on_progress=self.on_progress_callback)
        return self.prepare_retrieved_data(retrieved_data)

    @staticmethod
    def _parse_json_action_input(raw_input: str, required_keys: tuple[str, ...]) -> dict:
        if not isinstance(raw_input, str):
            raise ValueError(f"Expected JSON-string Action Input, got {type(raw_input).__name__}.")
        text = raw_input.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Action Input must be a JSON object with keys {list(required_keys)}. Could not parse: {raw_input!r} ({e})"
            )
        if not isinstance(parsed, dict):
            raise ValueError(f"Action Input JSON must be an object, got {type(parsed).__name__}.")
        missing = [k for k in required_keys if k not in parsed]
        if missing:
            raise ValueError(f"Action Input JSON is missing required keys: {missing}.")
        return parsed

    def _retrieve_document_context_from_json(self, raw_input: str) -> str:
        parsed = self._parse_json_action_input(raw_input, ("document_name", "query"))
        return self.retrieve_document_context(str(parsed["document_name"]), str(parsed["query"]))

    def _retrieve_slide_context_from_json(self, raw_input: str) -> str:
        parsed = self._parse_json_action_input(raw_input, ("document_name", "slide_number"))
        # convert to 0-based index
        slide_number = int(parsed["slide_number"]) - 1
        return self.retrieve_slide_context(str(parsed["document_name"]), slide_number)

    def get_tool(self, tool_name: str):
        if tool_name == "Search_All_Course_Context":
            return StructuredTool.from_function(
                name="Search_All_Course_Context",
                func=self.retrieve_full_context,
                description=(
                    "Searches all lecture transcripts and course slides across the entire course using the provided query. "
                    "Use this when you want to retrieve relevant information from any document in the course. "
                    "Input: a specific search query string."
                )
            )
        elif tool_name == "Search_Document_Context":
            return Tool.from_function(
                name="Search_Document_Context",
                func=self._retrieve_document_context_from_json,
                description=(
                    "Searches within a specific document (lecture or slide deck) using the provided document name and query. "
                    "Use this when you want to find relevant information inside one particular lecture or slide deck. "
                    "Use only if you know exactly the document name you wish to query. "
                    'Input: a single JSON object string with keys "document_name" (str) and "query" (str), '
                    'e.g. {"document_name": "Lecture 2 - Backpropagation", "query": "what is gradient descent"}.'
                )
            )
        elif tool_name == "Retrieve_Slide_Context":
            return Tool.from_function(
                name="Retrieve_Slide_Context",
                func=self._retrieve_slide_context_from_json,
                description=(
                    "Directly retrieves the context (transcript and slide) for a specific slide in a document. "
                    "Use this when you know the document name and slide index you wish to query. "
                    "Use only if you know exactly the document name and slide index you wish to query. "
                    'Input: a single JSON object string with keys "document_name" (str) and "slide_number" (int), '
                    'e.g. {"document_name": "Lecture 2 - Backpropagation", "slide_number": 26}.'
                )
            )
