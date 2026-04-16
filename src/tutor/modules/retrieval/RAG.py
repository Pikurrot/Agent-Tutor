from __future__ import annotations
from typing import Any, Callable, Optional
from langchain_core.tools import StructuredTool

from tutor.modules.retrieval.retriever import Retriever


class RAGModule:
    def __init__(self, config: dict):
        self.config = config
        self.retriever = Retriever(self.config)

    def retrieve(
        self,
        query: str,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> tuple[list[dict], dict]:
        return self.retriever.retrieve(query, on_progress=on_progress)

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

    def retrieve_and_augment(
        self,
        query: str,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> tuple[str, list[dict[str, Any]], list]:
        retrieved_data, _metadata = self.retrieve(query, on_progress=on_progress)
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

    def retrieve_for_agent(
        self,
        query: str
    ) -> str:
        retrieved_data, _ = self.rag_module.retrieve(query, on_progress=self.on_progress_callback)

        self.retrieved_slides.extend(self.rag_module.slides_for_ui(retrieved_data))

        transcripts = [d["transcript"] for d in retrieved_data]
        if not transcripts:
            return "No relevant slides found."

        return "\n---\n".join(transcripts)

    def get_tool(self):
        return StructuredTool.from_function(
            name="Search_Course_Slides",
            func=self.retrieve_for_agent,
            description="Useful for searching lecture transcripts and course slides to answer questions. Input should be a specific search query."
        )
