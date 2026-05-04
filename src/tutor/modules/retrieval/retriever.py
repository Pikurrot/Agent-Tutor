from __future__ import annotations
import os
import json
import torch
from pathlib import Path
from collections import defaultdict
from typing import Callable, Optional
from pdf2image import convert_from_path

from tutor.utils.paths import SUBJECTS_ROOT
from tutor.modules.retrieval.embeder import Embeder

class Retriever:
    def __init__(self, config: dict):
        self.config = config
        self.my_config = config["retrieval_config"]
        self.subject_name = self.my_config["subject_name"]
        self.processed_dir = SUBJECTS_ROOT / self.subject_name / "processed"
        self.raw_dir = SUBJECTS_ROOT / self.subject_name / "raw"

        self.embeder = Embeder(self.config)
        self.aggregated_document_embeddings = []
        self.documents_embeddings = {}
        self.documents_data = {}
        self.documents_names = [pdf_name.stem for pdf_name in Path(SUBJECTS_ROOT / self.subject_name / "raw" / "slides").glob("*.pdf")]

    def _get_top_indices(
        self,
        sim: torch.Tensor
    ) -> torch.Tensor:
        sim = sim.detach().reshape(-1)
        n = sim.numel()
        if n == 0:
            return torch.tensor([], dtype=torch.long, device=sim.device)

        mode = self.my_config["retrieval_mode"]
        if mode == "topk":
            k = self.my_config["retrieval_k"]
            k = min(k, n)
            indices = torch.topk(sim, k=k).indices
        elif mode == "threshold":
            threshold = self.my_config["retrieval_threshold"]
            indices = torch.where(sim > threshold)[0]
        elif mode == "zscore":
            zscore = self.my_config["retrieval_zscore"]
            zscore_max_k = self.my_config["retrieval_zscore_max_k"]
            mean = torch.mean(sim)
            std = torch.std(sim)
            indices = torch.where(sim > mean + zscore * std)[0]
        else:
            raise ValueError(f"Unknown retrieval_mode: {mode}")

        if indices.numel() == 0:
            return indices

        indices = indices[torch.argsort(sim[indices], descending=True)]
        if mode == "zscore" and indices.numel() > zscore_max_k:
            indices = indices[:zscore_max_k]
        return indices

    def _get_transcript(self, data: dict) -> str:
        transcript = []
        for video_data in data.values():
            for segment_data in video_data:
                transcript.append(segment_data[2])
        return " ".join(transcript).replace("  ", " ").strip()

    def load_document_data(
        self,
        document_name: str
    ):
        transcription_file = self.processed_dir / "transcriptions" / f"{document_name}.json"
        with open(transcription_file, "r") as f:
            transcription_data = json.load(f)
        pdf_path = self.raw_dir / "slides" / f"{document_name}.pdf"
        pdf_images = convert_from_path(pdf_path)
        slide2data = []
        for slide_index, slide_data in transcription_data.items():
            slide_index = int(slide_index)
            transcript = self._get_transcript(slide_data)
            slide2data.append({"transcript": transcript, "image": pdf_images[slide_index]})
        self.documents_data[document_name] = slide2data
        return slide2data

    def encode_document(
        self,
        document_name: str,
        save_path: Path
    ) -> torch.Tensor:
        if document_name in self.documents_embeddings:
            return self.documents_embeddings[document_name]
        elif save_path.exists():
            loaded = torch.load(save_path, map_location="cpu", weights_only=False)
            self.documents_embeddings[document_name] = loaded
            return loaded

        slide2data = self.documents_data[document_name]
        transcripts = [slide["transcript"] for slide in slide2data]
        transcript_embeddings = self.embeder.encode(transcripts)
        self.documents_embeddings[document_name] = transcript_embeddings

        os.makedirs(self.processed_dir / "embeddings", exist_ok=True)
        torch.save(transcript_embeddings, save_path)

        return transcript_embeddings

    def aggregate_document_embeddings(self, save_path: Path) -> torch.Tensor:
        if save_path.exists():
            loaded = torch.load(save_path, map_location="cpu", weights_only=False)
            if isinstance(loaded, torch.Tensor) and loaded.ndim == 2:
                self.aggregated_document_embeddings = loaded
                return self.aggregated_document_embeddings

        per_doc_aggregates = []
        for doc_name in self.documents_names:
            if doc_name not in self.documents_embeddings:
                doc_save_path = self.processed_dir / "embeddings" / f"{doc_name}.pt"
                if not doc_save_path.exists() and doc_name not in self.documents_data:
                    self.load_document_data(doc_name)
                self.encode_document(doc_name, doc_save_path)
            document_embeddings = self.documents_embeddings[doc_name]
            document_embeddings = torch.as_tensor(document_embeddings)
            aggregated = torch.sum(document_embeddings, dim=0)
            per_doc_aggregates.append(aggregated)
        self.aggregated_document_embeddings = torch.stack(per_doc_aggregates, dim=0)
        os.makedirs(save_path.parent, exist_ok=True)
        torch.save(self.aggregated_document_embeddings, save_path)
        return self.aggregated_document_embeddings

    def get_similar_document(
        self,
        query: str
    ) -> str:
        if not isinstance(self.aggregated_document_embeddings, torch.Tensor):
            save_path = self.processed_dir / "embeddings" / "aggregated_document_embeddings.pt"
            self.aggregate_document_embeddings(save_path)
        query_embedding = self.embeder.encode([query], mode="query")
        sim = self.embeder.similarity(query_embedding, self.aggregated_document_embeddings)
        sim = sim.detach().reshape(-1)
        if sim.numel() == 0:
            raise RuntimeError("No documents available to compare against the query.")
        best_index = int(torch.argmax(sim).item())
        return self.documents_names[best_index]

    def retrieve(
        self,
        query: Optional[str] = None,
        document_name: Optional[str] = None,
        slide_number: Optional[int] = None,
        on_progress: Optional[Callable] = None,
    ) -> tuple[list[dict], dict]:
        if query is None and document_name is not None and slide_number is None:
            raise ValueError("query is required if document_name is provided and slide_number is not")
        if slide_number is not None and document_name is None:
            raise ValueError("document_name is required if slide_number is provided")

        def progress(msg: str) -> None:
            if on_progress is not None:
                on_progress(msg)

        if self.my_config["mode"] == "tree" and document_name is None:
            document_name = self.get_similar_document(query)
            progress(f"Retrieving context from similar document: {document_name}")
            return self.retrieve(query, document_name, slide_number, on_progress)

        if document_name is None:
            progress("Preparing retrieval...")
            total_document_embeddings = []
            total_document_names = []
            total_original_indices = []
            for doc_name in self.documents_names:
                if doc_name not in self.documents_data:
                    progress(f"Loading slides/transcripts: {doc_name}")
                    self.load_document_data(doc_name)
                if doc_name not in self.documents_embeddings:
                    save_path = self.processed_dir / "embeddings" / f"{doc_name}.pt"
                    if save_path.exists():
                        progress(f"Loading embeddings: {doc_name}")
                    else:
                        progress(f"Encoding embeddings: {doc_name}")
                    self.encode_document(doc_name, save_path)
                document_embeddings = self.documents_embeddings[doc_name]
                document_embeddings = torch.as_tensor(document_embeddings)
                total_document_embeddings.append(document_embeddings)
                total_document_names.extend([doc_name] * document_embeddings.shape[0])
                total_original_indices.extend(list(range(document_embeddings.shape[0])))
            total_document_embeddings = torch.cat(total_document_embeddings, dim=0)
            progress("Retrieving context...")
            query_embedding = self.embeder.encode([query], mode="query")
            sim = self.embeder.similarity(query_embedding, total_document_embeddings)
            top_indices = self._get_top_indices(sim)
            document_names_top_indices = [total_document_names[int(i)] for i in top_indices]
            original_indices_top_indices = [total_original_indices[int(i)] for i in top_indices]
            retrieved_data = []
            metadata = defaultdict(list)
            for doc_nm, original_index in zip(document_names_top_indices, original_indices_top_indices):
                document_data = self.documents_data[doc_nm]
                oi = int(original_index)
                item = document_data[oi].copy()
                item["document_name"] = doc_nm
                item["slide_index"] = oi
                retrieved_data.append(item)
                metadata[doc_nm].append(oi)

        elif document_name in self.documents_names:
            name = document_name
            if name not in self.documents_data:
                progress(f"Loading slides/transcripts: {name}")
                self.load_document_data(name)
            document_data = self.documents_data[name]
            if slide_number is not None:
                num_slides = len(document_data)
                if not 0 <= int(slide_number) < num_slides:
                    raise ValueError(
                        f"slide_number {slide_number} is out of range for document "
                        f"\"{name}\" (valid range: 0..{num_slides - 1})."
                    )
                top_indices = [int(slide_number)]
            else:
                if name not in self.documents_embeddings:
                    save_path = self.processed_dir / "embeddings" / f"{name}.pt"
                    if save_path.exists():
                        progress(f"Loading embeddings: {name}")
                    else:
                        progress(f"Encoding embeddings: {name}")
                    self.encode_document(name, save_path)
                document_embeddings = self.documents_embeddings[name]

                progress("Retrieving context...")
                query_embedding = self.embeder.encode([query], mode="query")
                sim = self.embeder.similarity(query_embedding, document_embeddings)
                top_indices = self._get_top_indices(sim)

            retrieved_data = []
            for i in top_indices:
                ii = int(i)
                item = document_data[ii].copy()
                item["document_name"] = name
                item["slide_index"] = ii
                retrieved_data.append(item)
            metadata = {name: [int(i) for i in top_indices]}
        else:
            raise ValueError(f"Document {document_name} not in index.")
        msg = "Retrieved slides "
        for i, (document_name, slide_indices) in enumerate(metadata.items()):
            if i > 0:
                if i == len(metadata) - 1:
                    msg += " and "
                else:
                    msg += ", "
            msg += ", ".join(str(index) for index in slide_indices) + " from \"" + document_name + "\""
        progress(msg)
        return retrieved_data, metadata
