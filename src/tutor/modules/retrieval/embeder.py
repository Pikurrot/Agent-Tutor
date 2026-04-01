from __future__ import annotations
import torch
from typing import Optional
from sentence_transformers import SentenceTransformer
from tutor.utils.paths import MODELS_CACHE_DIR


class Embeder:
    def __init__(self, config: dict):
        self.config = config
        self.my_config = config["retrieval_config"]["embeder_config"]
        self.model = SentenceTransformer(self.my_config["model_path"], cache_folder=MODELS_CACHE_DIR)

    def encode(
        self,
        texts: list[str],
        mode: Optional[str] = None
    ) -> torch.Tensor:
        if mode == "query":
            return self.model.encode(texts, prompt_name="query")
        else:
            return self.model.encode(texts)

    def similarity(
        self,
        query_embedding: torch.Tensor,
        document_embeddings: torch.Tensor
    ) -> torch.Tensor:
        return self.model.similarity(query_embedding, document_embeddings)

