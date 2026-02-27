from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Generator


class BaseModel(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        ...

    @abstractmethod
    def stream_generate(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        ...

    def eval(self):
        pass

    def train(self):
        pass
