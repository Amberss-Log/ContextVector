from abc import ABC, abstractmethod
from typing import List

class EmbeddingProvider(ABC):
    """
    Abstract Base Class (ABC). It defines a strict template that any 
    model provider (like Ollama or Hugging Face) must follow.
    """
    @abstractmethod
    def embed_text(self, text: str, model_name: str) -> List[float]:
        """Must accept text and a model name, and return a list of floats."""
        pass