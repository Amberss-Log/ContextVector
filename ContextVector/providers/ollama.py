import ollama
from typing import List
from interfaces import EmbeddingProvider

class OllamaProvider(EmbeddingProvider):
    """
    A concrete implementation of EmbeddingProvider that communicates 
    with your local Ollama background server.
    """
    def embed_text(self, text: str, model_name: str) -> List[float]:
        """Calls the local Ollama SDK to convert a text string into a vector."""
        response = ollama.embed(model=model_name, input=text)
        return response['embeddings'][0]