import numpy as np
from typing import Any, Union, Dict, Self
from interfaces import EmbeddingProvider

class ContextVector:
    """
    ContextVector: The custom multimodal AI data type.
    Binds high-dimensional vector embeddings directly to payloads and modalities.
    """
    def __init__(self, embedding: Union[list, np.ndarray], payload: Any, modality: str = "text", metadata: Dict = None):
        if not isinstance(embedding, (list, np.ndarray)):
            raise TypeError("Embedding must be a list of numbers or a NumPy ndarray.")
            
        self._embedding = np.array(embedding, dtype=np.float32)
        self._embedding.flags.writeable = False 

        self.payload = payload
        self.modality = modality.lower()
        self.metadata = metadata if metadata is not None else {}

    @property
    def embedding(self) -> np.ndarray:
        return self._embedding

    @property
    def dimensions(self) -> int:
        return self._embedding.shape[0]

    @classmethod
    def from_text(cls, text: str, provider: EmbeddingProvider, model_name: str, metadata: Dict = None) -> Self:
        """Factory Method: Automates vector extraction using any compliant provider."""
        generated_vector = provider.embed_text(text, model_name=model_name)
        return cls(
            embedding=generated_vector,
            payload=text,
            modality="text",
            metadata=metadata
        )

    def similarity(self, other: Self) -> float:
        """Computes native Cosine Similarity between two ContextVector instances."""
        if not isinstance(other, ContextVector):
            raise TypeError(f"Unsupported operand type for similarity: 'ContextVector' and '{type(other).__name__}'")
        if self.dimensions != other.dimensions:
            raise ValueError(f"Dimension Mismatch: {self.dimensions} vs {other.dimensions}")

        dot_product = np.dot(self._embedding, other._embedding)
        norm_self = np.linalg.norm(self._embedding)
        norm_other = np.linalg.norm(other._embedding)
        
        if norm_self == 0.0 or norm_other == 0.0:
            return 0.0
            
        return float(dot_product / (norm_self * norm_other))

    def __repr__(self) -> str:
        return f"<ContextVector | Modality: {self.modality.upper()} | Dimensions: {self.dimensions}>"