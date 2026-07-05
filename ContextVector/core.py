import numpy as np
from typing import Any, Union, Dict, Optional, Self
from .interfaces import EmbeddingProvider

class ContextVector:
    """
    ContextVector: The custom multimodal AI data type.
    Binds high-dimensional vector embeddings directly to payloads and modalities.
    """
    def __init__(
        self, 
        embedding: Union[list, np.ndarray], 
        payload: Any, 
        modality: str = "text", 
        metadata: Optional[Dict[str, Any]] = None  # Clean, precise type hinting
    ):
        
        try:
            self._embedding = np.asarray(embedding, dtype=np.float32)
        except (TypeError, ValueError):
            raise TypeError(
                "Embedding must be an iterable of numeric values."
            )

        if self._embedding.ndim != 1:
            raise ValueError(
                "Embedding must be a one-dimensional vector."
            )

        if self._embedding.size == 0:
            raise ValueError(
                "Embedding cannot be empty."
            )

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
    def from_text(
        cls, 
        text: str, 
        provider: EmbeddingProvider, 
        model_name: str, 
        metadata: Optional[Dict[str, Any]] = None  # Aligned factory type hinting
    ) -> Self:
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

    # NATIVE SERIALIZATION CORE UTILITIES

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialization: Converts the active memory object into a standard, 
        JSON-compliant Python dictionary. Unpacks NumPy binary arrays.
        """
        return {
            "embedding": self._embedding.tolist(),  # Critical conversion to plain list
            "payload": self.payload,
            "modality": self.modality,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """
        Deserialization Factory: Reconstitutes a live, functional ContextVector 
        instance from a standard dictionary payload (e.g. from a database).
        """
        # Extract the fields safely with default fallback parameters
        return cls(
            embedding=data["embedding"],
            payload=data["payload"],
            modality=data.get("modality", "text"),
            metadata=data.get("metadata", {})
        )

    # NATIVE COMPARISON UTILITIES

    def __eq__(self, other: object) -> bool:
        """
        Performs structural equality checks between two instances.
        Verifies modality, payload, and exact vector weight alignment.
        """
        if not isinstance(other, ContextVector):
            return False
            
        return (
            self.modality == other.modality and
            self.payload == other.payload and
            np.array_equal(self._embedding, other._embedding)
        )
        
    def __len__(self) -> int:
        """
        Returns the dimensionality of the embedding.
        """
        return self.dimensions
    
    def __getitem__(self, index: int) -> float:
        """
        Returns the embedding coordinate at the given index.
        """
        return float(self._embedding[index])
    
    def __iter__(self):
        """
        Iterate over embedding coordinates.
        """
        return iter(self._embedding)
    
    def __contains__(self, value: float) -> bool:
    
        return value in self._embedding

        
    def __repr__(self) -> str:
        return (
            f"ContextVector("
            f"modality='{self.modality}', "
            f"dimensions={self.dimensions}, "
            f"payload_type={type(self.payload).__name__}"
            f")"
        )