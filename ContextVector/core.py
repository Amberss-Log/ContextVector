import numpy as np
from typing import Any, Union, Dict, Optional, Self
from .interfaces import EmbeddingProvider

class ContextVector:
    """
    ContextVector: An immutable, framework-agnostic multimodal AI data primitive.
    Encapsulates coordinates, raw payloads, provenance markers, and metadata.
    """
    def __init__(
        self, 
        embedding: Union[list, np.ndarray], 
        payload: Any, 
        modality: str = "text", 
        metadata: Optional[Dict[str, Any]] = None,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        # 1. Structural & Datatype Validation
        try:
            self._embedding = np.asarray(embedding, dtype=np.float32)
        except (TypeError, ValueError):
            raise TypeError("Embedding must be an iterable of numeric values.")

        if self._embedding.ndim != 1:
            raise ValueError("Embedding must be a one-dimensional vector.")

        if self._embedding.size == 0:
            raise ValueError("Embedding cannot be empty.")
            
        if np.isnan(self._embedding).any() or np.isinf(self._embedding).any():
            raise ValueError("Embedding contains invalid numeric values (NaN or Inf).")

        # Freeze the underlying NumPy binary allocation array flags
        self._embedding.flags.writeable = False

        # 2. Immutable Semantic State Allocation
        self.payload = payload
        self.modality = modality.lower()
        self.metadata = metadata if metadata is not None else {}
        
        # Provenance properties to prevent comparing vectors from different spaces
        self._provider_name = str(provider_name) if provider_name is not None else "unknown"
        self._model_name = str(model_name) if model_name is not None else "unknown"

    @property
    def embedding(self) -> np.ndarray:
        return self._embedding

    @property
    def dimensions(self) -> int:
        return self._embedding.shape[0]

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @classmethod
    def from_text(
        cls, 
        text: str, 
        provider: EmbeddingProvider, 
        model_name: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Self:
        """Factory Method: Generates a vector wrapper with explicit provenance tagging."""
        if not isinstance(text, str):
            raise TypeError("Input payload for from_text factory must be a string.")
        if not text.strip():
            raise ValueError("Input text payload cannot be empty or whitespace-only.")

        generated_vector = provider.embed_text(text, model_name=model_name)
        
        return cls(
            embedding=generated_vector,
            payload=text,
            modality="text",
            metadata=metadata,
            provider_name=type(provider).__name__,
            model_name=model_name
        )

    def similarity(self, other: Self) -> float:
        """
        Computes the Cosine Similarity metric against another ContextVector instance.
        If either vector is a zero-vector, mathematically undefined (0/0) defaults to 0.0.
        """
        if not isinstance(other, ContextVector):
            raise TypeError(f"Unsupported operand type for similarity: 'ContextVector' and '{type(other).__name__}'")
            
        # Semantic Integrity Protection: Enforce dimension bounds
        if self.dimensions != other.dimensions:
            raise ValueError(f"Dimension Mismatch: Cannot align {self.dimensions}D vs {other.dimensions}D.")

        # Provenance Cross-Space Protection: Catch matching dimensions from different models
        if self.model_name != other.model_name or self.provider_name != other.provider_name:
            raise ValueError(
                f"Provenance Conflict: Vectors belong to incompatible spaces. "
                f"Source A: [{self.provider_name}/{self.model_name}] vs "
                f"Source B: [{other.provider_name}/{other.model_name}]"
            )

        dot_product = np.dot(self._embedding, other._embedding)
        norm_self = np.linalg.norm(self._embedding)
        norm_other = np.linalg.norm(other._embedding)
        
        # Zero-vector guard configuration handling
        if norm_self == 0.0 or norm_other == 0.0:
            return 0.0
            
        return float(dot_product / (norm_self * norm_other))

    def to_dict(self) -> Dict[str, Any]:
        """Maps data properties into a standard JSON-compliant dictionary payload."""
        return {
            "embedding": self._embedding.tolist(),
            "payload": self.payload,
            "modality": self.modality,
            "metadata": self.metadata,
            "provenance": {
                "provider_name": self._provider_name,
                "model_name": self._model_name,
                "dimensions": self.dimensions
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """
        Deserialization Factory: Reconstitutes a ContextVector instance.
        Executes structural integrity checks against tampering or serialization corruption.
        """
        if not isinstance(data, dict):
            raise TypeError("Deserialization source payload must be a dictionary map.")
            
        required_keys = {"embedding", "payload"}
        if not required_keys.issubset(data):
            raise ValueError(f"Malformed payload data structural map. Missing keys: {required_keys - data.keys()}")

        # Extract provenance dictionary block if it exists
        provenance = data.get("provenance", {})
        provider_name = provenance.get("provider_name")
        model_name = provenance.get("model_name")
        expected_dims = provenance.get("dimensions")

        # Initialize instance (triggers core array validation rules)
        instance = cls(
            embedding=data["embedding"],
            payload=data["payload"],
            modality=data.get("modality", "text"),
            metadata=data.get("metadata", {}),
            provider_name=provider_name,
            model_name=model_name
        )

        # Audit structural dimensions to catch truncation or database row corruptions
        if expected_dims is not None and instance.dimensions != expected_dims:
            raise ValueError(
                f"Integrity Check Failed: Vector size mismatch during recovery. "
                f"Expected {expected_dims} dimensions, but unpacked {instance.dimensions}."
            )

        return instance

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ContextVector):
            return False
            
        return (
            self.modality == other.modality and
            self.payload == other.payload and
            self.provider_name == other.provider_name and
            self.model_name == other.model_name and
            np.array_equal(self._embedding, other._embedding)
        )

    def __len__(self) -> int:
        return self.dimensions
    
    def __getitem__(self, index: int) -> float:
        return float(self._embedding[index])
    
    def __iter__(self):
        return iter(self._embedding)
    
    def __contains__(self, value: float) -> bool:
        return value in self._embedding
        
    def __repr__(self) -> str:
        return (
            f"ContextVector("
            f"modality='{self.modality}', "
            f"dimensions={self.dimensions}, "
            f"source='{self.provider_name}/{self.model_name}'"
            f")"
        )