from abc import ABC, abstractmethod
import numpy as np
import ollama
from typing import Any, Union, Dict, List, Self

# =====================================================================
# 1. THE ABSTRACT INTERFACE (The Contract)
# =====================================================================

class EmbeddingProvider(ABC):
    """
    Abstract Base Class (ABC). It defines a strict template that any 
    model provider (like Ollama or Hugging Face) must follow.
    """
    @abstractmethod
    def embed_text(self, text: str, model_name: str) -> List[float]:
        pass


# =====================================================================
# 2. THE CONCRETE OLLAMA PROVIDER (The Engine)
# =====================================================================

class OllamaProvider(EmbeddingProvider):
    """
    A concrete implementation of EmbeddingProvider that communicates 
    with your local Ollama background server.
    """
    def embed_text(self, text: str, model_name: str) -> List[float]:

        response = ollama.embed(model=model_name, input=text)
        return response['embeddings'][0]


# =====================================================================
# 3. THE CORE DATA TYPE (With Native Vector Math)
# =====================================================================

class ContextVector:
    """
    ContextVector: The custom multimodal AI data type.
    Binds high-dimensional vector embeddings directly to payloads and modalities.
    """
    def __init__(self, embedding: Union[list, np.ndarray], payload: Any, modality: str = "text", metadata: Dict = None):
        if not isinstance(embedding, (list, np.ndarray)):
            raise TypeError("Embedding must be a list of numbers or a NumPy ndarray.")
            
        # Freeze the vector array so it cannot be altered after creation
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
        generated_vector = provider.embed_text(text, model_name=model_name)
        return cls(
            embedding=generated_vector,
            payload=text,
            modality="text",
            metadata=metadata
        )

    # -----------------------------------------------------------------
    # NATIVE MATHEMATICAL UTILITIES
    # -----------------------------------------------------------------

    def similarity(self, other: Self) -> float:
        """
        Computes the native Cosine Similarity between this ContextVector 
        and another ContextVector instance.
        
        Formula: (A • B) / (||A|| * ||B||)
        Returns a float between -1.0 and 1.0 representing semantic closeness.
        """
        if not isinstance(other, ContextVector):
            raise TypeError(
                f"Unsupported operand type for similarity: 'ContextVector' and '{type(other).__name__}'"
            )
            
        if self.dimensions != other.dimensions:
            raise ValueError(
                f"Dimension Mismatch: Cannot calculate similarity between spaces of "
                f"{self.dimensions} and {other.dimensions} dimensions."
            )

        dot_product = np.dot(self._embedding, other._embedding)
        
        norm_self = np.linalg.norm(self._embedding)
        norm_other = np.linalg.norm(other._embedding)
        
        if norm_self == 0.0 or norm_other == 0.0:
            return 0.0
            
        return float(dot_product / (norm_self * norm_other))

    def __repr__(self) -> str:
        return f"<ContextVector | Modality: {self.modality.upper()} | Dimensions: {self.dimensions}>"


# =====================================================================
# 4. THE LIVE TEST BENCH (Cross-Sentence Semantic Test)
# =====================================================================

# if __name__ == "__main__":
#     print(" Initializing the Local Ollama Provider...")
#     provider = OllamaProvider()
#     target_model = "nomic-embed-text"
    
#     # Let's generate THREE distinct vectors to test semantic math rules
#     print("\n Generating high-dimensional coordinates for 3 sentences...")
    
#     cv_ai_1 = ContextVector.from_text(
#         text="Artificial intelligence and machine learning architectures require novel engineering pipelines.", 
#         provider=provider, 
#         model_name=target_model
#     )
    
#     cv_ai_2 = ContextVector.from_text(
#         text="Deep neural networks and statistical computer models optimize processing infrastructure.", 
#         provider=provider, 
#         model_name=target_model
#     )
    
#     cv_pizza = ContextVector.from_text(
#         text="A wood-fired neapolitan pizza topped with fresh basil and melted mozzarella cheese.", 
#         provider=provider, 
#         model_name=target_model
#     )
    
#     print(" All 3 ContextVectors loaded into memory successfully.")
#     print("-" * 65)
    
#     # Test 1: Compare related concepts (AI vs AI)
#     sim_related = cv_ai_1.similarity(cv_ai_2)
#     print(f"Test 1 (Related Meanings):\n -> AI Concept 1 vs AI Concept 2\n -> Cosine Similarity Score: {sim_related:.4f}")
#     print("-" * 65)
    
#     # Test 2: Compare completely unrelated concepts (AI vs Pizza)
#     sim_unrelated = cv_ai_1.similarity(cv_pizza)
#     print(f"Test 2 (Unrelated Meanings):\n -> AI Concept 1 vs Pizza Recipe\n -> Cosine Similarity Score: {sim_unrelated:.4f}")
#     print("-" * 65)
    
#     # Test 3: System Robustness Check (Should reject invalid types gracefully)
#     print("Test 3 (Security Guardrail Validation):")
#     try:
#         cv_ai_1.similarity("Just a plain string object") # type: ignore
#     except TypeError as error:
#         print(f" -> Caught Expected Error Successfully: {error}")
#     print("-" * 65)