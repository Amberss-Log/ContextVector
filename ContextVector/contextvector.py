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
        """Must accept text and a model name, and return a list of floats."""
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
        """
        Calls the local Ollama SDK to convert a text string into a vector.
        """
        # Call the local service running on your machine
        response = ollama.embed(model=model_name, input=text)
        
        # Ollama returns a dictionary. The vector is stored inside the 'embeddings' key.
        # It returns a list of lists (for batching), so we take the first element [0]
        return response['embeddings'][0]


# =====================================================================
# 3. THE CORE DATA TYPE (The Custom Primitive)
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
        """
        Factory Method: Automates vector extraction using any compliant provider.
        """
        # Polymorphism: The data type executes whatever provider is passed in
        generated_vector = provider.embed_text(text, model_name=model_name)
        
        return cls(
            embedding=generated_vector,
            payload=text,
            modality="text",
            metadata=metadata
        )

    def __repr__(self) -> str:
        return f"<ContextVector | Modality: {self.modality.upper()} | Dimensions: {self.dimensions}>"


# =====================================================================
# 4. THE LIVE TEST RUN (The Test Bench)
# =====================================================================

if __name__ == "__main__":
    print(" Initializing the Local Ollama Provider...")
    provider = OllamaProvider()
    
    # Define our raw data
    raw_text = "Emerging multimodal AI architectures require native data types."
    target_model = "nomic-embed-text"
    
    print(f" Processing raw payload text: '{raw_text}'")
    print(f" Generating vector weights using model: '{target_model}'...")
    
    # Execute the factory method! No manual array generation required by the developer.
    try:
        cv = ContextVector.from_text(text=raw_text, provider=provider, model_name=target_model)
        
        print("\n SUCCESS! ContextVector Created Successfully:")
        print("-" * 50)
        print(f"Instance Representation : {cv}")
        print(f"Payload Text Stored     : {cv.payload}")
        print(f"Vector Dimensions Check : {cv.dimensions} dimensions")
        print(f"First 5 Vector Weights  : {cv.embedding[:5]}")
        print("-" * 50)
        
    except Exception as e:
        print(f"\n Error encountered during execution: {e}")
        print("Ensure the Ollama application is running and 'ollama pull nomic-embed-text' has completed.")