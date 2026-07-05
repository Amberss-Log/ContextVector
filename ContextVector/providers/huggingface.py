from typing import List
from ..interfaces import EmbeddingProvider

class HuggingFaceProvider(EmbeddingProvider):
    """
    Concrete implementation of EmbeddingProvider running local Hugging Face
    models via the sentence-transformers ecosystem. No background servers required.
    """
    def __init__(self):
        self._cached_models = {}

    def embed_text(self, text: str, model_name: str = "all-MiniLM-L6-v2") -> List[float]:
        """Loads model into active memory and generates embeddings locally."""
        # Deferred import to ensure fast package boot time
        from sentence_transformers import SentenceTransformer
        
        # Cache model weights to avoid reloading from disk on every call
        if model_name not in self._cached_models:
            self._cached_models[model_name] = SentenceTransformer(model_name)
            
        model = self._cached_models[model_name]
        embedding = model.encode(text, convert_to_numpy=False)
        
        # Convert values explicitly to standard python float types
        return [float(x) for x in embedding]