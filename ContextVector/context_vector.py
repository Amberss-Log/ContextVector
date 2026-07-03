import numpy as np
from typing import Any, Union, Dict

class ContextVector:
    """
    ContextVector: A novel unified data type for emerging multimodal AI technologies.
    Binds high-dimensional vector embeddings directly to their source context and modalities.
    """
    
    def __init__(self, embedding: Union[list, np.ndarray], payload: Any, modality: str = "text", metadata: Dict = None):
        if not isinstance(embedding, (list, np.ndarray)):
            raise TypeError("Embedding must be a list of numbers or a NumPy ndarray.")
            
        self.embedding = np.array(embedding, dtype=np.float32)
        

        self.payload = payload
        self.modality = modality.lower()
        self.metadata = metadata if metadata is not None else {}
        
    def __repr__(self) -> str:
        """Defines how the data type visually appears when printed in the terminal."""
        dimensions = self.embedding.shape[0]
        return f"<ContextVector | Modality: {self.modality.upper()} | Dimensions: {dimensions}>"
