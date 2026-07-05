import os
from typing import List
from ..interfaces import EmbeddingProvider

class OpenAIProvider(EmbeddingProvider):
    """
    Concrete implementation of EmbeddingProvider using the OpenAI cloud API.
    Requires the 'OPENAI_API_KEY' environment variable to be set.
    """
    def __init__(self, api_key: str = None):
        # Fall back to environment variable if key is not passed directly
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API Key not found. Pass it to the constructor or set OPENAI_API_KEY.")
        
        # Deferred import to keep core package lightweight if library isn't used
        import openai
        self.client = openai.OpenAI(api_key=self.api_key)

    def embed_text(self, text: str, model_name: str = "text-embedding-3-small") -> List[float]:
        """Dispatches text payload to OpenAI cloud embedding endpoints."""
        response = self.client.embeddings.create(input=[text], model=model_name)
        return response.data[0].embedding