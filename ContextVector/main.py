from core import ContextVector
from providers.ollama import OllamaProvider

if __name__ == "__main__":
    print(" Running Modular Architecture Playground...")
    
    provider = OllamaProvider()
    target_model = "nomic-embed-text"
    
    cv1 = ContextVector.from_text("Quantum computing pipelines.", provider, target_model)
    cv2 = ContextVector.from_text("High-performance server clusters.", provider, target_model)
    
    print(f"Vector 1: {cv1}")
    print(f"Vector 2: {cv2}")
    print(f"Cross-Module Similarity Score: {cv1.similarity(cv2):.4f}")