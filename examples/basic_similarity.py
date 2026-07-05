from ContextVector.core import ContextVector
from ContextVector.providers.ollama import OllamaProvider

if __name__ == "__main__":
    print(" Running Sequence Protocol & Serialization Test Bench...")
    provider = OllamaProvider()
    target_model = "nomic-embed-text"
    
    cv_original = ContextVector.from_text(
        text="Structural equality validation ensures object parity.",
        provider=provider,
        model_name=target_model,
        metadata={"version": 1.1}
    )
    
    payload_map = cv_original.to_dict()
    cv_reconstituted = ContextVector.from_dict(payload_map)
    
    # Showcase the new __repr__ format layout
    print(f"\nInitialized Object Instance: {cv_original}")
    
    # Showcase container feature set utilities
    print("\n Testing Sequence Capabilities:")
    print(f" -> Dimension Length (__len__)   : {len(cv_original)}")
    print(f" -> Read First Coordinate ([0])  : {cv_original[0]}")
    print(f" -> Value Search Checking (in)   : {cv_original[0] in cv_original}")
    
    print("\n Running Parity Checks:")
    print(f" -> Memory Address Match (is) : {cv_original is cv_reconstituted}")
    print(f" -> Structural Data Match (==): {cv_original == cv_reconstituted}")