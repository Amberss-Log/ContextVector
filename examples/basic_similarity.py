import json
from ContextVector.core import ContextVector
from ContextVector.providers.ollama import OllamaProvider

if __name__ == "__main__":
    print(" Running Structural Equality & Serialization Test...")
    provider = OllamaProvider()
    target_model = "nomic-embed-text"
    
    # Generate the base vector instance
    cv_original = ContextVector.from_text(
        text="Structural equality validation ensures object parity.",
        provider=provider,
        model_name=target_model,
        metadata={"version": 1.0}
    )
    
    # Convert to dictionary and back to simulate a storage cycle
    payload_map = cv_original.to_dict()
    cv_reconstituted = ContextVector.from_dict(payload_map)
    
    print(f"\nOriginal:      {cv_original}")
    print(f"Reconstituted: {cv_reconstituted}")
    
    # Evaluate identity vs structural parity
    print("\n Running Parity Checks:")
    print(f" -> Memory Address Match (is) : {cv_original is cv_reconstituted}")
    print(f" -> Structural Data Match (==): {cv_original == cv_reconstituted}")
    
    # Defensive check against a completely different vector
    cv_different = ContextVector.from_text("Apples and oranges.", provider, target_model)
    print(f" -> Distinct Objects Check (==): {cv_original == cv_different}")