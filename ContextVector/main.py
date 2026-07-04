import json
from core import ContextVector
from providers.ollama import OllamaProvider

if __name__ == "__main__":
    print(" Running Serialization Playground...")
    provider = OllamaProvider()
    target_model = "nomic-embed-text"
    
    # 1. Create a live vector object via our automated factory
    original_cv = ContextVector.from_text(
        text="Data serialization enables network transmission pipeline stability.",
        provider=provider,
        model_name=target_model,
        metadata={"user_id": "tanis", "environment": "production"}
    )
    
    print(f"\n Step 1: Created Original Live Instance: {original_cv}")
    
    # 2. Serialize to Dictionary/JSON payload string (Simulating saving to disk)
    serialized_dict = original_cv.to_dict()
    json_string = json.dumps(serialized_dict, indent=2)
    
    print("\n Step 2: Serialized into Clean JSON String Format:")
    # Print just a snapshot of the output string so it doesn't flood the console
    print(json_string[:350] + "\n... [Remaining Vector Weights Truncated for Readability] ...\n}")
    
    # 3. Simulate a database reload (Convert back from raw JSON data map)
    loaded_data_map = json.loads(json_string)
    reconstituted_cv = ContextVector.from_dict(loaded_data_map)
    
    print(f"\n Step 3: Reconstituted from Raw JSON Data: {reconstituted_cv}")
    
    # 4. Math verification: Do they align perfectly?
    reconstituted_similarity = original_cv.similarity(reconstituted_cv)
    print(f" Security Validation Score (Original vs Reconstituted): {reconstituted_similarity:.4f}")