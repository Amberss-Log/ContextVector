import os
from ContextVector.core import ContextVector
from ContextVector.providers.ollama import OllamaProvider
from ContextVector.providers.huggingface import HuggingFaceProvider
from ContextVector.providers.openai import OpenAIProvider

if __name__ == "__main__":
    text_sample = "Decoupled architecture allows effortless engine swapping."
    print(" Initializing Multi-Provider Architecture Demo...\n")

    # -----------------------------------------------------------------
    # 1. Run Local Ollama Provider
    # -----------------------------------------------------------------
    print(" Processing with Local Ollama...")
    ollama_prov = OllamaProvider()
    cv_ollama = ContextVector.from_text(text_sample, ollama_prov, "nomic-embed-text")
    print(f"   Success: {cv_ollama}\n")

    # -----------------------------------------------------------------
    # 2. Run Local Hugging Face Provider (Sentence-Transformers)
    # -----------------------------------------------------------------
    print(" Processing with Local Hugging Face Model (Downloads on first run)...")
    hf_prov = HuggingFaceProvider()
    cv_hf = ContextVector.from_text(text_sample, hf_prov, "all-MiniLM-L6-v2")
    print(f"   Success: {cv_hf}\n")

    # -----------------------------------------------------------------
    # 3. Run Cloud OpenAI Provider (Requires API Key setup)
    # -----------------------------------------------------------------
    print(" Checking for OpenAI API credentials...")
    if os.getenv("OPENAI_API_KEY"):
        openai_prov = OpenAIProvider()
        cv_openai = ContextVector.from_text(text_sample, openai_prov, "text-embedding-3-small")
        print(f"   Success: {cv_openai}\n")
    else:
        print("   [Skipped] OpenAI API key variable not detected in environment.\n")

    print(" All initialized provider pipelines executed successfully!")