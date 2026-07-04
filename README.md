# ContextVector

ContextVector is a lightweight, decoupled Python primitive data type designed for multimodal AI applications. It provides a structured wrapper that binds raw payloads, modalities, and high-dimensional vector embeddings together into a single, immutable object with native mathematical capabilities.

## Features

- **Model Agnostic:** Decoupled from specific ML frameworks via an abstract provider interface. Works natively with local engines (Ollama) or cloud APIs (OpenAI, Hugging Face).
- **Immutability:** Vector arrays are frozen upon initialization using NumPy flags to ensure mathematical data integrity at runtime.
- **Native Similarity Math:** Computes cosine similarity directly between instances without requiring external pipeline processing libraries.
- **Built-in Serialization:** Clean translation to and from JSON-compliant dictionaries for database persistence.
- **Operator Overloading:** Implements structural equality (`==`) checks out of the box.

## Project Structure

```text
ContextVector/
│
├── interfaces.py       # Abstract provider contracts
├── core.py             # Core ContextVector primitive implementation
├── main.py             # Operational test bench
│
└── providers/          # ML engine integrations
    ├── __init__.py     
    └── ollama.py       # Ollama integration wrapper

# Installation

Clone the repository and navigate to the project root:

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/ContextVector.git
cd ContextVector
```

Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

Ensure you have Ollama installed and running locally with the embedding model downloaded:

```bash
ollama pull nomic-embed-text
```

# Quick Start

```python
from core import ContextVector
from providers.ollama import OllamaProvider

# Initialize the local embedding model engine
provider = OllamaProvider()
model = "nomic-embed-text"

# Create context vectors directly from text strings
cv1 = ContextVector.from_text("Artificial intelligence architecture.", provider, model)
cv2 = ContextVector.from_text("Deep neural networks.", provider, model)

# Compute native similarity metrics directly between objects
score = cv1.similarity(cv2)
print(f"Semantic Alignment: {score:.4f}")

# Serialize for database storage or network transmission
data_payload = cv1.to_dict()

# Reconstitute a live instance from raw dictionary data
cv_recovered = ContextVector.from_dict(data_payload)

# Verify structural data parity
print(f"Objects are identical: {cv1 == cv_recovered}")
```
