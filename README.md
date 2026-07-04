# ContextVector

> **A provider-agnostic embedding datatype for Python.**

ContextVector is a lightweight Python library that introduces a first-class data type for working with vector embeddings. Instead of treating embeddings as anonymous lists of floating-point numbers, ContextVector binds them together with their original payload, modality, metadata, and native vector operations.

The library is designed around a simple idea:

> **Embeddings should behave like structured objects, not raw arrays.**

---

## Features

- Immutable embedding storage
- Provider-independent embedding generation
- Built-in cosine similarity
- Native serialization and deserialization
- Structural equality comparison
- Lightweight architecture
- NumPy-powered vector operations
- Easily extensible provider interface

---

## Why ContextVector?

Most embedding workflows produce something like this:

```python
embedding = [0.12, -0.44, 0.98, ...]
```

After generation, developers must manually manage:

- the original text
- metadata
- embedding dimensions
- similarity calculations
- serialization
- storage

ContextVector encapsulates everything into a single object.

```python
ContextVector
├── embedding
├── payload
├── modality
└── metadata
```

This makes embeddings easier to manipulate, serialize, compare, and store.

---

# Design Principles

## Immutable Embeddings

Embedding vectors become read-only immediately after construction.

```python
self._embedding.flags.writeable = False
```

This prevents accidental modification during downstream processing.

---

## Provider Independence

ContextVector does not know how embeddings are generated.

Instead, every embedding engine implements the same interface.

```
EmbeddingProvider
        │
        ├── OllamaProvider
        ├── OpenAIProvider
        ├── HuggingFaceProvider
        └── CustomProvider
```

The core library never needs modification when adding new providers.

---

## Native Vector Operations

Operations such as cosine similarity belong to the object itself.

```python
score = vector1.similarity(vector2)
```

No external machine learning library is required.

---

## Architecture

```text
                Text
                 │
                 ▼
        EmbeddingProvider
                 │
        +----------------+
        | OllamaProvider |
        +----------------+
                 │
                 ▼
          Vector Embedding
                 │
                 ▼
          ContextVector
        ┌──────────────┐
        │ embedding    │
        │ payload      │
        │ modality     │
        │ metadata     │
        └──────────────┘
                 │
                 ▼
     Similarity • Serialization
     Equality   • Persistence
```

---

# Project Structure

```text
ContextVector/
│
├── core.py
├── interfaces.py
├── main.py
│
└── providers/
    ├── __init__.py
    └── ollama.py
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ContextVector.git
cd ContextVector
```

Install dependencies

```bash
pip install -r requirements.txt
```

Download an embedding model for Ollama

```bash
ollama pull nomic-embed-text
```

---

# Quick Start

```python
from core import ContextVector
from providers.ollama import OllamaProvider

provider = OllamaProvider()

model = "nomic-embed-text"

cv1 = ContextVector.from_text(
    "Artificial intelligence architecture.",
    provider,
    model
)

cv2 = ContextVector.from_text(
    "Deep neural networks.",
    provider,
    model
)

score = cv1.similarity(cv2)

print(score)
```

---

# Creating a ContextVector

```python
cv = ContextVector.from_text(
    text="Hello World",
    provider=provider,
    model_name="nomic-embed-text",
    metadata={
        "source": "documentation",
        "language": "english"
    }
)
```

---

# Accessing Properties

```python
print(cv.embedding)

print(cv.payload)

print(cv.modality)

print(cv.metadata)

print(cv.dimensions)
```

---

# Similarity

Cosine similarity is computed directly by the object.

```python
similarity = cv1.similarity(cv2)
```

The implementation computes

\[
Similarity =
\frac{A \cdot B}
{\|A\|\|B\|}
\]

Values range from

```
-1.0 ← opposite direction

 0.0 ← orthogonal

+1.0 ← identical direction
```

Dimension mismatches automatically raise a `ValueError`.

---

# Serialization

Convert a ContextVector into a JSON-compatible dictionary.

```python
payload = cv.to_dict()
```

Example

```python
{
    "embedding": [...],
    "payload": "...",
    "modality": "text",
    "metadata": {...}
}
```

---

# Deserialization

Restore an object from serialized data.

```python
restored = ContextVector.from_dict(payload)
```

---

# Structural Equality

Unlike normal Python objects, ContextVector compares actual content.

```python
cv1 == cv2
```

The comparison checks

- modality
- payload
- embedding values

rather than memory addresses.

---

# Embedding Providers

Every provider inherits from

```python
EmbeddingProvider
```

The only required method is

```python
embed_text(
    text: str,
    model_name: str
) -> List[float]
```

This allows developers to integrate any embedding backend.

Example:

- Ollama
- OpenAI
- Hugging Face
- Cohere
- SentenceTransformers

without modifying ContextVector.

---

# Example Provider

```python
class MyProvider(EmbeddingProvider):

    def embed_text(
        self,
        text,
        model_name
    ):
        ...
```

---

# Example Workflow

```python
provider = OllamaProvider()

cv = ContextVector.from_text(
    "Context vectors are useful.",
    provider,
    "nomic-embed-text"
)

payload = cv.to_dict()

restored = ContextVector.from_dict(payload)

assert cv == restored
```

---

# Current Capabilities

- Text embeddings
- Immutable vectors
- Cosine similarity
- Serialization
- Deserialization
- Structural equality
- Metadata support
- Provider abstraction

---

# Future Roadmap

- Image embeddings
- Audio embeddings
- Video embeddings
- Hugging Face provider
- OpenAI provider
- Cohere provider
- Vector arithmetic
- Distance metrics
- Batch embedding
- PyPI package
- Documentation website

---

# Contributing

Contributions are welcome.

Feel free to submit issues, feature requests, or pull requests.

---

# License

This project is licensed under the MIT License.
