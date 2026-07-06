# ContextVector

> **A first-class semantic data type for AI applications.**

ContextVector is an open-source Python library that introduces a unified data type for semantic AI systems. Instead of managing embeddings, payloads, metadata, and modality information as separate objects, ContextVector encapsulates them into a single immutable object.

The goal is to make semantic information a first-class citizen in software, enabling developers to build AI applications around a reusable, provider-agnostic semantic primitive rather than disconnected arrays and dictionaries.

---

## Why ContextVector?

Traditional programming languages provide primitive data types such as
`int`, `float`, `str`, `list`, and `dict`.

Modern AI applications, however, work with semantic information composed of multiple independent pieces:

- Embedding vectors
- Original payloads
- Metadata
- Modality information
- Provenance

Different frameworks and vector databases provide their own representations of semantic data. ContextVector aims to provide a lightweight, framework-agnostic representation that can be used consistently across AI pipelines.

It groups semantic information into a single immutable object that can be compared, serialized, validated, and transported throughout an application.

Instead of writing:

```python
embedding = provider.embed(text)

payload = text

metadata = {
    "source": "documentation"
}

modality = "text"
```

developers can simply create:

```python
cv = ContextVector(...)
```

The resulting object represents a complete semantic entity that can be compared, serialized, stored, and transported throughout an application.

---

# Features

-   Immutable embedding storage
-   Provider-agnostic architecture
-   Provenance-aware vectors
-   Safe cosine similarity with provenance validation
-   Integrity-checked serialization/deserialization
-   Structural equality comparison
-   Python sequence protocol support (`len`, indexing, iteration, membership)
-   Multiple embedding providers
-   Lightweight NumPy implementation
-   Extensible provider interface

---

# Installation

Clone the repository

```bash
git clone https://github.com/Amberss-Log/ContextVector.git

cd ContextVector
```

Install the package

```bash
pip install -e .
```

or install the required dependencies

```bash
pip install -r requirements.txt
```

---

# Supported Providers

ContextVector is intentionally independent of any specific embedding model.

Currently supported providers include:

- Ollama
- Hugging Face Sentence Transformers
- OpenAI Embeddings

Additional providers can be added by implementing the `EmbeddingProvider` interface.

---

## Provider Setup

### Ollama

Install Ollama and download an embedding model.

```bash
ollama pull nomic-embed-text
```

---

### Hugging Face

Models download automatically on first use.

Example model:

```
all-MiniLM-L6-v2
```

---

### OpenAI

Set your API key before creating the provider.

Linux / macOS

```bash
export OPENAI_API_KEY="your-api-key"
```

Windows PowerShell

```powershell
$env:OPENAI_API_KEY="your-api-key"
```

---

# Quick Start

```python
from ContextVector.core import ContextVector
from ContextVector.providers.ollama import OllamaProvider

provider = OllamaProvider()

cv1 = ContextVector.from_text(
    text="Artificial Intelligence",
    provider=provider,
    model_name="nomic-embed-text"
)

cv2 = ContextVector.from_text(
    text="Machine Learning",
    provider=provider,
    model_name="nomic-embed-text"
)

similarity = cv1.similarity(cv2)

print(similarity)
```

---

# Creating a ContextVector

The recommended way to create objects is through the factory method.

```python
cv = ContextVector.from_text(
    text="Hello World",
    provider=provider,
    model_name="nomic-embed-text",
    metadata={
        "source": "example"
    }
)
```

---

# Working with ContextVectors

## Dimensions

```python
print(cv.dimensions)
```

---

## Access the embedding

```python
embedding = cv.embedding
```

---

## Sequence operations

```python
len(cv)

cv[0]

for value in cv:
    print(value)

0.25 in cv
```

---

## Similarity

Compute cosine similarity directly between two ContextVector objects. Similarity requires matching dimensions, provider, and model.


```python
score = cv1.similarity(cv2)
```

---

## Serialization

Convert a ContextVector into a JSON-compatible dictionary.

```python
data = cv.to_dict()
```

---

## Deserialization

Reconstruct a ContextVector from stored data.

```python
cv = ContextVector.from_dict(data)
```

---

## Structural Equality

Unlike ordinary Python objects, ContextVector compares stored semantic data instead of object identity.

```python
cv1 == cv2
```

returns `True` only if

- embeddings are identical
- payloads match
- modalities match
- provider names match
- model names match

---
## Safety Guarantees

-   Rejects empty embeddings
-   Rejects NaN and Inf values
-   Embeddings are immutable
-   Validates serialized payloads
-   Prevents cross-model similarity comparisons

# Provider Architecture

ContextVector separates the semantic data type from the embedding engine.

Every provider simply implements one interface.

```python
class EmbeddingProvider(ABC):

    @abstractmethod
    def embed_text(self, text, model_name):
        ...
```

This allows applications to switch between providers without changing how ContextVector objects are created or used.

For example:

```python
provider = OllamaProvider()
```

can later become

```python
provider = HuggingFaceProvider()
```

or

```python
provider = OpenAIProvider()
```

while the rest of the application remains unchanged.

---

# Project Structure

```
ContextVector/

├── ContextVector/
│   ├── core.py
│   ├── interfaces.py
│   ├── providers/
│   │   ├── huggingface.py
│   │   ├── ollama.py
│   │   └── openai.py
│
├── examples/
│   ├── basic_similarity.py
│   └── multi_provider_demo.py
│
├── tests/
│   └── test_core.py
│
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```

---

# Design Principles

## Immutable by Design

Embedding vectors become read-only immediately after construction, preventing accidental modification during downstream processing.

---

## Provider Agnostic

The core data type has no dependency on a particular embedding model.

Embeddings can be generated locally or through cloud APIs while using the exact same ContextVector interface.

---

## Native Operations

Similarity computation belongs to the object itself.

Developers do not need external helper functions simply to compare vectors.

---

## Serializable

ContextVector objects can be safely stored, transmitted over networks, or persisted in databases using native Python dictionaries.

---

# Running the Example

```bash
python examples/basic_similarity.py
```

---

# Running Tests

```bash
python -m unittest discover tests
```

---

# Roadmap

Future releases are planned to include:

-   Image and audio support
-   Vector arithmetic
-   Additional similarity metrics
-   Vector database adapters
-   Performance benchmarks
-   Expanded documentation

---

# Contributing

Contributions are welcome through issues and pull requests.

---

# License

This project is licensed under the Apache License 2.0.

See the [LICENSE](LICENSE) file for details.
