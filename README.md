# AI-RAG: Web Document Q&A System

A Retrieval-Augmented Generation system for querying web documents using local embeddings and Hugging Face LLMs.

## Features

- Load documents from web URLs
- Local embedding generation with sentence-transformers
- Query with Hugging Face LLMs (Qwen, Llama, Mistral, etc.)
- Context-aware responses with source citation
- Fast indexing and retrieval

## Installation

```bash
# Clone repository
git clone https://github.com/ruslanprashchurovich/ai-rag.git
cd ai-rag

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your Hugging Face token

# Python example
python example_usage.py

# cli.py
python -m ai_rag.cli https://en.wikipedia.org/wiki/Python_(programming_language) -i
```

## Quick Start

```python
from ai_rag import RAG

# Initialize with web URLs
rag = RAG([
    "https://en.wikipedia.org/wiki/Python_(programming_language)",
    "https://en.wikipedia.org/wiki/C%2B%2B",
])

# Query the system
response = rag("Which language has better performance?")
print(response)
```

## Configuration

Create a `.env` file:

```env
HUGGING_FACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

Or pass parameters:

```python
rag = RAG(
    urls=["https://example.com"],
    model="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.3,
    chunk_size=1024,
    similarity_top_k=3,
    token="your-token-here"  # Optional if set in .env
)
```

## Examples

See `example_usage.py` for complete examples including:

- Basic Q&A from multiple sources
- Programming language comparison
- Technical documentation queries

## API Reference

### `RAG(urls, **kwargs)`

**Parameters:**

- `urls`: List of web URLs to process
- `model`: Hugging Face model name (default: "Qwen/Qwen3-235B-A22B")
- `temperature`: Generation temperature (default: 0.7)
- `max_tokens`: Maximum tokens in response (default: 128)
- `chunk_size`: Document chunk size (default: 512)
- `chunk_overlap`: Chunk overlap (default: 64)
- `similarity_top_k`: Number of chunks to retrieve (default: 5)
- `token`: Hugging Face token (optional, uses env var)
- `verbose`: Show query details (default: False)

**Methods:**

- `query(question)`: Get answer to question
- `__call__(question)`: Same as query()
- `get_document_count()`: Number of loaded documents
- `get_node_count()`: Number of document chunks

## License

MIT
