"""
Core RAG implementation using llama_index for web document processing
"""

import os
from typing import List, Any
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    PromptTemplate,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.readers.web import SimpleWebPageReader


class RAG:
    """
    Retrieval-Augmented Generation system for processing web documents.

    This class implements a RAG pipeline that:
    1. Loads documents from web URLs
    2. Chunks and indexes them using local embeddings
    3. Queries the index using a Hugging Face LLM
    4. Returns context-aware responses

    Args:
        urls (List[str]): List of web URLs to process
        **kwargs: Additional configuration parameters

    Example:
        >>> rag = RAG(["https://example.com"])
        >>> response = rag("What is this page about?")
    """

    def __init__(self, urls: List[str], **kwargs: Any) -> None:
        # Config
        self.model = kwargs.get("model", "Qwen/Qwen3-235B-A22B")
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens", 128)
        self.chunk_size = kwargs.get("chunk_size", 512)
        self.chunk_overlap = kwargs.get("chunk_overlap", 64)
        self.similarity_top_k = kwargs.get("similarity_top_k", 5)

        # Get token from environment or kwargs
        token = kwargs.get("token") or os.getenv("HUGGING_FACE_TOKEN")
        if not token:
            raise ValueError(
                "Hugging Face token is required. "
                "Set HUGGING_FACE_TOKEN environment variable or pass token parameter."
            )

        # Initialize LLM
        self.llm = HuggingFaceInferenceAPI(
            model_name=self.model,
            token=token,
            provider="auto",
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Use local embeddings
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Set global parameters
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.chunk_size = self.chunk_size
        Settings.chunk_overlap = self.chunk_overlap

        # Load web page
        self.documents = SimpleWebPageReader(html_to_text=True).load_data(urls)

        # Parsing documents
        node_parser = SimpleNodeParser.from_defaults(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        self.nodes = node_parser.get_nodes_from_documents(self.documents)

        # Create index
        self.index = VectorStoreIndex(self.nodes)

        # Custom prompt for better inference
        qa_prompt = PromptTemplate(
            "Context information below:\n"
            "----------------------------\n"
            "{context_str}\n"
            "----------------------------\n"
            "Answer the question using only the information provided: {query_str}\n"
            "If you don't have enough information, say: 'I can't answer based on the documents provided.'\n"
            "Answer:"
        )

        # Create query engine
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=self.similarity_top_k,
            text_qa_template=qa_prompt,
            verbose=kwargs.get("verbose", False),
        )

    def query(self, query: str) -> str:
        """
        Query the RAG system with a question.

        Args:
            query (str): The question to ask

        Returns:
            str: The generated response
        """
        response = self.query_engine.query(query)
        return str(response)

    def __call__(self, query: str) -> str:
        """Alias for query method."""
        return self.query(query)

    def get_document_count(self) -> int:
        """Get the number of loaded documents."""
        return len(self.documents)

    def get_node_count(self) -> int:
        """Get the number of document chunks/nodes."""
        return len(self.nodes)
