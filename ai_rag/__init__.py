"""
AI-RAG: Retrieval-Augmented Generation System for web documents
"""

from .core import RAG
from .config import get_settings

__version__ = "0.1.0"
__all__ = ["RAG", "get_settings"]
