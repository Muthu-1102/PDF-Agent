"""
Thin wrapper around an embedded (local, file-persisted) ChromaDB instance.

Deliberately NOT a networked/routed DB service -- this project's focus is
the agentic retrieval + LangGraph orchestration, not infra plumbing.
"""

import chromadb
from chromadb.utils import embedding_functions

from app.core.config import settings

_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)

_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=settings.EMBEDDING_MODEL
)

_collection = None


def get_collection():
    global _collection
    if _collection is None:
        _collection = _client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION,
            embedding_function=_embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection
