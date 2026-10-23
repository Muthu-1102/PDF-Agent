"""
Isolated ChromaDB index builder for evaluation.
Ensures zero interaction with the production/default chroma store.
"""

import os
import sys
import uuid
from datetime import datetime
from typing import Dict, Any, Tuple

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(BASE_DIR, "backend"))

import chromadb
from chromadb.utils import embedding_functions

from app.core.config import settings
from app.ingestion.loader import load_document
from app.ingestion.chunker import chunk_document

STANDARD_DOCS = [
    "vrcc_travel_policy_v1.pdf",
    "vrcc_travel_policy_v2.pdf",
    "aethelgard_titan_drone_specs.pdf",
    "zephyros_avionics_handbook.pdf",
    "nyx_security_incident_response.docx",
    "project_chronos_research.txt",
    "aurora_quantum_compiler.pdf",
    "orion_fleet_logistics.docx"
]


def build_evaluation_index(persist_dir: str) -> Tuple[chromadb.PersistentClient, Any]:
    """
    Creates an isolated persistent ChromaDB instance at persist_dir,
    loads the standard synthetic corpus, chunks it, and indexes all chunks.
    """
    client = chromadb.PersistentClient(path=persist_dir)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.EMBEDDING_MODEL
    )
    collection = client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

    docs_dir = os.path.join(os.path.dirname(__file__), "..", "dataset", "documents")
    total_indexed = 0

    for fname in STANDARD_DOCS:
        fpath = os.path.join(docs_dir, fname)
        ftype = fname.split(".")[-1].lower()
        pages = load_document(fpath, ftype)
        chunks = chunk_document(pages)

        if not chunks:
            continue

        doc_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        ids, docs, metadatas = [], [], []

        for idx, (page_num, chunk_text) in enumerate(chunks):
            # Document and chunk metadata matching routes_upload.py
            ids.append(f"{doc_id}_{idx}")
            docs.append(chunk_text)
            metadatas.append({
                "document_id": doc_id,
                "document_name": fname,
                "chunk_index": idx,
                "page_number": page_num,
                "upload_timestamp": timestamp
            })

        collection.add(ids=ids, documents=docs, metadatas=metadatas)
        total_indexed += len(chunks)

    return client, collection
