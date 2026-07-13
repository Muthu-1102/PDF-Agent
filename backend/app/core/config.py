import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # --- LLM ---
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # --- Embeddings ---
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Vector store (embedded Chroma, no separate DB service) ---
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_store")
    CHROMA_COLLECTION: str = "agentic_rag_kb"

    # --- Retrieval / Agent behavior ---
    DEFAULT_TOP_K: int = 4
    MAX_TOP_K: int = 12
    TOP_K_STEP: int = 4
    MAX_RETRIES: int = 3
    SIMILARITY_THRESHOLD: float = 0.35
    MIN_TRUSTED_CHUNKS: int = 2

    # --- Evidence Agent judging mode ---
    # "llm"  -> ask the LLM to judge relevance + sufficiency (default)
    # "similarity" -> pure cosine-similarity threshold (old behavior, used
    #                 as an automatic fallback if the LLM call/parse fails)
    EVIDENCE_JUDGE_MODE: str = "llm"
    EVIDENCE_CANDIDATE_FLOOR: float = 0.15  # loose pre-filter before LLM sees chunks

    # --- Chunking ---
    CHUNK_SIZE_WORDS: int = 500
    CHUNK_OVERLAP_WORDS: int = 80


settings = Settings()