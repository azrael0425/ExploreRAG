"""Pre-download sentence-transformers models for offline use.

Usage:
    python backend/scripts/download_models.py

Environment variables (optional):
    EXPLORERAG_EMBEDDING_MODEL  — default: BAAI/bge-m3
    EXPLORERAG_RERANKER_MODEL   — default: BAAI/bge-reranker-v2-m3
"""
import os
import sys


def download_models():
    embedding_model = os.environ.get("EXPLORERAG_EMBEDDING_MODEL", "BAAI/bge-m3")
    from sentence_transformers import SentenceTransformer

    print(f"Downloading embedding model: {embedding_model}")
    SentenceTransformer(embedding_model)
    print("BGE-M3 downloaded successfully.")


if __name__ == "__main__":
    download_models()
