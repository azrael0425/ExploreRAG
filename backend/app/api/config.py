"""
Config status endpoint — expose active LLM/embedding provider info to frontend.
"""
from fastapi import APIRouter

from app.core.config import settings

router = APIRouter(prefix="/config", tags=["config"])


@router.get("/status")
async def get_config_status():
    """Return active provider and model names for UI display."""
    kg_provider = settings.KG_EMBEDDING_PROVIDER.lower()
    kg_model = settings.KG_EMBEDDING_MODEL

    return {
        "llm_provider": "qwen",
        "llm_model": settings.LLM_MODEL_FAST,
        "kg_embedding_provider": kg_provider,
        "kg_embedding_model": kg_model,
        "kg_embedding_dimension": settings.KG_EMBEDDING_DIMENSION,
        "explorerag_embedding_model": settings.EXPLORERAG_EMBEDDING_MODEL,
        "explorerag_reranker_model": settings.EXPLORERAG_RERANKER_MODEL,
    }
