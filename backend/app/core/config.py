from typing import Literal

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from pathlib import Path

# Find .env file - check project root first, fallback for Docker
_candidate = Path(__file__).resolve().parent.parent.parent.parent / ".env"
ENV_FILE = str(_candidate) if _candidate.exists() else ".env"


class Settings(BaseSettings):
    # App
    APP_NAME: str = "ExploreRAG"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"

    # Source identity injected by the reproducible container build helper.
    EXPLORERAG_BUILD_GIT_COMMIT: str = ""
    EXPLORERAG_BUILD_GIT_BRANCH: str = ""
    EXPLORERAG_BUILD_GIT_DIRTY: str = "unknown"

    # Base directory (backend folder)
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent

    # Database
    # No repository-level credential fallback: every runtime must opt in to a
    # database explicitly through .env or its process environment.
    DATABASE_URL: str

    # Qwen / Alibaba Cloud Model Studio (DashScope OpenAI-compatible API)
    DASHSCOPE_API_KEY: str = Field(default="")
    DASHSCOPE_BASE_URL: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    # Text generation can use either DashScope or the DeepSeek
    # OpenAI-compatible endpoint.  DashScope settings are retained for the
    # Qwen vision model when text generation is switched to DeepSeek.
    LLM_PROVIDER: Literal["dashscope", "deepseek"] = Field(default="dashscope")
    DEEPSEEK_API_KEY: str = Field(default="")
    DEEPSEEK_BASE_URL: str = Field(default="https://api.deepseek.com/v1")
    QWEN_VISION_MODEL: str = Field(default="qwen-vl-plus")
    QWEN_ENABLE_THINKING: bool = Field(default=False)

    # Local OpenAI-compatible multimodal server (Ollama by default).  Explicit
    # local mode never falls back to the cloud, so a local outage is visible.
    LOCAL_LLM_BASE_URL: str = Field(default="http://ollama:11434/v1")
    LOCAL_LLM_NATIVE_BASE_URL: str = Field(default="http://ollama:11434")
    LOCAL_LLM_API_KEY: str = Field(default="ollama")
    LOCAL_LLM_MODEL: str = Field(default="qwen3-vl:4b-instruct")
    LOCAL_LLM_VISION_MODEL: str = Field(default="qwen3-vl:4b-instruct")
    LOCAL_LLM_ENABLE_THINKING: bool = Field(default=False)
    LOCAL_LLM_CONTEXT_WINDOW: int = Field(default=8192, ge=2048, le=262144)
    LOCAL_LLM_MAX_OUTPUT_TOKENS: int = Field(default=2048, ge=256, le=32768)
    LOCAL_LLM_MAX_CONCURRENCY: int = Field(default=1, ge=1, le=4)
    LOCAL_LLM_HEALTH_TIMEOUT_SECONDS: float = Field(default=3.0, gt=0, le=30)
    LOCAL_LLM_REQUEST_TIMEOUT_SECONDS: float = Field(default=600.0, gt=0, le=3600)

    # Main text model for chat and knowledge-graph extraction.
    LLM_MODEL_FAST: str = Field(default="qwen-plus")

    # Ragas uses structured claim extraction and may need a judge model that
    # differs from the production answer model.  ``workspace`` preserves the
    # original behavior; the explicit providers still use the existing
    # OpenAI-compatible credentials and never persist API keys in snapshots.
    EVALUATION_LLM_PROVIDER: Literal["workspace", "dashscope", "deepseek"] = Field(
        default="workspace"
    )
    EVALUATION_LLM_MODEL: str = Field(default="")
    EVALUATION_LLM_MAX_TOKENS: int = Field(default=8192, ge=1024, le=16384)
    EVALUATION_LLM_MAX_ATTEMPTS: int = Field(default=3, ge=1, le=5)

    # Max output tokens for Qwen chat responses.
    LLM_MAX_OUTPUT_TOKENS: int = Field(default=8192)

    # BGE-M3 is mounted at this container path and serves Chinese and English.
    KG_EMBEDDING_PROVIDER: str = Field(default="sentence_transformers")
    KG_EMBEDDING_MODEL: str = Field(default="/models/bge-m3")
    KG_EMBEDDING_DIMENSION: int = Field(default=1024)
    KG_EMBEDDING_DEVICE: str = Field(default="cuda")

    # ChromaDB
    CHROMA_HOST: str = Field(default="localhost")
    CHROMA_PORT: int = Field(default=8002)

    # ExploreRAG Pipeline (orchestrated exclusively through LangChain/LCEL)
    # Keep callback telemetry local by default.  No LangSmith credentials or
    # document contents are sent outside the deployment.
    EXPLORERAG_LANGCHAIN_LOCAL_CALLBACKS: bool = True
    EXPLORERAG_ENABLE_KG: bool = True
    EXPLORERAG_ENABLE_RERANKER: bool = False
    EXPLORERAG_ENABLE_IMAGE_EXTRACTION: bool = True
    EXPLORERAG_ENABLE_IMAGE_CAPTIONING: bool = True
    EXPLORERAG_ENABLE_TABLE_CAPTIONING: bool = True
    EXPLORERAG_MAX_TABLE_MARKDOWN_CHARS: int = 8000
    EXPLORERAG_CHUNK_MAX_TOKENS: int = 512
    EXPLORERAG_KG_QUERY_TIMEOUT: float = 30.0
    # Query-time graph augmentation must fail open quickly enough that an
    # unavailable graph never stalls an otherwise healthy vector RAG answer.
    EXPLORERAG_KG_AUGMENTATION_TIMEOUT_SECONDS: float = 10.0
    EXPLORERAG_KG_AUGMENTATION_MAX_CHARS: int = 6000
    EXPLORERAG_KG_AUGMENTATION_TOP_K: int = 10
    EXPLORERAG_KG_AUGMENTATION_CHUNK_TOP_K: int = 5
    EXPLORERAG_KG_CHUNK_TOKEN_SIZE: int = 1200
    # Local BGE-M3 on CPU needs smaller, serialized LightRAG embedding jobs.
    # The upstream defaults submit many jobs concurrently and can exceed its
    # per-worker timeout during graph indexing.
    EXPLORERAG_KG_EMBEDDING_BATCH_SIZE: int = 4
    EXPLORERAG_KG_EMBEDDING_MAX_ASYNC: int = 1
    EXPLORERAG_KG_EMBEDDING_TIMEOUT: int = 180
    EXPLORERAG_KG_LANGUAGE: str = "Chinese"
    EXPLORERAG_KG_ENTITY_TYPES: list[str] = [
        "Organization", "Person", "Product", "Location", "Event",
        "Financial_Metric", "Technology", "Date", "Regulation",
    ]
    EXPLORERAG_DEFAULT_QUERY_MODE: str = "hybrid"
    EXPLORERAG_DOCLING_IMAGES_SCALE: float = 2.0
    # Optional mounted Docling model cache for offline/default parsing.
    EXPLORERAG_DOCLING_ARTIFACTS_PATH: str = ""
    EXPLORERAG_MAX_IMAGES_PER_DOC: int = 50
    EXPLORERAG_ENABLE_FORMULA_ENRICHMENT: bool = True

    # Docling is the single supported document parser.
    EXPLORERAG_DOCUMENT_PARSER: str = "docling"

    # Processing timeout (minutes) — stale documents auto-recover to FAILED
    EXPLORERAG_PROCESSING_TIMEOUT_MINUTES: int = 10

    # Pre-ingestion Deduplication
    EXPLORERAG_DEDUP_ENABLED: bool = True
    EXPLORERAG_DEDUP_MIN_CHUNK_LENGTH: int = 50       # min meaningful chars
    EXPLORERAG_DEDUP_NEAR_THRESHOLD: float = 0.85     # Jaccard similarity cutoff

    # ExploreRAG Retrieval Quality
    EXPLORERAG_EMBEDDING_PROVIDER: str = "sentence_transformers"
    EXPLORERAG_EMBEDDING_MODEL: str = "/models/bge-m3"
    EXPLORERAG_EMBEDDING_DEVICE: str = "cuda"
    EXPLORERAG_RERANKER_MODEL: str = "/models/bge-reranker-v2-m3"
    EXPLORERAG_RERANKER_DEVICE: str = "cuda"
    EXPLORERAG_RERANKER_DTYPE: str = "float16"
    # Use a 768-token ceiling for small candidate sets and the 640-token tier
    # for a full candidate list to bound latency and VRAM.
    EXPLORERAG_RERANKER_MAX_LENGTH: int = Field(default=768, ge=64, le=2048)
    EXPLORERAG_RERANKER_LONG_LIST_MAX_LENGTH: int = Field(default=640, ge=64, le=2048)
    EXPLORERAG_RERANKER_LONG_LIST_THRESHOLD: int = Field(default=16, ge=1, le=100)
    EXPLORERAG_RERANKER_BATCH_SIZE: int = Field(default=4, ge=1, le=32)
    EXPLORERAG_RERANKER_TIMEOUT_SECONDS: float = Field(default=5.0, gt=0)
    EXPLORERAG_RERANKER_CIRCUIT_BREAKER_FAILURES: int = Field(default=1, ge=1)
    EXPLORERAG_RERANKER_CIRCUIT_BREAKER_RECOVERY_SECONDS: float = Field(default=60.0, gt=0)
    EXPLORERAG_MODEL_WARMUP: bool = True
    EXPLORERAG_MODEL_WARMUP_TIMEOUT_SECONDS: float = Field(default=120.0, gt=0)
    # Deprecated compatibility fields. Chat retrieval uses the KB-prefixed
    # settings below so there is one unambiguous source of truth.
    EXPLORERAG_VECTOR_PREFETCH: int = 20
    EXPLORERAG_RERANKER_TOP_K: int = 8
    EXPLORERAG_MIN_RELEVANCE_SCORE: float = 0.15

    # Chat attachments are deliberately isolated from the durable knowledge
    # base.  These limits protect the single-worker/local-CPU deployment and
    # are configurable so operators can tune them to their model capacity.
    CHAT_ATTACHMENT_MAX_COUNT: int = 8
    CHAT_ATTACHMENT_MAX_FILE_SIZE: int = 25 * 1024 * 1024
    CHAT_ATTACHMENT_MAX_PDF_PAGES: int = 300
    CHAT_ATTACHMENT_MAX_ZIP_ENTRIES: int = 4_000
    CHAT_ATTACHMENT_MAX_ZIP_UNCOMPRESSED_SIZE: int = 150 * 1024 * 1024
    CHAT_ATTACHMENT_MAX_ZIP_RATIO: int = 100
    CHAT_ATTACHMENT_MAX_IMAGES: int = 50
    CHAT_ATTACHMENT_MAX_TOTAL_PIXELS: int = 80_000_000
    CHAT_ATTACHMENT_MAX_CHUNKS: int = 2_000
    CHAT_ATTACHMENT_CONTEXT_WINDOW_TOKENS: int = 32_768
    CHAT_ATTACHMENT_CONTEXT_SAFETY_TOKENS: int = 2_048
    CHAT_ATTACHMENT_DIRECT_MIN_TOKENS: int = 1_024
    CHAT_ATTACHMENT_PREFETCH: int = 20
    CHAT_ATTACHMENT_RERANK_TOP_K: int = 4
    EXPLORERAG_KB_PREFETCH: int = Field(default=20, ge=4, le=100)
    EXPLORERAG_KB_RERANK_TOP_K: int = 4
    CHAT_ATTACHMENT_MAX_VISION_IMAGES: int = 3
    CHAT_ATTACHMENT_PAGE_IMAGE_SCALE: float = 1.5
    CHAT_ATTACHMENT_MAX_PAGE_IMAGES: int = 24
    CHAT_ATTACHMENT_SCAN_TEXT_CHARS_PER_PAGE: int = 40
    CHAT_ATTACHMENT_SCAN_LOW_CONFIDENCE_CHARS: int = 20
    CHAT_ATTACHMENT_EAGER_INDEX_TOKENS: int = Field(default=12_000, ge=1_024)
    CHAT_ATTACHMENT_PREPARE_TIMEOUT_SECONDS: float = Field(default=600.0, gt=0)

    # Indexing controls for an 8 GB local GPU. The token guard is applied
    # after image/table enrichment, where Docling chunks can otherwise grow far
    # beyond the HybridChunker's original limit.
    EXPLORERAG_EMBEDDING_BATCH_SIZE: int = Field(default=4, ge=1, le=32)
    EXPLORERAG_EMBEDDING_MAX_TOKENS: int = Field(default=512, ge=64, le=8192)
    EXPLORERAG_EMBEDDING_TOKEN_OVERLAP: int = Field(default=32, ge=0, le=512)
    EXPLORERAG_CAPTION_MAX_WORKERS: int = Field(default=3, ge=1, le=8)
    EXPLORERAG_CAPTION_CACHE_SIZE: int = Field(default=2_048, ge=0, le=100_000)

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:5174", "http://localhost:3000"]

    model_config = {
        "env_file": str(ENV_FILE),
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
