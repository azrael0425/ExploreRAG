"""
ExploreRAG — standalone Knowledge Base + RAG application.
"""
from contextlib import asynccontextmanager
from pathlib import Path
import asyncio
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import logging

from datetime import datetime, timedelta

from sqlalchemy import text, update

from app.core.config import settings
from app.core.database import engine, Base

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting ExploreRAG API...")
    from app.core.accelerator import validate_local_model_accelerators
    validate_local_model_accelerators()
    import os
    auto_create = os.environ.get("AUTO_CREATE_TABLES", "true").lower() == "true"
    if auto_create:
        async with engine.begin() as conn:
            # Check if tables already exist (e.g., alembic_version)
            result = await conn.execute(
                text("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'alembic_version');")
            )
            is_initialized = result.scalar()

            if not is_initialized:
                schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
                if os.path.exists(schema_path):
                    with open(schema_path, "r", encoding="utf-8") as f:
                        schema_sql = f.read()
                    
                    # Split and execute each statement to avoid asyncpg multi-statement issues
                    for statement in schema_sql.split(';'):
                        stmt = statement.strip()
                        if stmt:
                            await conn.execute(text(stmt))
                    logger.info("Database tables created from schema.sql")
                    
                    # Stamp the alembic version
                    await conn.execute(text("INSERT INTO public.alembic_version (version_num) VALUES ('c1e2a3b4d5e6') ON CONFLICT DO NOTHING;"))
                else:
                    await conn.run_sync(Base.metadata.create_all)
                    logger.info("Database tables created/verified (Base.metadata.create_all)")
            else:
                logger.info("Database is already initialized.")

        # The original application only used schema.sql on first boot.  Run
        # Alembic on subsequent boots as well so existing installations gain
        # the evaluation tables/feedback columns without a manual SQL step.
        def upgrade_database_schema() -> None:
            from alembic import command
            from alembic.config import Config

            config = Config(str(Path(__file__).resolve().parent.parent / "alembic.ini"))
            command.upgrade(config, "head")

        await asyncio.to_thread(upgrade_database_schema)

        # Recover stale processing documents (stuck from previous runs)
        from app.models.document import Document, DocumentStatus
        from sqlalchemy.ext.asyncio import AsyncSession
        from sqlalchemy import select as sa_select
        async with AsyncSession(engine) as session:
            timeout = settings.EXPLORERAG_PROCESSING_TIMEOUT_MINUTES
            cutoff = datetime.utcnow() - timedelta(minutes=timeout)
            stale_statuses = [
                DocumentStatus.PROCESSING,
                DocumentStatus.PARSING,
                DocumentStatus.INDEXING,
            ]
            result = await session.execute(
                update(Document)
                .where(
                    Document.status.in_(stale_statuses),
                    Document.updated_at < cutoff,
                )
                .values(
                    status=DocumentStatus.FAILED,
                    error_message=f"Processing timeout ({timeout}min). Click Analyze to retry.",
                )
                .returning(Document.id)
            )
            stale_ids = [row[0] for row in result.fetchall()]
            if stale_ids:
                await session.commit()
                logger.warning(f"Recovered {len(stale_ids)} stale documents: {stale_ids}")
        from app.evaluation.task_manager import recover_interrupted_runs
        recovered_runs = await recover_interrupted_runs()
        if recovered_runs:
            logger.warning("Marked %s interrupted evaluation runs as failed", recovered_runs)
    else:
        logger.info("AUTO_CREATE_TABLES=false — skipping auto-migration")
    if settings.EXPLORERAG_MODEL_WARMUP:
        async def warmup_model(name: str, warmup) -> None:
            started_at = time.perf_counter()
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(warmup),
                    timeout=settings.EXPLORERAG_MODEL_WARMUP_TIMEOUT_SECONDS,
                )
                logger.info(
                    "%s warmup completed in %.2fs",
                    name,
                    time.perf_counter() - started_at,
                )
            except Exception as exc:
                logger.warning("%s warmup failed; requests will use runtime fallback: %s", name, exc)

        from app.services.embedder import get_embedding_service
        await warmup_model("BGE embedding model", get_embedding_service().warmup)
        if settings.EXPLORERAG_ENABLE_RERANKER:
            from app.services.reranker import get_reranker_service
            reranker = get_reranker_service()
            await warmup_model("BGE reranker model", reranker.warmup)
    yield
    logger.info("Shutting down...")
    from app.services.knowledge_graph_service import cleanup_knowledge_graph_service_cache
    await cleanup_knowledge_graph_service_cache()
    await engine.dispose()


app = FastAPI(
    title=settings.APP_NAME,
    description="ExploreRAG — Knowledge base with hybrid retrieval, knowledge graph evidence, and grounded LLM chat",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    redirect_slashes=False,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/")
async def root():
    """Small landing response for direct browser visits to the API port."""
    return {
        "name": settings.APP_NAME,
        "status": "healthy",
        "health": "/health",
        "docs": "/docs",
        "api_prefix": settings.API_V1_PREFIX,
    }


@app.get("/ready")
async def ready():
    return {"status": "ready"}


# API routes
from app.api.router import api_router  # noqa: E402

app.include_router(api_router, prefix="/api/v1")

# Static files — document images extracted by ExploreRAG (Docling)
_docling_data = Path(__file__).resolve().parent.parent / "data" / "docling"
_docling_data.mkdir(parents=True, exist_ok=True)
app.mount("/static/doc-images", StaticFiles(directory=str(_docling_data)), name="static_doc_images")

# Import models so SQLAlchemy registers them
from app.models import knowledge_base, document, chat_message, chat_attachment, evaluation  # noqa: E402, F401
