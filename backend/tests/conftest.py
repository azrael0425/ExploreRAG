"""Keep unit tests independent from a running PostgreSQL service.

Production imports create the async engine eagerly.  These focused service
tests use SQLAlchemy's declarative base only, so replace that module before
model imports while leaving application behavior untouched.
"""
from __future__ import annotations

import os
import sys
import types
from importlib.util import find_spec

import pytest
from sqlalchemy.orm import DeclarativeBase


# Environment variables take precedence over the repository-root .env file.
# Unit tests must never inherit a developer's real cloud credentials.
os.environ["DASHSCOPE_API_KEY"] = ""
os.environ["DEEPSEEK_API_KEY"] = ""
os.environ["DATABASE_URL"] = "postgresql+asyncpg://test:test@127.0.0.1:1/test"


class _TestBase(DeclarativeBase):
    pass


database = types.ModuleType("app.core.database")
database.Base = _TestBase
sys.modules.setdefault("app.core.database", database)

# The workspace's system Python intentionally has no web-runtime extras.  The
# unit tests below exercise pure attachment behavior, so provide the tiny
# FastAPI surface needed for imports when the real dependency is unavailable.
if find_spec("fastapi") is None:
    class HTTPException(Exception):
        def __init__(self, status_code: int = 500, detail: str = ""):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    fastapi = types.ModuleType("fastapi")
    fastapi.HTTPException = HTTPException
    fastapi.UploadFile = object
    fastapi.status = types.SimpleNamespace(
        HTTP_400_BAD_REQUEST=400,
        HTTP_404_NOT_FOUND=404,
        HTTP_409_CONFLICT=409,
        HTTP_422_UNPROCESSABLE_ENTITY=422,
    )
    responses = types.ModuleType("fastapi.responses")
    responses.StreamingResponse = object
    responses.FileResponse = object
    fastapi.responses = responses
    sys.modules["fastapi"] = fastapi
    sys.modules["fastapi.responses"] = responses


@pytest.fixture(autouse=True)
def _isolate_cached_model_providers():
    """Prevent provider instances and patched credentials leaking between tests."""
    from app.services.llm import get_embedding_provider, get_llm_provider

    get_llm_provider.cache_clear()
    get_embedding_provider.cache_clear()
    yield
    get_llm_provider.cache_clear()
    get_embedding_provider.cache_clear()
