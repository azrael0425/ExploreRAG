from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.core.config import settings
from app.schemas.workspace import WorkspaceCreate, WorkspaceUpdate
from app.services.knowledge_graph_service import (
    clear_knowledge_graph_service_cache_for_tests,
    get_knowledge_graph_service,
)
from app.services.llm import get_llm_provider


def test_workspace_schema_defaults_to_cloud_and_rejects_unknown_mode() -> None:
    assert WorkspaceCreate(name="default").llm_mode == "cloud"
    assert WorkspaceUpdate(llm_mode="local").llm_mode == "local"

    with pytest.raises(ValidationError):
        WorkspaceUpdate(llm_mode="automatic")


def test_provider_factory_builds_separate_cloud_and_strict_local_clients(monkeypatch) -> None:
    monkeypatch.setattr(settings, "LLM_PROVIDER", "dashscope")
    monkeypatch.setattr(settings, "DASHSCOPE_API_KEY", "test-cloud-key")
    monkeypatch.setattr(settings, "LOCAL_LLM_MODEL", "qwen3-vl:4b-instruct")
    monkeypatch.setattr(settings, "LOCAL_LLM_VISION_MODEL", "qwen3-vl:4b-instruct")
    monkeypatch.setattr(settings, "LOCAL_LLM_MAX_OUTPUT_TOKENS", 2048)
    get_llm_provider.cache_clear()

    cloud = get_llm_provider("cloud")
    local = get_llm_provider("local")

    assert cloud is get_llm_provider("cloud")
    assert local is get_llm_provider("local")
    assert cloud is not local
    assert cloud.provider_name == "dashscope"
    assert not cloud.is_local
    assert local.provider_name == "ollama"
    assert local.is_local
    assert local._strict_errors
    assert local._bounded_max_tokens(8192) == 2048
    get_llm_provider.cache_clear()


def test_cloud_provider_requires_an_explicit_test_credential(monkeypatch) -> None:
    monkeypatch.setattr(settings, "LLM_PROVIDER", "dashscope")
    monkeypatch.setattr(settings, "DASHSCOPE_API_KEY", "")

    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY is required"):
        get_llm_provider("cloud")


def test_local_healthcheck_requires_configured_model(monkeypatch) -> None:
    get_llm_provider.cache_clear()
    local = get_llm_provider("local")

    class _Models:
        async def list(self):
            return SimpleNamespace(data=[SimpleNamespace(id="another-model:latest")])

    local._async_client = SimpleNamespace(models=_Models())
    available, detail = asyncio.run(local.ahealthcheck())

    assert not available
    assert "qwen3-vl:4b-instruct" in (detail or "")
    get_llm_provider.cache_clear()


def test_light_rag_cache_is_replaced_when_workspace_mode_changes(monkeypatch) -> None:
    monkeypatch.setattr(settings, "LLM_PROVIDER", "dashscope")
    monkeypatch.setattr(settings, "DASHSCOPE_API_KEY", "test-cloud-key")
    clear_knowledge_graph_service_cache_for_tests()
    cloud = get_knowledge_graph_service(901, llm_mode="cloud")
    local = get_knowledge_graph_service(901, llm_mode="local")

    assert local is not cloud
    assert local.llm_mode == "local"
    assert get_knowledge_graph_service(901) is local
    clear_knowledge_graph_service_cache_for_tests()
