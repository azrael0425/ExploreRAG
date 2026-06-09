import asyncio
from types import SimpleNamespace

from app.services.knowledge_graph_service import KnowledgeGraphService


def test_graph_export_batches_degree_calculation() -> None:
    class FakeStorage:
        node_degree_calls = 0

        async def get_knowledge_graph(self, **_kwargs):
            return SimpleNamespace(
                nodes=[
                    SimpleNamespace(id="Alpha", properties={"entity_type": "concept"}),
                    SimpleNamespace(id="Beta", properties={"entity_type": "concept"}),
                ],
                edges=[
                    SimpleNamespace(source="Alpha", target="Beta", properties={"weight": 2}),
                ],
                is_truncated=True,
            )

        async def get_all_edges(self):
            return [
                {"source": "Alpha", "target": "Beta"},
                {"source": "Alpha", "target": "Gamma"},
            ]

        async def node_degree(self, _node_id: str):
            self.node_degree_calls += 1
            raise AssertionError("node_degree should not be called when edges can be exported")

    storage = FakeStorage()
    service = KnowledgeGraphService(workspace_id=12)

    async def get_fake_rag():
        return SimpleNamespace(chunk_entity_relation_graph=storage)

    service._get_rag = get_fake_rag  # type: ignore[method-assign]

    result = asyncio.run(service.get_graph_data(max_nodes=300))

    assert {node["id"]: node["degree"] for node in result["nodes"]} == {
        "Alpha": 2,
        "Beta": 1,
    }
    assert storage.node_degree_calls == 0
    assert result["is_truncated"] is True


def test_focus_graph_keeps_low_degree_seeds_and_reports_missing_entities() -> None:
    class FakeStorage:
        async def get_all_nodes(self):
            return [
                {"id": "Popular", "entity_type": "concept"},
                {"id": "Rare cited entity", "entity_type": "concept"},
                {"id": "Neighbour", "entity_type": "concept"},
            ]

        async def get_all_edges(self):
            return [
                {
                    "source": "Popular",
                    "target": "Neighbour",
                    "weight": 5,
                },
                {
                    "source": "Rare cited entity",
                    "target": "Neighbour",
                    "weight": 1,
                    "source_id": "kb:12:doc:7",
                },
            ]

    service = KnowledgeGraphService(workspace_id=12)

    async def get_fake_rag():
        return SimpleNamespace(chunk_entity_relation_graph=FakeStorage())

    service._get_rag = get_fake_rag  # type: ignore[method-assign]
    result = asyncio.run(service.get_focus_graph_data(
        ["rare cited entity", "Missing"],
        document_ids=[7],
        max_depth=1,
        max_nodes=10,
    ))

    assert result["matched_entities"] == ["Rare cited entity"]
    assert result["missing_entities"] == ["Missing"]
    assert [node["id"] for node in result["nodes"]][:2] == [
        "Rare cited entity",
        "Neighbour",
    ]
    assert result["nodes"][0]["degree"] == 1
    assert result["edges"][0]["source_document_ids"] == [7]


def test_focus_graph_never_drops_seeds_when_the_requested_budget_is_smaller() -> None:
    class FakeStorage:
        async def get_all_nodes(self):
            return [
                {"id": f"Seed {index}", "entity_type": "concept"}
                for index in range(12)
            ]

        async def get_all_edges(self):
            return []

    service = KnowledgeGraphService(workspace_id=12)

    async def get_fake_rag():
        return SimpleNamespace(chunk_entity_relation_graph=FakeStorage())

    service._get_rag = get_fake_rag  # type: ignore[method-assign]
    requested = [f"Seed {index}" for index in range(12)]
    result = asyncio.run(service.get_focus_graph_data(
        requested,
        max_depth=1,
        max_nodes=10,
    ))

    assert result["matched_entities"] == requested
    assert len(result["nodes"]) == 12
