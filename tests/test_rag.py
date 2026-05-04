from fastapi.testclient import TestClient
from datetime import datetime
import uuid

from app.main import app
from app.core.database import get_db
from app.models.database import Query as QueryModel
from app.services.embedding import get_embedding_service
from app.services.vector_store import get_vector_store
from app.services.web_search import get_web_search_service
from app.api.routes import query as query_route
import pytest

client = TestClient(app)


class FakeRAGService:
    """RAG service fake that avoids external API and vector DB calls."""

    def answer_query(
        self,
        query: str,
        top_k: int = 5,
        use_web_search: bool = True,
        web_results: int = 3,
    ):
        return {
            "answer": f"Test answer for: {query}",
            "sources": [
                {
                    "content": "Artificial Intelligence is the simulation of human intelligence by machines.",
                    "metadata": {"filename": "test.txt", "chunk_index": 0},
                    "score": 0.9,
                    "source_type": "document",
                }
            ][:top_k],
            "query_time_ms": 12.5,
        }


class FakeQueryResult:
    def __init__(self, records):
        self.records = records

    def order_by(self, *args, **kwargs):
        return self

    def limit(self, limit: int):
        self.records = self.records[:limit]
        return self

    def all(self):
        return self.records


class FakeSession:
    def __init__(self):
        self.records = []

    def add(self, record):
        record.id = str(uuid.uuid4())
        record.created_at = datetime.now()
        self.records.append(record)

    def commit(self):
        pass

    def close(self):
        pass

    def query(self, model):
        assert model is QueryModel
        return FakeQueryResult(self.records)


@pytest.fixture(autouse=True)
def fake_dependencies(monkeypatch):
    fake_session = FakeSession()

    def override_get_db():
        yield fake_session

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_embedding_service] = lambda: object()
    app.dependency_overrides[get_vector_store] = lambda: object()
    app.dependency_overrides[get_web_search_service] = lambda: object()
    monkeypatch.setattr(
        query_route,
        "get_rag_service",
        lambda embedding_service, vector_store, web_search_service: FakeRAGService(),
    )

    yield

    app.dependency_overrides.clear()


def test_query_endpoint():
    """Test basic query functionality."""
    response = client.post(
        "/api/v1/query/",
        json={
            "query": "What is AI?",
            "top_k": 3,
            "include_sources": True
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "query_time_ms" in data
    assert len(data["sources"]) <= 3


def test_query_without_sources():
    """Test query without source inclusion."""
    response = client.post(
        "/api/v1/query/",
        json={
            "query": "What is machine learning?",
            "top_k": 5,
            "include_sources": False
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["sources"] == []


def test_query_history():
    """Test query history endpoint."""
    # Make a query first
    client.post(
        "/api/v1/query/",
        json={"query": "Test query", "top_k": 3}
    )
    
    # Check history
    response = client.get("/api/v1/query/history?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
