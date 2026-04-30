from fastapi.testclient import TestClient
from app.main import app
import pytest

client = TestClient(app)


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