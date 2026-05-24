import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_search_empty():
    response = client.post(
        "/api/search",
        json={"query": "测试查询", "top_k": 5, "search_type": "hybrid"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "knowledge_points" in data


def test_stats():
    response = client.get("/api/stats")
    assert response.status_code == 200


def test_file_upload():
    with open(__file__, "rb") as f:
        response = client.post(
            "/api/ingest/file?file_type=text",
            files={"file": ("test.py", f, "text/x-python")},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("completed", "failed")