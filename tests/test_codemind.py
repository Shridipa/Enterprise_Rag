from __future__ import annotations

from backend.code_intelligence.repository_indexer import index_repository
from backend.main import app
from fastapi.testclient import TestClient


def test_repository_ingest_endpoint_accepts_folder(tmp_path):
    repo_dir = tmp_path / "sample_repo"
    repo_dir.mkdir()
    (repo_dir / "app").mkdir()
    (repo_dir / "app" / "main.py").write_text(
        "def login():\n    return 'ok'\n",
        encoding="utf-8",
    )
    (repo_dir / "README.md").write_text("# Sample repo\n", encoding="utf-8")

    client = TestClient(app)
    response = client.post(
        "/v1/repositories/ingest",
        json={"source_type": "folder", "path": str(repo_dir), "name": "sample_repo"},
        headers={"X-API-Key": "test-api-key-12345"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "indexed"
    assert data["repo_id"]
    assert data["summary"]["file_count"] >= 2


def test_architecture_explore_endpoint_returns_graph(tmp_path):
    repo_dir = tmp_path / "graph_repo"
    repo_dir.mkdir()
    (repo_dir / "src").mkdir()
    (repo_dir / "src" / "auth.py").write_text(
        "def authenticate():\n    return True\n",
        encoding="utf-8",
    )
    (repo_dir / "src" / "main.py").write_text(
        "from src.auth import authenticate\n\ndef run():\n    return authenticate()\n",
        encoding="utf-8",
    )

    client = TestClient(app)
    ingest_response = client.post(
        "/v1/repositories/ingest",
        json={"source_type": "folder", "path": str(repo_dir), "name": "graph_repo"},
        headers={"X-API-Key": "test-api-key-12345"},
    )
    repo_id = ingest_response.json()["repo_id"]

    response = client.get(
        f"/v1/architecture/explore/{repo_id}",
        headers={"X-API-Key": "test-api-key-12345"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data
    assert "edges" in data
    assert any(node["type"] == "file" for node in data["nodes"])


def test_cpp_repository_analysis_extracts_symbols_and_graph(tmp_path):
    repo_dir = tmp_path / "cpp_repo"
    repo_dir.mkdir()
    (repo_dir / "src").mkdir()
    (repo_dir / "src" / "engine.cpp").write_text(
        "#include <vector>\n\nclass Base {\npublic:\n    virtual ~Base() = default;\n};\n\nnamespace trading {\nclass Engine : public Base {\npublic:\n    void run();\n};\n\nvoid Engine::run() {\n}\n}\n",
        encoding="utf-8",
    )

    index = index_repository(repo_dir)

    assert index.summary["languages"] == ["cpp"]
    analysis = index.summary["analysis"]
    assert analysis["cpp"]["class_count"] >= 1
    assert analysis["cpp"]["namespace_count"] == 1
    assert analysis["cpp"]["function_count"] >= 1
    assert any(edge["type"] == "inheritance" for edge in index.graph["edges"])


def test_repository_details_endpoint_returns_analysis_payload(tmp_path):
    repo_dir = tmp_path / "details_repo"
    repo_dir.mkdir()
    (repo_dir / "src").mkdir()
    (repo_dir / "src" / "main.py").write_text(
        "def run():\n    return 'ok'\n",
        encoding="utf-8",
    )

    client = TestClient(app)
    ingest_response = client.post(
        "/v1/repositories/ingest",
        json={"source_type": "folder", "path": str(repo_dir), "name": "details_repo"},
        headers={"X-API-Key": "test-api-key-12345"},
    )
    repo_id = ingest_response.json()["repo_id"]

    response = client.get(
        f"/v1/repositories/{repo_id}",
        headers={"X-API-Key": "test-api-key-12345"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["repo_id"] == repo_id
    assert "analysis" in data["summary"]
