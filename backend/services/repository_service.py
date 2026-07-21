from __future__ import annotations

import os
from typing import Any

from backend.code_intelligence.repository_indexer import RepositoryIndex, index_repository

_REPOSITORIES: dict[str, RepositoryIndex] = {}


def ingest_repository(source_type: str, path: str, name: str | None = None) -> dict[str, Any]:
    if source_type not in {"folder", "git"}:
        raise ValueError("source_type must be 'folder' or 'git'")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")

    index = index_repository(path)
    index.name = name or index.name
    _REPOSITORIES[index.repo_id] = index
    return {
        "repo_id": index.repo_id,
        "status": "indexed",
        "name": index.name,
        "summary": index.summary,
        "file_count": len(index.files),
        "graph": index.graph,
    }


def get_repository(repo_id: str) -> RepositoryIndex | None:
    return _REPOSITORIES.get(repo_id)


def list_repositories() -> list[dict[str, Any]]:
    return [
        {"repo_id": index.repo_id, "name": index.name, "path": index.path, "summary": index.summary}
        for index in _REPOSITORIES.values()
    ]
